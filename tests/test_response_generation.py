"""Offline invariants for the manifest-first Anthropic response runner."""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import pyarrow.parquet as pq
import yaml

from src.python.experiment_library.io import read_jsonl, write_jsonl
from src.python.experiment_library.response_generation import (
    _chunk_requests,
    _parse_choice,
    build_manifest,
    estimate_cost,
    parse,
    validate_run,
)


PROJECT_ROOT = Path.cwd()
CONFIG_PATH = PROJECT_ROOT / "configs" / "haiku_main_effects.yaml"


class ResponseGenerationTests(unittest.TestCase):
    """Keep the no-spend lifecycle and normalization rules deterministic."""

    def _small_config(self, directory: Path) -> Path:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        config["scenarios"]["ids"] = ["WVS_Q8_S01"]
        config["design"]["context"]["frames"] = ["F01"]
        config["design"]["context"]["history_lengths"] = [1]
        config["design"]["profiles"]["profiles"] = ["P001"]
        path = directory / "small.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path

    def test_build_manifest_is_idempotent_and_renders_native_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._small_config(root)
            run_id, paths = build_manifest(config_path, output_root=root)
            second_run_id, second_paths = build_manifest(config_path, output_root=root)
            self.assertEqual(run_id, second_run_id)
            self.assertEqual(paths, second_paths)
            manifest = read_jsonl(paths.manifest)
            requests = {row["custom_id"]: row for row in read_jsonl(paths.requests)}
            self.assertEqual(len(manifest), 120)
            self.assertEqual(len({row["custom_id"] for row in manifest}), 120)
            self.assertEqual({row["status"] for row in manifest}, {"pending"})
            baseline = next(row for row in manifest if row["experiment_arm"] == "baseline")
            rendered = requests[baseline["custom_id"]]
            self.assertIn("<known_user_profile>\nNONE\n</known_user_profile>", rendered["system"])
            self.assertIn("<frame_instructions>\nNONE\n</frame_instructions>", rendered["system"])
            self.assertEqual(rendered["messages"][-1]["role"], "user")
            self.assertTrue(all(message["role"] in {"user", "assistant"} for message in rendered["messages"]))
            validate_run(paths)

    def test_parser_preserves_raw_letter_and_normalizes_ba_to_canonical_option(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._small_config(root)
            run_id, paths = build_manifest(config_path, output_root=root)
            manifest = read_jsonl(paths.manifest)
            target = next(
                row for row in manifest
                if row["scenario_id"] == "WVS_Q8_S01" and row["option_order"] == "BA" and row["elicitation_id"] == "E01"
            )
            raw = {
                "batch_id": "msgbatch_test",
                "collected_at": "2026-08-14T00:00:00+00:00",
                "custom_id": target["custom_id"],
                "result": {
                    "type": "succeeded",
                    "message": {
                        "id": "msg_test", "model": "claude-haiku-4-5-20251001", "stop_reason": "end_turn",
                        "usage": {"input_tokens": 42, "output_tokens": 1},
                        "content": [{"type": "text", "text": "A."}],
                    },
                },
            }
            write_jsonl(paths.raw_results, [raw])
            self.assertEqual(parse(paths), len(manifest))
            rows = {row["request_id"]: row for row in pq.read_table(paths.responses).to_pylist()}
            parsed = rows[target["request_id"]]
            self.assertEqual(parsed["raw_response"], "A.")
            self.assertEqual(parsed["raw_choice"], "A")
            self.assertEqual(parsed["canonical_choice"], "B")
            self.assertEqual(parsed["selected_option_id"], "WVS_Q8_S01:B")
            self.assertEqual(parsed["selected_value_pole"], "following direction")
            self.assertTrue(parsed["valid_response"])
            self.assertEqual(parsed["batch_id"], "msgbatch_test")

    def test_cost_estimate_counts_each_unique_rendered_prompt_once(self) -> None:
        class TokenCounter:
            def __init__(self) -> None:
                self.calls = 0

            def count_tokens(self, **_: object) -> dict[str, int]:
                self.calls += 1
                return {"input_tokens": 9}

        class FakeClient:
            def __init__(self) -> None:
                self.messages = TokenCounter()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._small_config(root)
            _, paths = build_manifest(config_path, output_root=root)
            client = FakeClient()
            estimate = estimate_cost(paths, client=client)
            self.assertEqual(client.messages.calls, 12)
            self.assertEqual(estimate["unique_rendered_prompts"], 12)
            self.assertEqual(estimate["total_api_requests"], 120)
            self.assertEqual(estimate["estimated_total_input_tokens"], 1_080)
            self.assertEqual(estimate["number_of_batches_required"], 1)

    def test_strict_parser_and_batch_chunking(self) -> None:
        self.assertEqual(_parse_choice('"A"', False), ("A", True))
        self.assertEqual(_parse_choice("No preference.", True), ("NO_PREFERENCE", True))
        self.assertEqual(_parse_choice("I would choose A", False), ("INVALID", False))
        request = {
            "custom_id": "exp_1", "system": "system", "messages": [{"role": "user", "content": "prompt"}],
            "params": {"model": "claude-haiku-4-5-20251001", "temperature": 1.0, "max_tokens": 16},
        }
        chunks = _chunk_requests([copy.deepcopy(request) for _ in range(5)], count_limit=2, byte_limit=10_000)
        self.assertEqual([len(chunk) for chunk in chunks], [2, 2, 1])


if __name__ == "__main__":
    unittest.main()
