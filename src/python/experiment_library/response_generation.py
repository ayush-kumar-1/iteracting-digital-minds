"""Manifest-first response generation for the frozen experiment library.

This module deliberately stops at response generation and strict normalization.
It does not score WVS items or estimate utilities, effects, or significance.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Protocol

import yaml

from src.python.experiment_library.composition import BASE_INSTRUCTIONS
from src.python.experiment_library.io import read_jsonl, write_json, write_jsonl
from src.python.experiment_library.paths import ENGLISH_ROOT, PROJECT_ROOT
from src.python.experiment_library.validation import validate_library


PINNED_MODEL = "claude-haiku-4-5-20251001"
BATCH_INPUT_PRICE_PER_MILLION = 0.50
BATCH_OUTPUT_PRICE_PER_MILLION = 2.50
DEFAULT_BATCH_REQUEST_LIMIT = 90_000
DEFAULT_BATCH_BYTE_LIMIT = 240 * 1024 * 1024
NONE = "NONE"


class AnthropicClient(Protocol):
    """The small synchronous Anthropic SDK surface used by this runner."""

    messages: Any


@dataclass(frozen=True)
class RunPaths:
    """Stable artifact locations for one immutable intended experiment."""

    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.jsonl"

    @property
    def requests(self) -> Path:
        return self.root / "requests.jsonl"

    @property
    def metadata(self) -> Path:
        return self.root / "run_metadata.json"

    @property
    def source_hashes(self) -> Path:
        return self.root / "source_hashes.json"

    @property
    def batches(self) -> Path:
        return self.root / "batches.jsonl"

    @property
    def raw_results(self) -> Path:
        return self.root / "raw_results.jsonl"

    @property
    def responses(self) -> Path:
        return self.root / "responses.parquet"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _git_commit() -> str | None:
    """Return the checked-out commit without failing outside a Git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _plain(value: Any) -> Any:
    """Convert SDK objects to lossless JSON-compatible data where possible."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _record_index(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {record[key]: record for record in records}


def _load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Experiment config does not exist: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Experiment config must be a mapping.")
    return loaded


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Experiment config is missing {key!r}.")
    return mapping[key]


def validate_config(config: dict[str, Any]) -> None:
    """Validate the explicit phase-one configuration contract."""
    experiment = _required(config, "experiment")
    model = _required(config, "model")
    sampling = _required(config, "sampling")
    design = _required(config, "design")
    generation = _required(config, "generation")
    if not all(isinstance(section, dict) for section in (experiment, model, sampling, design, generation)):
        raise ValueError("experiment, model, sampling, design, and generation must be mappings.")
    if experiment.get("language") != "en":
        raise ValueError("Only the frozen English source is available in phase one.")
    if model.get("provider") != "anthropic":
        raise ValueError("Phase one supports the native Anthropic Messages API only.")
    if model.get("model") != PINNED_MODEL:
        raise ValueError(f"Phase one requires pinned model {PINNED_MODEL!r}.")
    if model.get("temperature") != 1.0 or model.get("max_tokens") != 16:
        raise ValueError("Primary preference experiments require temperature=1.0 and max_tokens=16.")
    if sampling.get("replicates_per_order") != 10:
        raise ValueError("Primary preference experiments require replicates_per_order=10.")
    if sampling.get("option_orders") != ["AB", "BA"]:
        raise ValueError("Primary preference experiments require option_orders: [AB, BA].")
    if generation.get("canonical_histories") is not True or generation.get("use_batch_api") is not True:
        raise ValueError("Phase one requires canonical histories and the Message Batches API.")
    interactions = design.get("interactions", {})
    if not isinstance(interactions, dict):
        raise ValueError("design.interactions must be a mapping.")
    if interactions.get("enabled") and interactions.get("dimensions", ["context_profile"]) != ["context_profile"]:
        raise ValueError("Phase one currently supports only the context_profile interaction dimension.")


def _materials() -> dict[str, dict[str, dict[str, Any]]]:
    within = read_jsonl(ENGLISH_ROOT / "wvs" / "scenarios.jsonl")
    cross = read_jsonl(ENGLISH_ROOT / "wvs" / "cross_value_scenarios.jsonl")
    return {
        "frames": _record_index(read_jsonl(ENGLISH_ROOT / "data" / "frames.jsonl"), "frame_id"),
        "histories": _record_index(read_jsonl(ENGLISH_ROOT / "data" / "histories.jsonl"), "history_id"),
        "profiles": _record_index(read_jsonl(ENGLISH_ROOT / "data" / "profiles.jsonl"), "profile_id"),
        "templates": _record_index(read_jsonl(ENGLISH_ROOT / "elicitation" / "templates.jsonl"), "elicitation_id"),
        "within_question": _record_index(within, "scenario_id"),
        "cross_value": _record_index(cross, "scenario_id"),
    }


def _select_ids(
    requested: str | list[str], available: Iterable[str], label: str
) -> list[str]:
    choices = sorted(available)
    selected = choices if requested == "all" else list(requested)
    unknown = sorted(set(selected).difference(choices))
    if unknown:
        raise ValueError(f"Unknown {label}: {', '.join(unknown)}")
    return selected


def _scenario_records(config: dict[str, Any], materials: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    included = _required(_required(config, "scenarios"), "include")
    records: list[tuple[str, dict[str, Any]]] = []
    for family in included:
        if family not in {"within_question", "cross_value"}:
            raise ValueError(f"Unsupported scenario family: {family}")
        records.extend((family, record) for record in materials[family].values())
    requested_ids = config["scenarios"].get("ids")
    if requested_ids is not None:
        available_ids = {record["scenario_id"] for _, record in records}
        unknown = sorted(set(requested_ids).difference(available_ids))
        if unknown:
            raise ValueError(f"Unknown scenario IDs: {', '.join(unknown)}")
        records = [pair for pair in records if pair[1]["scenario_id"] in requested_ids]
    if not records:
        raise ValueError("Scenario selection is empty.")
    return sorted(records, key=lambda pair: pair[1]["scenario_id"])


def _condition_specs(config: dict[str, Any], materials: dict[str, Any]) -> list[dict[str, Any]]:
    """Build unique scientific cells, with explicit absent-manipulation controls."""
    design = config["design"]
    language = config["experiment"]["language"]
    scenarios = _scenario_records(config, materials)
    orders = config["sampling"]["option_orders"]
    specs: list[dict[str, Any]] = []

    def add(
        *, frame_id: str | None, history_length: int, profile_id: str | None,
        elicitation_id: str, experiment_arm: str,
    ) -> None:
        history_id = None if frame_id is None else f"{frame_id}_H{history_length}_R01"
        for family, scenario in scenarios:
            for order in orders:
                specs.append({
                    "experiment_arm": experiment_arm,
                    "frame_id": frame_id,
                    "history_id": history_id,
                    "history_length": history_length,
                    "profile_id": profile_id,
                    "scenario_id": scenario["scenario_id"],
                    "wvs_item_id": scenario["wvs_item_id"],
                    "scenario_family": family,
                    "elicitation_id": elicitation_id,
                    "option_order": order,
                    "language": language,
                })

    if design.get("baseline"):
        add(frame_id=None, history_length=0, profile_id=None, elicitation_id="E01", experiment_arm="baseline")
    context = design.get("context", {})
    if context.get("enabled"):
        for frame_id in _select_ids(context.get("frames", "all"), materials["frames"], "frame IDs"):
            for history_length in context.get("history_lengths", []):
                history_id = f"{frame_id}_H{history_length}_R01"
                if history_id not in materials["histories"]:
                    raise ValueError(f"No canonical history exists for {frame_id} at H{history_length}.")
                add(frame_id=frame_id, history_length=history_length, profile_id=None, elicitation_id="E01", experiment_arm="context")
    profiles = design.get("profiles", {})
    if profiles.get("enabled"):
        for profile_id in _select_ids(profiles.get("profiles", "all"), materials["profiles"], "profile IDs"):
            add(frame_id=None, history_length=0, profile_id=profile_id, elicitation_id="E01", experiment_arm="profile")
    elicitation = design.get("elicitation", {})
    if elicitation.get("enabled"):
        for elicitation_id in _select_ids(elicitation.get("templates", "all"), materials["templates"], "elicitation IDs"):
            if elicitation_id != "E01":
                add(frame_id=None, history_length=0, profile_id=None, elicitation_id=elicitation_id, experiment_arm="elicitation")
    interactions = design.get("interactions", {})
    if interactions.get("enabled"):
        context = design.get("context", {})
        profiles = design.get("profiles", {})
        if not context.get("enabled") or not profiles.get("enabled"):
            raise ValueError("context_profile interactions require enabled context and profile designs.")
        for frame_id in _select_ids(context.get("frames", "all"), materials["frames"], "frame IDs"):
            for history_length in context.get("history_lengths", []):
                for profile_id in _select_ids(profiles.get("profiles", "all"), materials["profiles"], "profile IDs"):
                    add(
                        frame_id=frame_id, history_length=history_length, profile_id=profile_id,
                        elicitation_id="E01", experiment_arm="context_profile",
                    )

    deduplicated = {_canonical_json(spec): spec for spec in specs}
    return [deduplicated[key] for key in sorted(deduplicated)]


def _render_template(template: str, replacements: dict[str, str]) -> str:
    for key, value in replacements.items():
        template = template.replace("{{" + key + "}}", value)
    if "{{" in template or "}}" in template:
        raise ValueError(f"Unresolved placeholder in elicitation template: {template!r}")
    return template


def _validate_canonical_history(history: dict[str, Any]) -> None:
    messages = history.get("messages")
    if history.get("history_mode") != "canonical":
        raise ValueError(f"{history.get('history_id')} is not canonical.")
    if history.get("profile_placeholders"):
        raise ValueError(f"{history.get('history_id')} contains unresolved profile placeholders.")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{history.get('history_id')} has no canonical messages.")
    expected_role = "user"
    for message in messages:
        content = message.get("content")
        if (
            message.get("role") != expected_role or not isinstance(content, str) or not content.strip()
            or "{{" in content or "}}" in content
        ):
            raise ValueError(f"{history.get('history_id')} has a missing, placeholder, or non-alternating turn.")
        expected_role = "assistant" if expected_role == "user" else "user"
    if messages[-1]["role"] != "assistant":
        raise ValueError(f"{history.get('history_id')} must end with a canonical assistant turn.")


def render_anthropic_request(spec: dict[str, Any], materials: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    """Render immutable native-Messages inputs from a scientific condition."""
    frame_id, profile_id = spec["frame_id"], spec["profile_id"]
    frame = None if frame_id is None else materials["frames"][frame_id]
    profile = None if profile_id is None else materials["profiles"][profile_id]
    history = None if spec["history_id"] is None else materials["histories"][spec["history_id"]]
    scenario = materials[spec["scenario_family"]][spec["scenario_id"]]
    template = materials["templates"][spec["elicitation_id"]]
    if history is not None:
        _validate_canonical_history(history)
        if history["frame_id"] != frame_id or history["history_length"] != spec["history_length"]:
            raise ValueError("Requested canonical history does not match its frame or length.")
    if frame is None and spec["history_length"] != 0:
        raise ValueError("history=NONE must use history_length=0.")
    if spec["option_order"] == "AB":
        option_a, option_b = scenario["option_a"], scenario["option_b"]
    else:
        option_a, option_b = scenario["option_b"], scenario["option_a"]
    final_elicitation = _render_template(
        template["prompt_template"],
        {"option_a_label": "A", "option_b_label": "B", "option_a": option_a, "option_b": option_b},
    )
    profile_text = NONE if profile is None else profile["rendered_profile"]
    frame_text = NONE if frame is None or not frame["developer_instruction_required"] else frame["developer_instruction"]
    system = (
        f"{BASE_INSTRUCTIONS}\n\n"
        f"<known_user_profile>\n{profile_text}\n</known_user_profile>\n\n"
        f"<frame_instructions>\n{frame_text}\n</frame_instructions>"
    )
    messages = ([] if history is None else history["messages"].copy()) + [
        {"role": "user", "content": f"{scenario['context']}\n\n{final_elicitation}"}
    ]
    _validate_messages(messages)
    params = {"model": model["model"], "temperature": model["temperature"], "max_tokens": model["max_tokens"]}
    return {"system": system, "messages": messages, "params": params}


def _validate_messages(messages: list[dict[str, str]]) -> None:
    expected_role = "user"
    for message in messages:
        if message.get("role") != expected_role:
            raise ValueError("Anthropic conversation history must alternate user and assistant turns.")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError("Every rendered message must have non-empty string content.")
        expected_role = "assistant" if expected_role == "user" else "user"
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("Every rendered request must end in the final user elicitation.")


def _component_hashes(config_path: Path) -> dict[str, str]:
    source_paths = [
        config_path,
        ENGLISH_ROOT / "data" / "frames.jsonl",
        ENGLISH_ROOT / "data" / "histories.jsonl",
        ENGLISH_ROOT / "data" / "profiles.jsonl",
        ENGLISH_ROOT / "wvs" / "scenarios.jsonl",
        ENGLISH_ROOT / "wvs" / "cross_value_scenarios.jsonl",
        ENGLISH_ROOT / "elicitation" / "templates.jsonl",
    ]
    hashes: dict[str, str] = {}
    for path in source_paths:
        try:
            label = str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            label = str(path)
        hashes[label] = _sha256(path.read_bytes())
    return hashes


def _config_reference(config_path: Path) -> str:
    try:
        return str(config_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(config_path)


def _run_id(config: dict[str, Any], source_hashes: dict[str, str]) -> str:
    name = re.sub(r"[^a-z0-9-]+", "-", config["experiment"]["name"].lower()).strip("-")
    fingerprint = _sha256(_canonical_json({"config": config, "sources": source_hashes}))[:12]
    return f"{name}-{fingerprint}"


def run_paths(run_id: str, output_root: Path | None = None) -> RunPaths:
    root = (PROJECT_ROOT / "Output" / "experiments" if output_root is None else output_root) / run_id
    return RunPaths(root=root)


def _write_immutable_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    serialized = "".join(_canonical_json(record) + "\n" for record in records)
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise ValueError(f"Immutable artifact already exists with different contents: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized, encoding="utf-8")


def build_manifest(config_path: Path, requested_run_id: str | None = None, output_root: Path | None = None) -> tuple[str, RunPaths]:
    """Create the immutable manifest and exact rendered-request artifact."""
    config_path = config_path.resolve()
    config = _load_yaml_config(config_path)
    validate_config(config)
    library_errors = validate_library()
    if library_errors:
        raise ValueError("Frozen source-material validation failed:\n" + "\n".join(library_errors))
    hashes = _component_hashes(config_path)
    run_id = requested_run_id or _run_id(config, hashes)
    paths = run_paths(run_id, output_root)
    materials = _materials()
    manifest: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for spec in _condition_specs(config, materials):
        rendered = render_anthropic_request(spec, materials, config["model"])
        prompt_sha256 = _sha256(_canonical_json({"system": rendered["system"], "messages": rendered["messages"], "params": rendered["params"]}))
        scientific_payload = {
            "dataset_version": config["experiment"].get("dataset_version"),
            "spec": spec,
            "model": rendered["params"],
            "source_hashes": hashes,
        }
        condition_id = "cond_" + _sha256(_canonical_json(scientific_payload))[:24]
        for replicate in range(1, config["sampling"]["replicates_per_order"] + 1):
            request_payload = {"condition_id": condition_id, "replicate": replicate, "config": rendered["params"]}
            request_id = "req_" + _sha256(_canonical_json(request_payload))[:28]
            custom_id = "exp_" + _sha256(_canonical_json({"dataset": scientific_payload, "replicate": replicate}))[:28]
            manifest.append({
                "request_id": request_id,
                "condition_id": condition_id,
                "replicate": replicate,
                **spec,
                "model": rendered["params"]["model"],
                "temperature": rendered["params"]["temperature"],
                "max_tokens": rendered["params"]["max_tokens"],
                "prompt_sha256": prompt_sha256,
                "custom_id": custom_id,
                "status": "pending",
            })
            requests.append({"custom_id": custom_id, **rendered})
    _assert_manifest_integrity(manifest, requests)
    manifest.sort(key=lambda record: record["request_id"])
    requests.sort(key=lambda record: record["custom_id"])
    _write_immutable_jsonl(paths.manifest, manifest)
    _write_immutable_jsonl(paths.requests, requests)
    if not paths.source_hashes.exists():
        write_json(paths.source_hashes, hashes)
    elif json.loads(paths.source_hashes.read_text(encoding="utf-8")) != hashes:
        raise ValueError("Immutable source-hash artifact differs from current source materials.")
    if not paths.metadata.exists():
        write_json(paths.metadata, {
            "run_id": run_id,
            "created_at": _utc_now(),
            "git_commit": _git_commit(),
            "anthropic_sdk_version": importlib.metadata.version("anthropic"),
            "experiment_config": _config_reference(config_path),
            "pinned_model": PINNED_MODEL,
            "source_hashes": hashes,
            "request_count": len(manifest),
            "scientific_condition_count": len({row["condition_id"] for row in manifest}),
        })
    return run_id, paths


def _assert_manifest_integrity(manifest: list[dict[str, Any]], requests: list[dict[str, Any]]) -> None:
    if not manifest or len(manifest) != len(requests):
        raise ValueError("Manifest and requests must have the same non-zero number of rows.")
    for field in ("request_id", "custom_id"):
        values = [record[field] for record in manifest]
        if len(values) != len(set(values)):
            raise ValueError(f"Manifest has non-unique {field}s.")
    request_ids = {record["custom_id"] for record in requests}
    if request_ids != {record["custom_id"] for record in manifest}:
        raise ValueError("Manifest custom_ids do not exactly match rendered requests.")
    for record in manifest:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", record["custom_id"]):
            raise ValueError(f"Anthropic custom_id is invalid: {record['custom_id']}")


def load_run(paths: RunPaths) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest = read_jsonl(paths.manifest)
    requests = _record_index(read_jsonl(paths.requests), "custom_id")
    _assert_manifest_integrity(manifest, list(requests.values()))
    return manifest, requests


def validate_run(paths: RunPaths) -> None:
    """Validate immutable artifacts and all rendered Anthropic role invariants."""
    manifest, requests = load_run(paths)
    for row in manifest:
        request = requests[row["custom_id"]]
        if request["params"] != {"model": row["model"], "temperature": row["temperature"], "max_tokens": row["max_tokens"]}:
            raise ValueError(f"Request parameters disagree with manifest for {row['request_id']}.")
        if _sha256(_canonical_json({"system": request["system"], "messages": request["messages"], "params": request["params"]})) != row["prompt_sha256"]:
            raise ValueError(f"Prompt hash disagrees for {row['request_id']}.")
        _validate_messages(request["messages"])
        if row["history_id"] is None and row["history_length"] != 0:
            raise ValueError("history=NONE has nonzero history_length.")
        if row["history_id"] is not None and not row["history_id"].startswith(f"{row['frame_id']}_H"):
            raise ValueError("history_id does not belong to frame_id.")


def _load_client() -> AnthropicClient:
    """Read the existing read-only secret at execution time, never into artifacts."""
    secrets_path = PROJECT_ROOT / "secrets.json"
    if not secrets_path.is_file():
        raise ValueError("secrets.json is required for Anthropic API operations.")
    secret_values = json.loads(secrets_path.read_text(encoding="utf-8"))
    if not isinstance(secret_values, dict):
        raise ValueError("secrets.json must be a JSON object.")
    api_key = next((secret_values.get(key) for key in ("ANTHROPIC_API_KEY", "anthropic_api_key", "api_key") if secret_values.get(key)), None)
    if not isinstance(api_key, str):
        raise ValueError("secrets.json has no Anthropic API key under an accepted key name.")
    from anthropic import Anthropic

    return Anthropic(api_key=api_key)


def _counts(records: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(NONE if record.get(field) is None else str(record.get(field)) for record in records)
    return dict(sorted(counts.items()))


def _chunk_requests(requests: Iterable[dict[str, Any]], count_limit: int, byte_limit: int) -> list[list[dict[str, Any]]]:
    """Chunk serialized native batch requests below count and byte safety limits."""
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    for request in requests:
        batch_request = {"custom_id": request["custom_id"], "params": {**request["params"], "system": request["system"], "messages": request["messages"]}}
        request_bytes = len(_canonical_json(batch_request).encode("utf-8")) + 1
        if request_bytes > byte_limit:
            raise ValueError(f"A single request exceeds the configured batch byte limit: {request['custom_id']}")
        if current and (len(current) >= count_limit or current_bytes + request_bytes > byte_limit):
            chunks.append(current)
            current, current_bytes = [], 0
        current.append(request)
        current_bytes += request_bytes
    if current:
        chunks.append(current)
    return chunks


def estimate_cost(paths: RunPaths, client: AnthropicClient | None = None) -> dict[str, Any]:
    """Count unique exact prompts with Anthropic, then calculate batch estimates."""
    validate_run(paths)
    manifest, requests = load_run(paths)
    client = client or _load_client()
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        by_prompt[row["prompt_sha256"]].append(row)
    counts_by_prompt: dict[str, int] = {}
    for prompt_hash, rows in by_prompt.items():
        request = requests[rows[0]["custom_id"]]
        response = client.messages.count_tokens(
            model=request["params"]["model"], system=request["system"], messages=request["messages"]
        )
        counts_by_prompt[prompt_hash] = int(_plain(response)["input_tokens"])
    total_input_tokens = sum(counts_by_prompt[row["prompt_sha256"]] for row in manifest)
    max_prompt_tokens = max(counts_by_prompt.values())
    max_output_tokens = sum(row["max_tokens"] for row in manifest)
    batch_config = _load_run_config(paths)
    chunks = _chunk_requests(
        (requests[row["custom_id"]] for row in manifest),
        int(batch_config["generation"].get("batch_request_limit", DEFAULT_BATCH_REQUEST_LIMIT)),
        int(batch_config["generation"].get("batch_byte_limit", DEFAULT_BATCH_BYTE_LIMIT)),
    )
    report = {
        "run_id": paths.root.name,
        "estimated_at": _utc_now(),
        "unique_scientific_conditions": len({row["condition_id"] for row in manifest}),
        "total_api_requests": len(manifest),
        "requests_by_frame": _counts(manifest, "frame_id"),
        "requests_by_history_length": _counts(manifest, "history_length"),
        "requests_by_profile": _counts(manifest, "profile_id"),
        "requests_by_wvs_item": _counts(manifest, "wvs_item_id"),
        "requests_by_elicitation_template": _counts(manifest, "elicitation_id"),
        "requests_by_option_order": _counts(manifest, "option_order"),
        "unique_rendered_prompts": len(by_prompt),
        "estimated_total_input_tokens": total_input_tokens,
        "maximum_prompt_tokens": max_prompt_tokens,
        "maximum_output_tokens": max_output_tokens,
        "estimated_batch_cost_usd": round(
            total_input_tokens / 1_000_000 * BATCH_INPUT_PRICE_PER_MILLION
            + max_output_tokens / 1_000_000 * BATCH_OUTPUT_PRICE_PER_MILLION,
            6,
        ),
        "estimated_output_cost_basis": "max_tokens for every request; actual output cost will use collected usage",
        "batch_request_limit": int(batch_config["generation"].get("batch_request_limit", DEFAULT_BATCH_REQUEST_LIMIT)),
        "batch_byte_limit": int(batch_config["generation"].get("batch_byte_limit", DEFAULT_BATCH_BYTE_LIMIT)),
        "number_of_batches_required": len(chunks),
    }
    write_json(paths.root / "cost_estimate.json", report)
    return report


def _load_run_config(paths: RunPaths) -> dict[str, Any]:
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    return _load_yaml_config(PROJECT_ROOT / metadata["experiment_config"])


def _smoke_selection(manifest: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Deterministically choose heterogeneous rows without becoming observations."""
    fields = ("frame_id", "history_length", "profile_id", "elicitation_id", "option_order")
    uncovered = {(field, NONE if row.get(field) is None else str(row.get(field))) for row in manifest for field in fields}
    remaining = sorted(manifest, key=lambda row: row["request_id"])
    selected: list[dict[str, Any]] = []
    while remaining and len(selected) < limit:
        row = max(
            remaining,
            key=lambda candidate: sum(
                (field, NONE if candidate.get(field) is None else str(candidate.get(field))) in uncovered
                for field in fields
            ),
        )
        selected.append(row)
        remaining.remove(row)
        uncovered.difference_update((field, NONE if row.get(field) is None else str(row.get(field))) for field in fields)
    return selected


def smoke_test(paths: RunPaths, client: AnthropicClient | None = None, limit: int | None = None) -> dict[str, Any]:
    """Issue a small synchronous format/authentication test outside observations."""
    validate_run(paths)
    manifest, requests = load_run(paths)
    config = _load_run_config(paths)
    selected = _smoke_selection(manifest, limit or int(config["generation"].get("smoke_test_requests", 40)))
    client = client or _load_client()
    records: list[dict[str, Any]] = []
    for row in selected:
        request = requests[row["custom_id"]]
        message = client.messages.create(system=request["system"], messages=request["messages"], **request["params"])
        records.append({"request_id": row["request_id"], "custom_id": row["custom_id"], "response": _plain(message)})
    artifact = paths.root / "smoke_test_results.jsonl"
    _write_immutable_jsonl(artifact, records)
    summary = {"run_id": paths.root.name, "request_count": len(records), "artifact": str(artifact), "completed_at": _utc_now()}
    write_json(paths.root / "smoke_test_summary.json", summary)
    return summary


def _submitted_custom_ids(paths: RunPaths, client: AnthropicClient) -> set[str]:
    if not paths.batches.exists():
        return set()
    in_flight: set[str] = set()
    for batch in read_jsonl(paths.batches):
        status = _plain(client.messages.batches.retrieve(batch["batch_id"]))
        if status.get("processing_status") != "ended":
            in_flight.update(batch["custom_ids"])
    return in_flight


def _successful_custom_ids(paths: RunPaths) -> set[str]:
    if not paths.raw_results.exists():
        return set()
    return {
        record["custom_id"] for record in read_jsonl(paths.raw_results)
        if record.get("result", {}).get("type") == "succeeded"
    }


def submit(paths: RunPaths, client: AnthropicClient | None = None) -> list[dict[str, Any]]:
    """Submit only observations that are neither completed nor already in flight."""
    validate_run(paths)
    manifest, requests = load_run(paths)
    config = _load_run_config(paths)
    client = client or _load_client()
    completed = _successful_custom_ids(paths)
    in_flight = _submitted_custom_ids(paths, client)
    pending = [requests[row["custom_id"]] for row in manifest if row["custom_id"] not in completed | in_flight]
    chunks = _chunk_requests(
        pending,
        int(config["generation"].get("batch_request_limit", DEFAULT_BATCH_REQUEST_LIMIT)),
        int(config["generation"].get("batch_byte_limit", DEFAULT_BATCH_BYTE_LIMIT)),
    )
    submitted: list[dict[str, Any]] = []
    paths.root.mkdir(parents=True, exist_ok=True)
    with paths.batches.open("a", encoding="utf-8") as handle:
        for chunk in chunks:
            api_requests = [
                {"custom_id": request["custom_id"], "params": {**request["params"], "system": request["system"], "messages": request["messages"]}}
                for request in chunk
            ]
            batch = _plain(client.messages.batches.create(requests=api_requests))
            record = {
                "run_id": paths.root.name,
                "batch_id": batch["id"],
                "submitted_at": _utc_now(),
                "request_count": len(chunk),
                "custom_ids": [request["custom_id"] for request in chunk],
                "request_ids": [row["request_id"] for row in manifest if row["custom_id"] in {request["custom_id"] for request in chunk}],
                "api_submission": batch,
            }
            handle.write(_canonical_json(record) + "\n")
            handle.flush()
            submitted.append(record)
    return submitted


def status(paths: RunPaths, client: AnthropicClient | None = None) -> list[dict[str, Any]]:
    """Retrieve the current status of every persisted batch identifier."""
    client = client or _load_client()
    statuses = []
    for batch in read_jsonl(paths.batches) if paths.batches.exists() else []:
        response = _plain(client.messages.batches.retrieve(batch["batch_id"]))
        statuses.append({"batch_id": batch["batch_id"], "retrieved_at": _utc_now(), "status": response})
    write_json(paths.root / "batch_status.json", statuses)
    return statuses


def collect(paths: RunPaths, client: AnthropicClient | None = None) -> int:
    """Append unseen, complete batch results as the immutable raw source dataset."""
    client = client or _load_client()
    existing = {record["custom_id"] for record in read_jsonl(paths.raw_results)} if paths.raw_results.exists() else set()
    added = 0
    with paths.raw_results.open("a", encoding="utf-8") as handle:
        for batch in read_jsonl(paths.batches) if paths.batches.exists() else []:
            batch_status = _plain(client.messages.batches.retrieve(batch["batch_id"]))
            if batch_status.get("processing_status") != "ended":
                continue
            for result in client.messages.batches.results(batch["batch_id"]):
                raw_result = _plain(result)
                custom_id = raw_result["custom_id"]
                if custom_id in existing:
                    continue
                handle.write(_canonical_json({
                    "batch_id": batch["batch_id"], "collected_at": _utc_now(), "custom_id": custom_id,
                    "result": raw_result["result"],
                }) + "\n")
                existing.add(custom_id)
                added += 1
    return added


def _text_content(message: dict[str, Any]) -> str:
    return "".join(block.get("text", "") for block in message.get("content", []) if block.get("type") == "text")


def _parse_choice(text: str, allows_indifference: bool) -> tuple[str, bool]:
    """Apply the predeclared strict parser, never an inferential language model."""
    normalized = text.strip()
    normalized = re.sub(r"^[\"']|[\"']$", "", normalized).strip()
    normalized = re.sub(r"[.]$", "", normalized).strip()
    if normalized in {"A", "B"}:
        return normalized, True
    if allows_indifference and normalized.casefold() == "no preference":
        return "NO_PREFERENCE", True
    return "INVALID", False


def _parsed_response(row: dict[str, Any], raw: dict[str, Any] | None, materials: dict[str, Any]) -> dict[str, Any]:
    scenario = materials[row["scenario_family"]][row["scenario_id"]]
    template = materials["templates"][row["elicitation_id"]]
    result = None if raw is None else raw.get("result", {})
    message = result.get("message", {}) if result and result.get("type") == "succeeded" else {}
    raw_response = _text_content(message)
    raw_choice, valid_response = _parse_choice(raw_response, bool(template["allows_indifference"])) if message else ("INVALID", False)
    selected_option_id = selected_value_pole = canonical_choice = None
    if valid_response and raw_choice in {"A", "B"}:
        presented_to_canonical = {"A": "A", "B": "B"} if row["option_order"] == "AB" else {"A": "B", "B": "A"}
        canonical_choice = presented_to_canonical[raw_choice]
        selected_option_id = f"{row['scenario_id']}:{canonical_choice}"
        selected_value_pole = scenario[f"option_{canonical_choice.lower()}_pole"]
    elif valid_response:
        canonical_choice = "NO_PREFERENCE"
    usage = message.get("usage", {})
    error = result.get("error", {}) if result else {}
    return {
        "request_id": row["request_id"], "condition_id": row["condition_id"], "replicate": row["replicate"],
        "frame_id": row["frame_id"], "history_id": row["history_id"], "history_length": row["history_length"], "profile_id": row["profile_id"],
        "wvs_item_id": row["wvs_item_id"], "scenario_id": row["scenario_id"], "scenario_family": row["scenario_family"],
        "elicitation_id": row["elicitation_id"], "option_order": row["option_order"],
        "raw_response": raw_response, "raw_choice": raw_choice, "canonical_choice": canonical_choice,
        "valid_response": valid_response, "selected_option_id": selected_option_id, "selected_value_pole": selected_value_pole,
        "model": message.get("model"), "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
        "stop_reason": message.get("stop_reason"), "batch_id": None if raw is None else raw.get("batch_id"),
        "api_message_id": message.get("id"), "error_type": error.get("type") or ("not_collected" if raw is None else None),
    }


def parse(paths: RunPaths) -> int:
    """Deterministically produce exactly one normalized row per manifest request."""
    validate_run(paths)
    manifest, _ = load_run(paths)
    raw_by_id = _record_index(read_jsonl(paths.raw_results), "custom_id") if paths.raw_results.exists() else {}
    materials = _materials()
    rows = [_parsed_response(row, raw_by_id.get(row["custom_id"]), materials) for row in sorted(manifest, key=lambda item: item["request_id"])]
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        raise RuntimeError("Parsing requires pyarrow; install it with `uv add pyarrow`.") from error
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, paths.responses, compression="zstd")
    return len(rows)


def summarize_run(paths: RunPaths) -> dict[str, Any]:
    """Summarize operational provenance only; do not analyze preferences."""
    manifest, _ = load_run(paths)
    raw = read_jsonl(paths.raw_results) if paths.raw_results.exists() else []
    successful = sum(record.get("result", {}).get("type") == "succeeded" for record in raw)
    errors = Counter(
        record.get("result", {}).get("error", {}).get("type", "unknown")
        for record in raw if record.get("result", {}).get("type") != "succeeded"
    )
    usage = [record["result"]["message"].get("usage", {}) for record in raw if record.get("result", {}).get("type") == "succeeded"]
    summary = {
        "run_id": paths.root.name, "intended_requests": len(manifest), "collected_records": len(raw),
        "successful_requests": successful, "non_success_result_types": dict(sorted(errors.items())),
        "actual_input_tokens": sum(item.get("input_tokens", 0) for item in usage),
        "actual_output_tokens": sum(item.get("output_tokens", 0) for item in usage),
        "raw_results": str(paths.raw_results), "responses": str(paths.responses) if paths.responses.exists() else None,
    }
    summary["actual_batch_cost_usd"] = round(
        summary["actual_input_tokens"] / 1_000_000 * BATCH_INPUT_PRICE_PER_MILLION
        + summary["actual_output_tokens"] / 1_000_000 * BATCH_OUTPUT_PRICE_PER_MILLION,
        6,
    )
    write_json(paths.root / "run_summary.json", summary)
    return summary


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-manifest")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--run-id")
    for name in ("validate", "estimate-cost", "smoke-test", "submit", "status", "collect", "parse", "summarize-run"):
        command = commands.add_parser(name)
        command.add_argument("--run", required=True)
        if name == "smoke-test":
            command.add_argument("--limit", type=int)
    return parser


def main() -> None:
    """Expose the run lifecycle as one root-executed module command."""
    args = _cli().parse_args()
    if args.command == "build-manifest":
        run_id, paths = build_manifest(args.config, args.run_id)
        print(json.dumps({"run_id": run_id, "run_directory": str(paths.root)}, indent=2))
        return
    paths = run_paths(args.run)
    if args.command == "validate":
        validate_run(paths)
        result: Any = {"run_id": args.run, "valid": True}
    elif args.command == "estimate-cost":
        result = estimate_cost(paths)
    elif args.command == "smoke-test":
        result = smoke_test(paths, limit=args.limit)
    elif args.command == "submit":
        result = {"submitted_batches": [record["batch_id"] for record in submit(paths)]}
    elif args.command == "status":
        result = status(paths)
    elif args.command == "collect":
        result = {"new_raw_results": collect(paths)}
    elif args.command == "parse":
        result = {"normalized_rows": parse(paths), "responses": str(paths.responses)}
    else:
        result = summarize_run(paths)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
