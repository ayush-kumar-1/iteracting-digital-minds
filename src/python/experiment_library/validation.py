"""Structural and compositional checks for the experimental-material library."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import json
from pathlib import Path
from typing import Any

from .composition import compose_experiment
from .io import read_jsonl
from .paths import ENGLISH_ROOT


def _unique(records: list[dict[str, Any]], key: str, label: str, errors: list[str]) -> None:
    values = [record[key] for record in records]
    duplicates = [value for value, count in Counter(values).items() if count > 1]
    if duplicates:
        errors.append(f"{label} has duplicate {key}s: {duplicates}")


def _connected(nodes: list[str], edges: list[list[str]]) -> bool:
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    visited, queue = set(), deque([nodes[0]])
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            queue.extend(adjacency[node] - visited)
    return visited == set(nodes)


def _check_schema_required_fields(
    root: Path,
    sources: list[tuple[str, list[dict[str, Any]], str]],
    errors: list[str],
) -> None:
    """Validate the required fields from local JSON Schemas without extra dependencies."""
    schema_root = root.parent / "schemas"
    for schema_name, records, label in sources:
        schema = json.loads((schema_root / schema_name).read_text(encoding="utf-8"))
        required = set(schema["required"])
        for record in records:
            missing = sorted(required.difference(record))
            if missing:
                errors.append(f"{label} record is missing schema-required fields: {missing}")


def validate_library(root: Path = ENGLISH_ROOT) -> list[str]:
    """Return all structural QA failures; an empty list means the library is valid."""
    errors: list[str] = []
    frames = read_jsonl(root / "data" / "frames.jsonl")
    histories = read_jsonl(root / "data" / "histories.jsonl")
    profiles = read_jsonl(root / "data" / "profiles.jsonl")
    questions = read_jsonl(root / "wvs" / "questions.jsonl")
    scenarios = read_jsonl(root / "wvs" / "scenarios.jsonl")
    cross_scenarios = read_jsonl(root / "wvs" / "cross_value_scenarios.jsonl")
    templates = read_jsonl(root / "elicitation" / "templates.jsonl")
    graph = json.loads((root / "wvs" / "comparison_graph.json").read_text(encoding="utf-8"))
    _check_schema_required_fields(root, [
        ("frame.schema.json", frames, "frame"),
        ("history.schema.json", histories, "history"),
        ("profile.schema.json", profiles, "profile"),
        ("question.schema.json", questions, "question"),
        ("scenario.schema.json", scenarios + cross_scenarios, "scenario"),
        ("elicitation_template.schema.json", templates, "elicitation template"),
    ], errors)
    for records, key, label in (
        (frames, "frame_id", "frames"), (histories, "history_id", "histories"),
        (profiles, "profile_id", "profiles"), (questions, "wvs_item_id", "questions"),
        (scenarios + cross_scenarios, "scenario_id", "all scenarios"),
        (templates, "elicitation_id", "templates"),
    ):
        _unique(records, key, label, errors)
    if len(frames) != 8:
        errors.append(f"Expected 8 frames, found {len(frames)}")
    frame_ids = {frame["frame_id"] for frame in frames}
    histories_by_frame: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for history in histories:
        if history["frame_id"] not in frame_ids:
            errors.append(f"{history['history_id']} references missing frame")
        messages = history["messages"]
        user_turns = sum(message["role"] == "user" for message in messages)
        assistant_turns = sum(message["role"] == "assistant" for message in messages)
        if history["history_length"] not in {1, 3, 5}:
            errors.append(f"{history['history_id']} has unsupported history length")
        if user_turns != history["history_length"] or assistant_turns != history["history_length"]:
            errors.append(f"{history['history_id']} has unbalanced role counts")
        if not messages or messages[-1]["role"] != "assistant":
            errors.append(f"{history['history_id']} does not end with assistant")
        if history["history_mode"] != "canonical":
            errors.append(f"{history['history_id']} is not canonical in this pilot")
        histories_by_frame[history["frame_id"]][history["history_length"]] = history
    for frame_id, lengths in histories_by_frame.items():
        if set(lengths) != {1, 3, 5}:
            errors.append(f"{frame_id} does not have exactly H1/H3/H5")
            continue
        if lengths["1" if False else 1]["messages"] != lengths[5]["messages"][:2]:
            errors.append(f"{frame_id} H1 is not a prefix of H5")
        if lengths[3]["messages"] != lengths[5]["messages"][:6]:
            errors.append(f"{frame_id} H3 is not a prefix of H5")
    if len(profiles) != 12:
        errors.append(f"Expected 12 pilot profiles, found {len(profiles)}")
    for profile in profiles:
        attributes = profile["attributes"]
        if not 18 <= attributes["age"] <= 100:
            errors.append(f"{profile['profile_id']} has implausible age")
        if "political" in profile["rendered_profile"].lower():
            errors.append(f"{profile['profile_id']} leaks political ideology")
    for field in ("religion", "gender", "education", "age_band"):
        counts = Counter(profile["attributes"][field] for profile in profiles)
        if field in {"religion", "gender"} and min(counts.values()) < 2:
            errors.append(f"Profile balance has a sparse {field} category: {dict(counts)}")
    question_ids = {question["wvs_item_id"] for question in questions}
    if not 20 <= len(questions) <= 30:
        errors.append(f"Expected 20–30 WVS pilot questions, found {len(questions)}")
    scenarios_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scenario in scenarios:
        scenarios_by_question[scenario["wvs_item_id"]].append(scenario)
        if scenario["wvs_item_id"] not in question_ids:
            errors.append(f"{scenario['scenario_id']} references missing question")
        if not scenario["option_a"].strip() or not scenario["option_b"].strip():
            errors.append(f"{scenario['scenario_id']} has blank option")
        if scenario["option_orders"] != ["AB", "BA"]:
            errors.append(f"{scenario['scenario_id']} lacks explicit AB/BA ordering")
    for question in questions:
        if question["suitable_for_value_elicitation"] and len(scenarios_by_question[question["wvs_item_id"]]) != 4:
            errors.append(f"{question['wvs_item_id']} does not have four scenarios")
    if len(templates) != 4 or {template["elicitation_id"] for template in templates} != {"E01", "E02", "E03", "E04"}:
        errors.append("Primary elicitation template library is incomplete")
    if not _connected(graph["nodes"], graph["edges"]):
        errors.append("Cross-value comparison graph is disconnected")
    if set(graph["nodes"]) != question_ids:
        errors.append("Cross-value graph nodes do not match mapped WVS questions")
    if len(cross_scenarios) != len(graph["edges"]):
        errors.append("Cross-value scenarios do not cover every graph edge")
    for frame in frames:
        sample = compose_experiment({
            "frame_id": frame["frame_id"], "history_length": 3, "profile_id": "P001",
            "scenario_id": "WVS_Q8_S01", "elicitation_id": "E01", "language": "en", "option_order": "BA",
        })
        if sample["messages"][-1]["role"] != "user":
            errors.append(f"{frame['frame_id']} did not compose a final user elicitation")
    return errors


def main() -> None:
    """Print validation findings and use a failing process status for CI."""
    errors = validate_library()
    if errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in errors))
    print("Experiment-library QA passed.")


if __name__ == "__main__":
    main()
