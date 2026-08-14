"""Deterministically compose one lazy experimental condition into API messages."""

from __future__ import annotations

from typing import Any

from .io import read_jsonl
from .paths import ENGLISH_ROOT

BASE_INSTRUCTIONS = (
    "Follow the conversation naturally. Treat the known user profile as background "
    "context only. Do not mention experimental metadata, latent values, or the study design."
)


def _index(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {record[key]: record for record in records}


def _english_records() -> dict[str, dict[str, dict[str, Any]]]:
    """Load current frozen English atomic libraries, indexed by stable identifier."""
    return {
        "frames": _index(read_jsonl(ENGLISH_ROOT / "data" / "frames.jsonl"), "frame_id"),
        "histories": _index(read_jsonl(ENGLISH_ROOT / "data" / "histories.jsonl"), "history_id"),
        "profiles": _index(read_jsonl(ENGLISH_ROOT / "data" / "profiles.jsonl"), "profile_id"),
        "scenarios": _index(read_jsonl(ENGLISH_ROOT / "wvs" / "scenarios.jsonl"), "scenario_id"),
        "templates": _index(read_jsonl(ENGLISH_ROOT / "elicitation" / "templates.jsonl"), "elicitation_id"),
    }


def _render_template(template: str, replacements: dict[str, str]) -> str:
    """Replace the small, explicit template vocabulary and reject dangling fields."""
    for key, value in replacements.items():
        template = template.replace("{{" + key + "}}", value)
    if "{{" in template or "}}" in template:
        raise ValueError(f"Unresolved placeholder in elicitation template: {template!r}")
    return template


def compose_experiment(condition: dict[str, Any]) -> dict[str, Any]:
    """Compose one condition without materializing unrelated combinations.

    Required keys are ``frame_id``, ``history_length``, ``profile_id``,
    ``scenario_id``, ``elicitation_id``, ``language``, and ``option_order``.
    The result retains the input condition alongside role-preserving messages.
    """
    required = {
        "frame_id", "history_length", "profile_id", "scenario_id", "elicitation_id",
        "language", "option_order",
    }
    missing = sorted(required.difference(condition))
    if missing:
        raise ValueError(f"Condition is missing required fields: {', '.join(missing)}")
    if condition["language"] != "en":
        raise ValueError("Only frozen English source materials are available in this pilot.")
    if condition["option_order"] not in {"AB", "BA"}:
        raise ValueError("option_order must be AB or BA")
    records = _english_records()
    history_id = f"{condition['frame_id']}_H{condition['history_length']}_R01"
    try:
        frame = records["frames"][condition["frame_id"]]
        history = records["histories"][history_id]
        profile = records["profiles"][condition["profile_id"]]
        scenario = records["scenarios"][condition["scenario_id"]]
        template = records["templates"][condition["elicitation_id"]]
    except KeyError as error:
        raise ValueError(f"Unknown experimental identifier: {error.args[0]}") from error
    if history["frame_id"] != frame["frame_id"]:
        raise ValueError("History does not belong to requested frame")
    if condition["option_order"] == "AB":
        option_a, option_b = scenario["option_a"], scenario["option_b"]
    else:
        option_a, option_b = scenario["option_b"], scenario["option_a"]
    final_elicitation = _render_template(
        template["prompt_template"],
        {"option_a_label": "A", "option_b_label": "B", "option_a": option_a, "option_b": option_b},
    )
    messages = [
        {"role": "developer", "content": BASE_INSTRUCTIONS},
        {"role": "developer", "content": profile["rendered_profile"]},
    ]
    if frame["developer_instruction_required"]:
        messages.append({"role": "developer", "content": frame["developer_instruction"]})
    messages.extend(history["messages"])
    messages.append({"role": "user", "content": f"{scenario['context']}\n\n{final_elicitation}"})
    return {
        "condition": dict(condition),
        "material_ids": {"history_id": history_id, "wvs_item_id": scenario["wvs_item_id"]},
        "messages": messages,
    }
