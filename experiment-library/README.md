# Context-conditioned value elicitation pilot

This directory is a composable, English-source experimental instrument—not a
materialized factorial experiment. It holds 8 frames, 24 canonical histories,
12 structured synthetic profiles, 26 selected World Values Survey (WVS) Wave 7
items, four within-item scenarios per selected item, 25 cross-value bridge
scenarios, and four elicitation templates.

All experiment conditions are rendered lazily from stable IDs. For example:

```bash
uv run python -c "from src.python.experiment_library.composition import compose_experiment; print(compose_experiment({'frame_id': 'F05', 'history_length': 3, 'profile_id': 'P008', 'scenario_id': 'WVS_Q108_S03', 'elicitation_id': 'E01', 'language': 'en', 'option_order': 'BA'}))"
```

Regenerate frozen English materials after editing the pilot specification:

```bash
uv run python -m src.python.experiment_library.generate_pilot
uv run python -m src.python.experiment_library.validation
```

The WVS source, access date, questionnaire edition, selected item wording, and
response options are recorded in `en/wvs/survey.json` and `questions.jsonl`.
The pilot is intentionally limited to the 26 selected items; it does not
silently imply a classification of every WVS Wave 7 variable.

The JSONL files are canonical because they support streaming and record-level
validation. For review, `en/wvs/scenario_catalog.json` is an indented,
human-readable mirror of all 104 within-item and 25 cross-value scenarios.

Translations must be stored in a new language directory only after this English
source is reviewed and frozen. Follow [the translation protocol](en/docs/translation.md).

## Phase-one response generation

`src.python.experiment_library.response_generation` is the manifest-first
Anthropic runner for this frozen library. It generates and strictly normalizes
responses only; it does not calculate WVS scores, utilities, treatment effects,
or preference drift.

The supplied `configs/haiku_main_effects.yaml` uses pinned
`claude-haiku-4-5-20251001`, temperature 1.0, 10 replicates per A/B ordering,
and 16 maximum output tokens. It has explicit absent-manipulation controls:
`profile_id: null` and `history_id: null`/`history_length: 0` in the immutable
manifest. The default design runs baseline, context main effects, profile main
effects, and E02--E04 elicitation main effects; E01 baseline is intentionally
materialized once. A later config can set `design.interactions.enabled: true`
with `dimensions: [context_profile]` without changing the data model.

Run every command from the project root. The first command prints a deterministic
run ID; use that ID in later commands:

```bash
uv run python -m src.python.experiment_library.response_generation build-manifest --config configs/haiku_main_effects.yaml
uv run python -m src.python.experiment_library.response_generation validate --run <run-id>
uv run python -m src.python.experiment_library.response_generation estimate-cost --run <run-id>
uv run python -m src.python.experiment_library.response_generation smoke-test --run <run-id>
uv run python -m src.python.experiment_library.response_generation submit --run <run-id>
uv run python -m src.python.experiment_library.response_generation register-batch --run <run-id> --batch-id <existing-batch-id>
uv run python -m src.python.experiment_library.response_generation status --run <run-id>
uv run python -m src.python.experiment_library.response_generation collect --run <run-id>
uv run python -m src.python.experiment_library.response_generation parse --run <run-id>
uv run python -m src.python.experiment_library.response_generation summarize-run --run <run-id>
```

Run artifacts are written under `Output/experiments/<run-id>/`:

- `manifest.jsonl` is the immutable intended stochastic experiment, one row per
  API request.
- `requests.jsonl` holds the exact native Anthropic system and alternating
  user/assistant messages sent for every request.
- `batches.jsonl` persists each returned batch ID immediately after submission;
  `raw_results.jsonl` preserves complete API result objects for collection.
- `responses.parquet` is an idempotent, strict derivative. It retains raw text
  and letter choice, then maps A/B back to the canonical scenario option and
  value pole.

API operations read the existing root `secrets.json` only at runtime, accepting
the repository's `anthropic-api-key` label (plus conventional Anthropic aliases).
It is never copied to
the manifest, requests, metadata, or logs. `estimate-cost` counts each unique
rendered prompt using Anthropic's token-count endpoint before calculating the
discounted batch estimate. `smoke-test` is synchronous and writes separate
non-observation artifacts; it never enters `raw_results.jsonl`.

`register-batch` is a recovery operation only: it retrieves and records an
already-created batch ID against an existing manifest without sending any new
requests. It is useful if a process dies after Anthropic accepts a batch but
before local bookkeeping is written.
