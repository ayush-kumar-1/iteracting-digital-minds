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
