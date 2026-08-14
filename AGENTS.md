## Project Setup

### Running files with `uv`

This project is synchronized entirely from the top-level directory. Everything should be run from project root including notebooks and tests, this affects how imports are handled, and care should be taken to keep the import structure of the project clean.

### Running commands with `uv`

This project uses `uv` for Python environment and dependency management. Agents should use `uv`-based commands instead of directly invoking `python`, `pip`, or ad hoc virtual environments.

Use the following defaults:

```bash
uv run python <script.py>
uv run pytest
uv run ruff check .
uv run ruff format .
```

When running project modules, prefer:

```bash
uv run python -m <module>
```

When adding or changing dependencies, do not manually edit dependency files unless explicitly instructed. Prefer the appropriate `uv` command and clearly document the resulting changes.

Examples:

```bash
uv add <package>
uv add --dev <package>
uv sync
```

Do not manually edit `uv.lock`. If `uv.lock` changes as the result of an approved `uv` command, mention that in the session log.

Before finishing a session, record any important `uv` commands that were run in `docs/agent-session-log.md`, including whether tests or checks passed.

## Persisting Agent Knowledge Across Sessions

Agents should not rely on chat history as the source of truth. Durable project knowledge must be written into the repository.

### Where to put knowledge

Use the following hierarchy:

- `AGENTS.md`: stable instructions for agents, project conventions, workflows, commands, and links to important docs.
- `docs/agent-session-log.md`: concise per-session handoff notes describing what changed and what remains.
- `docs/adr/`: architectural decisions or durable design rationale.
- `README.md` / package docs: user-facing setup, usage, and development instructions.
- Issues or TODO-tracking system: future work that should be tracked outside the code.

Do not turn `AGENTS.md` into a running changelog. If a lesson is only about what happened in one session, put it in the session log. If it changes how future agents should work, update `AGENTS.md`.

### Start-of-session protocol

**At** the beginning of every session, the agent must:

1. Read `AGENTS.md`.
2. Review `docs/agent-session-log.md` if it exists.
3. Check relevant recent commits, open TODOs, and nearby documentation before editing.
4. Summarize the intended plan before making changes when the task is non-trivial.

### Git and Pull Request Workflow

Agents must keep their work self-contained. Do not start editing on `main` or on a shared branch.

#### Opening a new branch

1. Inspect the current repo state with `git status --short`.
2. Identify the branch you are branching from. By default, branch from the branch that should receive the work next, not automatically from `main`.
3. Sync that base branch before starting new work:
   ```bash
   git fetch origin
   git switch <base-branch>
   git pull --ff-only
   ```
4. Create a fresh branch from the updated base branch before making changes:
   ```bash
   git switch -c feature/<short-topic>
   ```
5. Use a short, descriptive branch name with a prefix such as `feature/`, `fix/`, `docs/`, or `chore/`.

#### Starting a feature

1. Confirm the branch is dedicated to the current request.
2. Keep the scope to one feature, fix, or documentation task.
3. Do not mix unrelated cleanup, refactors, dependency changes, or notebook edits into the same branch unless the user explicitly requests them.
4. If the worktree already contains unrelated user changes, leave them untouched and avoid including them in commits.
5. Before finishing, ensure `git status --short` shows only the intended files for the current task.

#### Opening a pull request

1. Before starting to open a new PR, ask the user if they're ready to do so. The user may want to make multiple changes before a PR is opened.
2. Run the relevant checks for the work you changed and record them in `docs/agent-session-log.md`.
3. Update durable documentation if the task introduced new workflow, architecture, or setup knowledge.
4. Commit only the self-contained changes for the current branch.
5. Push the branch:
   ```bash
   git push -u origin <branch-name>
   ```
6. Open the pull request from that branch into the same base branch you branched from. Do not assume the PR base is `main`.
7. Prefer `gh pr create` when GitHub CLI is available. Use a clear title and body that include:
   - the problem or goal
   - the files or areas changed
   - the checks run and their results
   - any follow-up work or known limitations

#### Sending the pull request

After opening the PR, send the user the PR in a self-contained handoff. Include:

1. the branch name
2. the PR URL or PR number
3. a short summary of what changed
4. the checks run and whether they passed
5. any follow-up work, blockers, or assumptions

If the work is not ready for review, open a draft PR and say that explicitly when sending it.

### End-of-session protocol

At the end of every session, the agent must add or update an entry in `docs/agent-session-log.md` with:

- Date
- Branch name
- User request / session goal
- Summary of changes made
- Files changed
- Commands run
- Tests run and results
- Important decisions or assumptions
- Follow-up work / known issues
- Files intentionally not touched

Keep entries concise, factual, and useful to the next agent.

### Documentation expectation

Every session should leave the repository easier for the next agent to understand. When the agent learns a durable fact about the codebase, workflow, architecture, tests, dependencies, or project conventions, it must document that knowledge in the appropriate place before finishing.

### Protected files

Do not manually edit protected generated or sensitive files unless the user explicitly requests it. In particular:

- Do not manually edit `uv.lock`.
- Do not edit `pyproject.toml` without explicit permission.
- Do not edit `.gitignore` without explicit permission.
- Do not read, print, modify, or commit secrets such as `secrets.json`.

If a dependency change intentionally updates `uv.lock` through `uv sync` or another approved command, mention that clearly in the session log.

### Data Access Rules

The project's datasets are exposed through the top-level `Data` symlink:

- `Data` -> `/Users/ayushkumar/Dropbox/Social Capital (Niche Data)/Data`

Agents must treat files under `Data` as large external data assets, not normal source files.

#### General rules

1. Do not open or print raw data files under `Data`.
2. In particular, do not read `.txt` or `.csv` data payloads such as `college_data.txt`, `k12_school_reviews.txt`, `seda_*.csv`, or `social_capital_high_school.csv`.
3. It is acceptable to inspect metadata files that explain dataset contents:
   - files named `README*`
   - files with `dictionary` in the filename
   - SEDA documentation and codebook files
4. If metadata is insufficient, ask the user before inspecting raw data. Do not sample raw rows by default.
5. When exploring the data tree, prefer filename-only commands such as `find -L Data ...` or `ls` rather than commands that print file contents.

#### Safe metadata entry points

Use these files first when you need to understand what is in the datasets:

- `Data/Niche_NYU_202604/README.md`
- `Data/Niche_NYU_202604/college_dictionary.txt`
- `Data/Niche_NYU_202604/college_reviews_dictionary.txt`
- `Data/Niche_NYU_202604/k12_school_dictionary.txt`
- `Data/Niche_NYU_202604/k12_school_district_dictionary.txt`
- `Data/Niche_NYU_202604/k12_school_district_reviews_dictionary.txt`
- `Data/Niche_NYU_202604/k12_school_reviews_dictionary.txt`
- `Data/SEDA/edu_opportunity/SEDA_documentation_6.0.pdf`
- `Data/SEDA/edu_opportunity/seda_codebook_admindist_6.0.xlsx`
- `Data/SEDA/edu_opportunity/seda_codebook_commzone_6.0.xlsx`
- `Data/SEDA/edu_opportunity/seda_codebook_county_6.0.xlsx`
- `Data/SEDA/edu_opportunity/seda_codebook_cov_admindist_6.0.xlsx`
- `Data/SEDA/edu_opportunity/seda_codebook_cov_county_6.0.xlsx`
- `Data/SEDA/edu_opportunity/seda_codebook_cov_geodist_6.0.xlsx`
- `Data/SEDA/edu_opportunity/seda_codebook_cov_school_6.0.xlsx`
- `Data/SEDA/edu_opportunity/seda_codebook_crosswalk_6.0.xlsx`
- `Data/SEDA/edu_opportunity/seda_codebook_geodist_6.0.xlsx`
- `Data/SEDA/edu_opportunity/seda_codebook_school_6.0.xlsx`
- `Data/Social Capital Data/data_release_readme_31_07_2022_nomatrix.pdf`

#### Raw data files that should not be read by default

These are examples of files agents should not open unless the user explicitly directs it:

- `Data/Niche_NYU_202604/college_data.txt`
- `Data/Niche_NYU_202604/college_reviews.txt`
- `Data/Niche_NYU_202604/k12_school_data_2019.txt` through `k12_school_data_2026.txt`
- `Data/Niche_NYU_202604/k12_school_district_data.txt`
- `Data/Niche_NYU_202604/k12_school_district_reviews.txt`
- `Data/Niche_NYU_202604/k12_school_reviews.txt`
- `Data/SEDA/edu_opportunity/seda_*.csv`
- `Data/Social Capital Data/social_capital_high_school.csv`
- `Data/Social Capital Data/crosswalks/*.csv`

### Output Conventions

The project's derived outputs are exposed through the top-level `Output` symlink:

- `Output` -> `/Users/ayushkumar/Dropbox/Social Capital (Niche Data)/Output`

Agents should place generated artifacts in `Output`, not in the repository itself.

#### Output paths

1. Save plots to `Output/figures/<figure_name>_<YYYY-MM-DD>.png`.
2. Save LaTeX tables to `Output/tables/<table_name>_<YYYY-MM-DD>.tex`.
3. Use ISO dates in filenames, for example `friending_bias_by_cohort_2026-06-23.png`.
4. Create `Output/figures` and `Output/tables` if they do not already exist.
5. Keep names short, descriptive, and stable enough to be searchable later.

#### Figure standards

1. Start from the default Opportunity Insights styling in `src/python/plotting.py`, typically through `use_oi_style()`.
2. Export figures at 300 dpi.
3. Make only sensible legibility adjustments on top of the default OI style, such as figure size, tick density, legend placement, or annotation sizing.
4. Prefer readable axis labels, titles, and legends over strict style purity when there is a conflict.

#### Table standards

1. Produce LaTeX tables in an AER-style academic format.
2. Format every table to fit within a single LaTeX page.
3. Use the `array` package in the LaTeX table markup so column widths can be controlled explicitly.
4. Avoid tables that are excessively wide or excessively narrow; use fixed-width column specifications where needed to balance the layout.
5. Prefer clean journal-style tables with no vertical rules, consistent decimal formatting, and concise notes when notes are needed.

### Efficient use of the session log

The agent session log may become long over time. Agents should not load the entire `docs/agent-session-log.md` into context by default.

Instead, agents should query or skim only the sections relevant to the current task. Use headings, dates, changed file paths, feature names, module names, commands, and error messages to locate relevant prior work.

Recommended lookup process:

1. Identify the files, modules, features, or workflows involved in the current task.
2. Search `docs/agent-session-log.md` for those terms.
3. Read only the matching entries and nearby context.
4. If the relevant history is unclear, broaden the search by related terms.
5. Do not summarize or depend on unrelated historical entries.

When adding a new session-log entry, include searchable keywords such as affected paths, feature names, command names, dependency names, and important errors. This makes future targeted retrieval easier.

If a piece of knowledge is repeatedly needed across unrelated sessions, promote it out of the session log and into the appropriate durable documentation, such as `AGENTS.md`, `README.md`, or an ADR.

## Task workflow

- Read the task file provided by the user.
- Read only the relevant parts of AGENTS.md and prior session logs.
- Before editing, state a concise plan.
- Prefer minimal, targeted changes.
- Run the project’s standard checks.
- Append a session summary to the session log after completing work.

## Code Style Preferences

- Prefer simple, explicit code over clever dynamic discovery when the relevant columns, labels, or paths are known.
- Put shared project paths, hardcoded column selections, thresholds, and reusable label dictionaries in `src/python/constants.py`.
- Use large explicit dictionaries for table labels and similar mappings rather than helper functions that infer human-readable names.
- Avoid arbitrary feature caps unless the task requires them; if a threshold is used, name it clearly and add a short comment explaining what it means.
- Split analysis workflows into data-preparation scripts under `src/python/data_prep/` that write intermediate artifacts and production analysis scripts under `src/python/analysis/` that consume those artifacts.
- Add PEP-style docstrings to reusable functions in analysis and data-preparation scripts.
