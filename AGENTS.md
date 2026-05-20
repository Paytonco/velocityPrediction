# AGENTS

## Project shape
- Minimal Django 5.2 repo with two local apps in `INSTALLED_APPS`: `django_experiment_tracker` (shared tracker models + `record_git_commit` command) and `rna_vel_pred` (experiment models + generation command).
- Settings module is `project.settings`; DB is SQLite at `db.sqlite3`; URL routing is only Django admin in `project/urls.py`.
- `README.rst` is empty; treat code/config as source of truth.

## Commands agents should use
- Python is pinned to `3.10` in `.python-version`.
- Dependencies are managed with `uv` (`pyproject.toml` + `uv.lock`): run `uv sync`.
- Run Django commands as `uv run python manage.py <command>`.
- Migrations order matters: `uv run python manage.py makemigrations` then `uv run python manage.py migrate`.
- Tests: `uv run python manage.py test`; focused app tests: `uv run python manage.py test rna_vel_pred` (or `django_experiment_tracker`).

## Data model and migration gotchas
- `rna_vel_pred.Experiment` subclasses abstract `django_experiment_tracker.Experiment`; schema changes in tracker models usually need tracker migrations before dependent `rna_vel_pred` migrations.
- Do not replace `Experiment.alt_id` generation with Python randomness: it is DB-side via SQLite `randomblob`/`hex` (`db_default` in model + migration).
- `rna_vel_pred.ExperimentParameter` is keyed by default `id` plus a uniqueness constraint on `('experiment', 'parameter', 'parameter_group')`; there is no composite primary key.

## Git hook automation in this repo
- Hooks are versioned in `.githooks/`; install once per clone with `bash scripts/setup-git-hooks.sh` (sets `core.hooksPath`).
- `post-commit` and `post-merge` run `record_git_commit`; `post-rewrite` forwards rewrite-map stdin and then backfills last 100 commits.
- Hook scripts are intentionally non-blocking: failures warn and exit 0.
- Manual recovery: `uv run python manage.py record_git_commit` or `uv run python manage.py record_git_commit --backfill 100`.

## Command wiring quirk
- `uv run python manage.py generate_generalization_experiments` imports from `notebooks/generate_generalization_experiments.py`; keep notebook helper functions import-safe if you edit that notebook module.
