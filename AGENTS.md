# AGENTS

## Fast facts
- This is a minimal Django 5.2 project with two local apps: `django_experiment_tracker` (base tracking models) and `rna_vel_pred` (concrete experiment models).
- Django settings module is `project.settings`; default DB is SQLite at `db.sqlite3`.
- There is no custom URL wiring yet beyond Django admin (`project/urls.py`).

## Setup and run commands
- Python version is pinned by `.python-version` to `3.10`.
- Dependencies are defined in `pyproject.toml` and locked in `uv.lock`; prefer `uv` commands.
- Install deps: `uv sync`
- Run management commands: `uv run python manage.py <command>`
- Apply schema changes: `uv run python manage.py makemigrations` then `uv run python manage.py migrate`
- Run tests: `uv run python manage.py test`
- Run one app's tests: `uv run python manage.py test rna_vel_pred` (or `django_experiment_tracker`)

## Model/migration quirks
- `rna_vel_pred.Experiment` subclasses the abstract `django_experiment_tracker.Experiment`; keep tracker changes migrated before app-level model changes.
- `rna_vel_pred.ExperimentParameter` uses `models.CompositePrimaryKey('experiment', 'parameter')` (Django 5.2 feature). Preserve this shape unless intentionally redesigning keys.
- `alt_id` on `Experiment` is DB-generated via SQLite functions (`randomblob`/`hex`) in model defaults and migrations; avoid replacing with Python-side random IDs without a deliberate migration plan.

## Current repo state caveats
- `README.rst` is empty; do not rely on it for workflow guidance.
