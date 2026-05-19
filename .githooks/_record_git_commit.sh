#!/usr/bin/env bash
set -u

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

warn() {
  printf '%s\n' "[git-hook] $1" >&2
}

run_record() {
  if ! command -v uv >/dev/null 2>&1; then
    warn "uv is not available; skipping GitCommit recording"
    return 0
  fi

  if ! uv run python manage.py record_git_commit "$@"; then
    warn "record_git_commit failed; commit flow will continue"
    return 0
  fi
}

cd "$REPO_ROOT" || exit 0
run_record "$@"
exit 0
