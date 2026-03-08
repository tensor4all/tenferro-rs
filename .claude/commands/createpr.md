Create a pull request using the repository-local PR workflow.

Workflow:

1. Re-read `README.md`, `AGENTS.md`, and the referenced `ai/*.md` rule files.
2. Review docs consistency across `README.md`, `docs/design/**`, and public rustdoc for the current diff.
3. Confirm the repository still has auto-merge enabled and the required branch protection checks configured.
4. Draft a concise PR title and body.
5. Run `bash scripts/create-pr.sh --ai-tool-name "Claude Code" --ai-tool-url "https://claude.com/claude-code" --title "<title>" --body-file <temp-file> "$@"`.
6. If the monitor reports a failed check, inspect that failure immediately. Do not wait for other jobs to finish.
7. Fix the failure locally, rerun the relevant local verification, push, and resume with `bash scripts/monitor-pr-checks.sh <pr-url-or-number> --interval 30` until all required checks pass.
8. Use `--allow-stale` only when the user explicitly accepts creating a PR with stale or unverified agent assets.

Do not skip the script's verification steps. The script is responsible for freshness checks, formatting, release-mode coverage verification, docs, PR creation, optional auto-merge, and fail-fast PR polling.
