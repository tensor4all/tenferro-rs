Create a pull request using the repository-local PR workflow.

Workflow:

1. Re-read `README.md`, `AGENTS.md`, `REPOSITORY_RULES.md`, and the relevant repository-local workflow docs under `ai/`.
2. Self-review the diff against `REPOSITORY_RULES.md` and the relevant shared tensor4all rules referenced by `AGENTS.md`. Fix any violations before proceeding.
3. Review docs consistency across `README.md`, `docs/design/**`, and public rustdoc for the current diff.
4. Confirm the repository still has auto-merge enabled and the required branch protection checks configured.
5. Draft a concise PR title and body.
6. Run `bash scripts/create-pr.sh --ai-tool-name "Claude Code" --ai-tool-url "https://claude.com/claude-code" --title "<title>" --body-file <temp-file> "$@"`.
7. If the monitor reports a failed check, inspect that failure immediately. Do not wait for other jobs to finish.
8. Fix the failure locally, rerun the relevant local verification, push, and resume with `bash scripts/monitor-pr-checks.sh <pr-url-or-number> --interval 30` until all required checks pass.

Do not skip the script's verification steps. The script is responsible for formatting, release-mode coverage verification, docs, PR creation, optional auto-merge, and fail-fast PR polling.
