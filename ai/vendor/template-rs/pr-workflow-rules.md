# PR Workflow Rules

## Branching

- Start feature work from the latest `main`.
- Prefer an isolated git worktree for multi-step implementation work.

## Required Checks

Before pushing or creating a PR, reload and review the full coding ruleset:
`README.md`, `AGENTS.md`, and the shared rule files under `ai/`. Do this every
time you prepare a PR, even within the same session; do not rely on memory of
an earlier read.

Before pushing or creating a PR, all of these must pass:

```bash
cargo fmt --all --check
cargo nextest run --workspace --release --no-fail-fast
cargo test --doc --workspace --release
cargo llvm-cov nextest --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

If formatting fails, run `cargo fmt --all` and rerun the checks.
Keep doctests as a dedicated `cargo test --doc` step; `cargo nextest` does not execute them.

## Repository Settings

- New repositories created from this template must enable GitHub auto-merge.
- The default branch must be protected by the full CI status-check set defined in `ai/repo-settings.json`.
- Repositories with non-template CI job names may override that file locally via `ai/repo-settings.local.json`.
- `createpr` must re-check these repository settings before creating each PR.

## Documentation Consistency

PR readiness always includes a docs gate:

- `README.md`
- `docs/design/**`
- `docs/api_index.md`
- public rustdoc comments
- generated docs from `cargo doc --workspace --no-deps`
- docs-site deployment content from `./scripts/build_docs_site.sh`

If an implementation changes behavior or public API, update the relevant docs before creating the PR.

## PR Creation

- Use `gh pr create` for PR creation.
- AI-generated PRs must include a short attribution line naming the tool used.
- Do not attach raw AI analysis reports as standalone files.
- Enable auto-merge when the repository policy allows it:

```bash
gh pr merge --auto --squash --delete-branch
```

- After creating the PR, poll required checks every 30 seconds:

```bash
bash scripts/monitor-pr-checks.sh <pr-url-or-number> --interval 30
```

- If any required check fails, inspect that failure immediately instead of waiting for the rest of the jobs.
- Fix the failure locally, rerun the relevant local verification, push, and resume the monitoring command until all required checks pass.

## Agent Asset Freshness

- Check for upstream agent-asset updates at startup when possible.
- `stale` means the local agent-assets lock revision is older than the current upstream bundle revision.
- `createpr` should stop on stale assets unless explicitly overridden.
- `sync-agent-assets` updates vendored shared rules, project-local commands, scripts, and the lockfile.
