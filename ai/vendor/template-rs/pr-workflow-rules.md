# PR Workflow Rules

## Branching

- Start feature work from the latest `main`.
- Prefer an isolated git worktree for multi-step implementation work.

## Required Checks

Before pushing or creating a PR, all of these must pass:

```bash
cargo fmt --all --check
cargo test --workspace
cargo llvm-cov --workspace --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --no-deps
```

If formatting fails, run `cargo fmt --all` and rerun the checks.

## Repository Settings

- New repositories created from this template must enable GitHub auto-merge.
- The default branch must be protected by the full CI status-check set defined in `ai/repo-settings.json`.
- Repositories with non-template CI job names may override that file locally via `ai/repo-settings.local.json`.
- `createpr` must re-check these repository settings before creating each PR.

## Documentation Consistency

PR readiness always includes a docs gate:

- `README.md`
- `docs/design/**`
- public rustdoc comments
- generated docs from `cargo doc --no-deps`

If an implementation changes behavior or public API, update the relevant docs before creating the PR.

## PR Creation

- Use `gh pr create` for PR creation.
- AI-generated PRs must include a short attribution line naming the tool used.
- Do not attach raw AI analysis reports as standalone files.
- Enable auto-merge when the repository policy allows it:

```bash
gh pr merge --auto --squash --delete-branch
```

## Agent Asset Freshness

- Check for upstream agent-asset updates at startup when possible.
- `stale` means the local agent-assets lock revision is older than the current upstream bundle revision.
- `createpr` should stop on stale assets unless explicitly overridden.
- `sync-agent-assets` updates vendored shared rules, project-local commands, scripts, and the lockfile.
