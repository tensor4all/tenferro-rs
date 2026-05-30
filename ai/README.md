# AI Workflow Assets

This directory contains tenferro-rs-specific agent workflows and automation
prompts. It is not a vendored copy of shared rules from another repository.

Shared tensor4all agent rules live in
`tensor4all/tensor4all-agent-rules` and are read online on demand, with the
optional sibling checkout fallback documented in `AGENTS.md`.

## Contents

- `contribution-workflows/`: reusable repository-local workflows for issue
  intake and bug-fix pull requests.
- `repo-settings.json`: the expected GitHub repository settings and required
  branch protection checks for this repository.
- `run-codex-solve-bug.sh`, `run-claude-solve-bug.sh`, and
  `solve_bug_issue.md`: headless bug-fix automation entry points.

## Rules

- Do not add vendored shared-rule bundles under `ai/`.
- Do not add agent asset lockfiles or sync manifests for external templates.
- Keep durable tenferro-specific policy in `REPOSITORY_RULES.md`.
- Keep contribution policy in `CONTRIBUTING.md`; keep workflow mechanics here.
