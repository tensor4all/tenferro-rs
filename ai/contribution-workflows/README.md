# Contribution Workflows

This directory contains the canonical instructions for repository-local
agent-assisted contribution workflows.

Tool-specific entry points should stay thin and refer to these files:

- Codex CLI: `.agents/skills/`
- Claude Code: `.claude/skills/`
- OpenCode: `.opencode/commands/`
- Kimi CLI: `.kimi/skills/`

The contribution policy itself lives in `CONTRIBUTING.md` and
`REPOSITORY_RULES.md`. These workflows help agents collect the right
information and enforce the contribution boundary; they do not replace
maintainer review or merge authority.

Available workflows:

- `issue-intake.md`: create or refine bug reports, feature requests, design
  discussions, and documentation/article topic issues.
- `bugfix-pr.md`: prepare pull requests that fix existing intended behavior
  without introducing new features or architectural changes.
- `repository-remediation.md`: batch and resolve repository-rule violations
  across related findings with local-first verification and one non-squash PR.
