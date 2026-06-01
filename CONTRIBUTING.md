# Contributing

Thanks for helping improve tenferro-rs. This file describes the external
contribution path. Repository-specific implementation rules live in
`REPOSITORY_RULES.md`.

Governance, maintainer roles, merge authority, and project-direction decisions
are defined in [GOVERNANCE.md](GOVERNANCE.md). The maintainer list is
maintained in [CONTRIBUTORS.md](CONTRIBUTORS.md).

## Agent-assisted workflows

Agent-assisted issue and bug-fix PR preparation is supported.

The canonical contribution policy is this file plus `REPOSITORY_RULES.md`.
Repository-local AI workflows are helpers for collecting the right information
and applying the contribution boundary; they do not replace maintainer review,
merge authority, or roadmap decisions.

Supported entry points:

- Codex CLI: use the `tenferro-issue-intake` or `tenferro-bugfix-pr` skill
  from `.agents/skills/`.
- Claude Code: invoke `/tenferro-issue-intake` or `/tenferro-bugfix-pr`; the
  project skills live in `.claude/skills/`.
- OpenCode: invoke `/tenferro-issue-intake` or `/tenferro-bugfix-pr`; the
  project commands live in `.opencode/commands/`.

The shared workflow bodies live in `ai/contribution-workflows/`.

Use the issue-intake workflow for bug reports, feature requests, design
discussion issues, and documentation or article topic issues. Use the
bug-fix PR workflow only for fixes to existing intended behavior. If a proposed
bug-fix PR needs a new public API, operation family, backend, dependency,
feature flag, architectural layer, or AD semantics change, move it to an issue
first.

## Bug fixes

Bug-fix pull requests are welcome.

A bug-fix PR should fix behavior that is already intended by current docs,
specs, or tests. It should not introduce a new public API, operation family,
backend, dependency, feature flag, or architectural layer.

Please include a minimal reproducer or regression test when practical.

## New features

New features must start as a feature request issue. Please do not open a
new-feature implementation PR before maintainers accept the issue and agree
that implementation should start.

If you already have prototype code, link to a fork branch, gist, or repository
from the feature request issue. The issue remains the source of truth for the
accepted API, dependency impact, backend behavior, AD behavior, tests, and
roadmap decision.

New-feature implementation PRs opened before an accepted issue may be closed
with a request to continue in an issue.

## Prototype code and provenance

By submitting code directly to this repository, you represent that you have the
right to submit it under this repository's license, `MIT OR Apache-2.0`.

If you link prototype code from a feature request issue, clearly state its
license if it is not `MIT OR Apache-2.0` or if the project should not use it as
an implementation reference.

If maintainers implement a feature using your prototype code as a basis,
including by rewriting it manually or with AI assistance, the resulting
implementation may still be derived from the prototype. In that case, the
project will preserve appropriate copyright notices, license obligations,
attribution, and links to the original prototype or issue discussion.

If your prototype is only meant to illustrate behavior and must not be used as
an implementation reference, say so explicitly in the issue.

## Contributors

Contributors may be listed in `CONTRIBUTORS.md`. Contributor recognition does
not imply maintainer status, merge authority, copyright transfer, or ownership
of project direction.
