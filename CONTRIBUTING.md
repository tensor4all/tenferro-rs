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

Bug reports, minimal reproducers, and regression tests are welcome in issues.

Bug-fix pull requests from collaborators are welcome when they fix behavior
that is already intended by current docs, specs, or tests. A bug-fix PR should
not introduce a new public API, operation family, backend, dependency, feature
flag, or architectural layer.

This repository restricts pull request creation to collaborators. If you are
not a collaborator, please open an issue with the reproducer, proposed test,
or prototype branch instead of opening an implementation PR.

## Feature requests, prototypes, and implementation ownership

New features, substantial behavior changes, new public APIs, new backends, new
dependencies, and architectural changes must start as an issue before an
implementation pull request is opened.

This boundary is intended to keep development fast and coherent in an
agentic-coding workflow. The project preserves API consistency, internal
architecture, test strategy, backend behavior, and long-term maintainability
best when maintainers and active collaborators own the final implementation
inside the repository.

Requests, prototype code, exploratory branches, gists, external repositories,
and focused unit tests are still useful contributions. Please link them from
the issue and explain what behavior they demonstrate. The issue remains the
source of truth for the accepted API, dependency impact, backend behavior, AD
behavior, tests, implementation plan, and roadmap decision.

Implementation PRs for new features or substantial changes that are opened
before an accepted issue may be closed with a request to continue the
discussion in an issue first.

## Prototype code and provenance

By submitting code directly to this repository, you represent that you have the
right to submit it under this repository's license, `MIT OR Apache-2.0`.

If you link prototype code from an issue, clearly state its license if it is
not `MIT OR Apache-2.0` or if the project should not use it as an
implementation reference.

When maintainers implement an accepted issue, they may rewrite the design from
scratch, use the prototype only as a behavioral reference, or take over a
prototype branch. If the project takes over a prototype branch, maintainers
should preserve the contributor's original commits where practical and add new
commits on top. If the final implementation is otherwise based on contributed
prototype code, including by rewriting it manually or with AI assistance, the
project will preserve appropriate copyright notices, license obligations,
attribution, and links to the original prototype or issue discussion.

If your prototype is only meant to illustrate behavior and must not be used as
an implementation reference, say so explicitly in the issue.

## Contributors

Contributors may be listed in `CONTRIBUTORS.md`. Contributor recognition does
not imply maintainer status, merge authority, copyright transfer, or ownership
of project direction.
