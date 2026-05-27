# Contributing

Thanks for helping improve tenferro-rs. This file describes the external
contribution path. Repository-specific implementation rules live in
`REPOSITORY_RULES.md`.

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
