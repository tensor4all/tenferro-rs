# Issue Intake Workflow

Use this workflow when the user wants to create or refine a tenferro-rs GitHub
issue. It covers bug reports, feature requests, design discussions, and
documentation or article topic issues.

Do not use this workflow to open a new-feature implementation PR. New features
must start as issues.

## Operating Rules

- Read `CONTRIBUTING.md` first. Read `REPOSITORY_RULES.md` when the issue may
  affect architecture, crate boundaries, dependencies, backend behavior, AD, or
  maintainer policy.
- Prefer repository and GitHub inspection over asking the user for facts that
  can be discovered locally.
- Ask only for missing judgment calls or unavailable facts. Ask at most three
  questions at a time.
- Do not create the issue until the user has seen the final draft and approved
  it, unless the user explicitly asked for direct creation without another
  confirmation.
- If the user asks for a new-feature PR, redirect to a feature request issue.
  If prototype code exists, collect it as a link and provenance note.
- Treat maintainers as the authority for labels, acceptance, roadmap priority,
  and merge decisions.

## Step 1: Classify

Determine the issue type:

- Bug report
- Feature request
- Design discussion
- Documentation or article topic

If unclear, ask the user to choose one. Do not offer "new feature PR" as a
classification.

## Step 2: Check Existing Context

Before drafting, inspect the relevant local and GitHub context when available:

- Existing issue templates under `.github/ISSUE_TEMPLATE/`
- Current docs and README sections related to the topic
- Existing open and closed issues or PRs with similar keywords
- Current crate boundaries and feature flags when dependency or backend impact
  is part of the request

If GitHub access is unavailable, say so and continue with a local draft.

## Step 3: Collect The Minimal Missing Inputs

Collect a one- or two-sentence goal for every issue type:

- What should be fixed, enabled, clarified, or discussed?
- Why does it matter for users or maintainers?

For bug reports, collect:

- Expected behavior
- Actual behavior
- Minimal reproducer, failing test, command, panic, error, or log
- Affected API, crate, backend, device, dtype, and feature flags when known
- tenferro-rs commit or version and relevant environment details
- Verification hint for how the fix should be checked

For feature requests and design discussions, collect:

- User workflow or scientific/computing use case
- Current limitation or friction
- Proposed behavior or API, if known
- Affected crates or layers
- Dependency, backend, CUDA, BLAS/LAPACK, AD, and cache implications
- Non-goals and compatibility constraints
- Acceptance criteria precise enough for an implementer to act on

For documentation or article topic issues, collect:

- Target audience
- Main claim or explanation to preserve
- Source issues, discussions, examples, or design decisions to link
- Where the result should live: docs site, README, rustdoc, blog/article draft,
  or undecided
- Acceptance criteria for publication or documentation completion

For any issue with prototype or external reference code, collect:

- URL to fork branch, gist, repository, paper, or external implementation
- License and copyright information when known
- Whether the prototype may be used as an implementation reference
- Whether it is only illustrative and must not be copied or derived from

## Step 4: Scope And Policy Gate

Apply these gates before drafting:

- If the request changes public API, adds an operation family, adds a backend,
  adds a dependency, changes feature flags, changes AD semantics, or changes
  crate boundaries, make it a feature request or design discussion issue.
- If it reports incorrect current behavior for an existing intended path, make
  it a bug report.
- If it mainly records material for future explanation, make it a documentation
  or article topic issue.
- If it combines several unrelated changes, split it or write an umbrella issue
  with explicit sub-issues.

## Step 5: Draft

Produce a concise draft before creating the issue.

Include:

- Type
- Title
- Problem statement
- Relevant current behavior
- Proposed direction, if any
- Affected areas or crates
- Dependency, backend, CUDA, BLAS/LAPACK, AD, cache, and feature-flag impact
  when relevant
- Non-goals
- Acceptance criteria
- Tests, verification, or documentation checks
- Prototype, provenance, and licensing notes when relevant
- Links to related issues, PRs, docs, or code paths

## Step 6: Create Or Hand Off

After confirmation, create the issue with the best matching GitHub template
when GitHub access is available. If not, provide the title and body so the user
can use it with the GitHub UI or another tool.

When creating with `gh`, prefer:

```bash
gh issue create --title "<title>" --body-file <body-file>
```

Use labels only after checking that they exist in the repository.
