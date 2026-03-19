# Common Agent Rules

## Software Engineering Best Practices

**Follow established software engineering best practices at all times.** The
rules in this file are concrete examples and reminders, not an exhaustive list.
When a situation is not explicitly covered, reason from first principles.

Key principles (illustrative, not exhaustive):

- **KISS** — write the simplest code that correctly solves the problem. Do not add abstraction or generality until a second concrete use case requires it.
- **DRY / Single Source of Truth** — every piece of knowledge has one authoritative representation. When logic, data, or structure appears in more than one place, extract it. This applies to production code, tests, and documentation equally.
- **Abstraction** — introduce an abstraction when it genuinely hides complexity or enables reuse. Do not abstract speculatively. Once introduced, maintain it: callers must not reach through it to touch internals.
- **Layering** — each crate exposes a deliberate public API. Downstream crates use high-level APIs; they must not bypass layers to access internals. If a detail is needed, evolve the upstream API instead of leaking it.
- **Separation of Concerns** — each module, file, and function has one clear responsibility.
- **Fix root causes, not symptoms** — prefer fundamental redesign over ad-hoc patches. When a proper fix requires changing more code, do it.
- **Design toward a cleaner end state** — when planning a change, prefer an approach that leaves the codebase clearer than it was before. Do not just bolt new behavior onto the current structure because it is already there; step back, ask how you would design that area from scratch, and move the implementation toward that shape within the task's scope.
- **No speculative backward compatibility** — do not preserve old interfaces or add shims unless the user explicitly asks. Clean up call sites instead.

When in doubt, ask: *"Would an experienced software engineer consider this clean, maintainable, and easy to reason about?"* If not, simplify.

## General

- Think and write in English.
- Keep source code, docs, and user-facing text in English unless a task explicitly requires another language.
- When fixing a bug, inspect nearby code for the same failure mode and call out related risk.

## Startup Context

- At session start, read `README.md`, `AGENTS.md`, and the shared rule files under `ai/`.
- Before creating any new plan, reload and review the full coding ruleset: `README.md`, `AGENTS.md`, and the shared rule files under `ai/`. Do this every time a new plan is written, even within the same session; do not rely on memory of an earlier read.
- If the repository contains local AI workflow files, inspect them before acting. This includes:
  - repo-local skill files such as `ai/**/SKILL.md`
  - project-local command docs such as `.claude/commands/*.md`
  - other repository-declared workflow docs referenced from `README.md` or `AGENTS.md`

## Documentation Requirements

- Every public type, trait, and function must include a minimal but sufficient `# Examples` section in rustdoc.
- Crate-level docs should include a short end-to-end example.
- Keep examples short and readable. Use `ignore` when examples cannot run in docs.

## Code Style

- Run `cargo fmt --all` before committing.
- Prefer `cargo clippy --workspace` before opening a PR.
- Avoid `unwrap()` and `expect()` in library code.
- Use `thiserror` for public API error types.
- Use `anyhow` only for internal glue where typed errors are not part of the public contract.

## Build Environment

- By default, use the repository's normal `./target` directory for Rust build
  artifacts.
- Do not treat NFS or other network filesystems as a normal location for Rust
  development worktrees. Strongly prefer putting the repository or worktree
  itself on local disk.
- If a repository is on NFS or another network filesystem, explicitly warn the
  user before running heavy Rust commands. State that compile and link times
  can degrade severely and that local-disk worktrees are strongly preferred.
- Continue on NFS only if the user explicitly wants to proceed anyway.
- When proceeding on NFS, place Cargo build artifacts on local disk rather than
  inside the repository checkout.
- Also move `CARGO_TARGET_DIR` outside the repository when run isolation or
  local disk layout makes a dedicated external target directory preferable.
- In those cases, prefer a stable repo-specific local path such as
  `/tmp/<repo>-target` or another local-disk directory.

## File Organization

Keep source files small and focused. Split by behavior or abstraction boundary, not by arbitrary line count.

- Prefer **feature-first locality** when a human is likely to trace one
  operation or feature end-to-end. If `svd`, `einsum`, or another concrete
  feature currently spans many unrelated top-level buckets, reorganize toward a
  shape where that feature's primal path, AD wiring, builders, result types,
  and tests live near each other.
- Do not default to broad buckets such as `api`, `common`, `impls`, `plan`, or
  `runtime` as the primary structure when they force a single feature to be
  scattered across the tree. Shared infrastructure should stay small and
  explicit; feature code should stay local.
- Optimize module boundaries for **human navigation**, not just mechanical
  separation of concerns. A developer should be able to answer "where is SVD
  implemented?" or "where does this reduction's AD rule live?" without reading
  half the crate.
- When choosing between `layer-first` and `feature-first`, prefer
  `feature-first + small shared core` unless the crate is genuinely an
  infrastructure-only crate with no coherent end-user features.

## Unit Test Organization

- Keep inline `#[cfg(test)]` blocks only in genuinely tiny leaf modules.
- For normal modules, prefer module-local test directories like `src/<module>/tests/*.rs` and leave only `#[cfg(test)] mod tests;` in the source file.
- Reserve crate-root `tests/` for integration tests.
- Split large test suites by concern instead of keeping one monolithic test module.
- Do not use `include!` to inject test files into modules.

## ASCII Diagrams

- Keep box widths uniform within the same diagram.
- Avoid nested boxes.
- Verify the inner text width matches the border width.

## Dependencies

- Use `[workspace.dependencies]` for dependencies shared across crates in the same workspace.
- Do not commit sibling local `path` dependencies for repositories meant to build in CI.
- Prefer reproducible sources for cross-repository dependencies.
