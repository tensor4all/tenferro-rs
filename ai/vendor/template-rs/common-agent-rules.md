# Common Agent Rules

## General

- Think and write in English.
- Keep source code, docs, and user-facing text in English unless a task explicitly requires another language.
- When fixing a bug, inspect nearby code for the same failure mode and call out related risk.

## Startup Context

- At session start, read `README.md`, `AGENTS.md`, and the shared rule files under `ai/`.
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

- Do not treat NFS or other network filesystems as a normal location for Rust
  development worktrees. Strongly prefer putting the repository or worktree
  itself on local disk.
- If a repository is on NFS or another network filesystem, explicitly warn the
  user before running heavy Rust commands. State that compile and link times
  can degrade severely and that local-disk worktrees are strongly preferred.
- Continue on NFS only if the user explicitly wants to proceed anyway.
- When proceeding on NFS, place Cargo build artifacts on local disk rather than
  inside the repository checkout.
- Prefer a stable repo-specific local target directory such as
  `CARGO_TARGET_DIR=/tmp/<repo>-target` for `cargo build`, `cargo test`,
  `cargo llvm-cov`, and similar heavy commands.

## File Organization

Keep source files small and focused. Split by behavior or abstraction boundary, not by arbitrary line count.

Benefits:

- smaller public/private surfaces to review
- easier parallel editing
- faster navigation for humans and AI

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

## Layering

- Keep public APIs small and deliberate.
- Downstream crates should use high-level APIs rather than reaching into lower-level internals.
- Tests should follow the same layering rules as production code.
