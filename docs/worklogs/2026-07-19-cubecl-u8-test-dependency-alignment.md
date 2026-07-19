# CubeCL `u8` Test Dependency Alignment

## Summary

Updated tenferro's Git dependencies to consume the CubeCL change that gates
the `u8` branch runtime tests by each backend's supported unsigned-type list.
The update also advances CubeK because its workspace dependency still pinned
the preceding CubeCL commit.

## Context Reviewed

- tensor4all shared repository, Rust, performance, documentation, and test rules
- tenferro `AGENTS.md`, `REPOSITORY_RULES.md`, contribution policy, and bug-fix workflow
- CubeCL runtime-test generation and backend type lists
- CubeK workspace CubeCL dependencies
- tenferro workspace CubeCL and CubeK dependencies

## Decision

Pin all direct CubeCL dependencies to CubeCL PR #14 head
`346135ab43cececf6405d52a3dbc987537402d27`. Pin all CubeK dependencies to
CubeK PR #10 head `efc43529449b67d83f53e585cc7d71018a252ac7`, which uses the same
CubeCL commit.

Updating only tenferro's direct CubeCL dependencies was rejected because Cargo
then resolved both `11b52669` through CubeK and `346135ab` directly. Separate
CubeCL runtime and IR crate identities are unsafe for the integration boundary
and also duplicate compilation.

## Verification

- confirmed the dependency tree contains CubeCL `346135ab` and no `11b52669`
- `cargo fmt --all --check`
- `cargo check -p tenferro-gpu --no-default-features --features cpu-faer,webgpu`

The repository-wide PR checks are recorded in the pull request. Hardware CUDA
execution is left to the existing gated CI lanes; this change does not alter
kernel or backend behavior.
