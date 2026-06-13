# CubeK Complex GEMM API Work Log

## Summary

This change moves WebGPU `C32` complex GEMM semantics from tenferro-local
lowering into CubeK and wires `WebGpuBackend::dot_general_with_conj` through the
new CubeK API. CUDA contraction code is intentionally unchanged.

## Context Read

- `AGENTS.md`
- `REPOSITORY_RULES.md`
- `docs/design/gpu-backend-design.md`
- `docs/guides/devices-and-gpu.md`
- `docs/superpowers/plans/2026-06-13-cubek-complex-gemm-api.md`
- CubeK `cubek-matmul` launch API and complex GEMM tests
- tenferro WebGPU dot-general planner and runtime tests

## Decisions

- CubeK owns `C32` split/compose, temporary real buffers, and conjugation signs.
- tenferro keeps only `DotGeneralConfig` normalization, operand packing, and
  provider dispatch.
- `C64` remains unsupported until the real `F64` WebGPU matmul path exists.
- General WebGPU elementwise `conj` remains a follow-up and is not required for
  complex GEMM conjugation.
- CUDA `dot_general` algorithms, cuTENSOR descriptors, workspace allocation, and
  buffer-pool behavior were left unchanged.

## Verification

- `cargo test -p cubek-matmul --test lib complex`
- `cargo test -p cubek-matmul --test lib`
- `cargo fmt --all --check` in the CubeK worktree
- `git diff --check` in the CubeK worktree
- `cargo test -p tenferro-gpu --test public_surface_contract webgpu_c32_dot_general_with_conj_uses_cubek_complex_api -- --nocapture`
- `cargo test -p tenferro-gpu --features webgpu webgpu_c32_dot_general_with_lhs_conj_matches_cpu_when_adapter_available -- --nocapture`
- `cargo test -p tenferro-gpu --features webgpu webgpu_c32_dot_general_with -- --nocapture`
- `cargo test -p tenferro-gpu --features webgpu webgpu_f32_dot_general_with_conj_is_identity_when_adapter_available -- --nocapture`
- `cargo test -p tenferro-gpu --features webgpu webgpu_c32_batched_dot_general_with_both_conj_matches_cpu_when_adapter_available -- --nocapture`
- `cargo test -p tenferro-gpu --features webgpu --test webgpu_matmul_runtime -- --nocapture`
- `cargo test -p tenferro-tensor tests::backend_default_read_tests::default_read_methods_delegate_owned_tensors_and_reject_views -- --nocapture`
- `cargo test -p tenferro-cpu test_dot_general_with_conj_matches_materialized_complex_matmul -- --nocapture`
- `cargo check -p tenferro-gpu --no-default-features --features webgpu,cpu-faer`
- `cargo check -p tenferro-gpu --no-default-features --features cuda,cpu-faer`
- `cargo check -p tenferro-gpu --no-default-features --features cuda,webgpu,cpu-faer`
- `cargo check -p tenferro-einsum --no-default-features --features autodiff,webgpu,cpu-faer`
- `cargo check -p tenferro-einsum --no-default-features --features autodiff,cuda,webgpu,cpu-faer`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
- `cargo test --workspace --release`
- `cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`
- `cargo fmt --all --check`
- `git diff --check`

## Residual Risks

- Runtime performance still uses four real `F32` GEMMs for `C32`; native complex
  kernels or fused epilogues are future CubeK work.
- WebGPU runtime tests depend on adapter availability and return early on
  ordinary runners without an adapter.
- The tenferro dependency is pinned to a tensor4all CubeK fork commit until
  CubeK complex GEMM support is upstreamed or published as tensor4all-owned
  crates.
