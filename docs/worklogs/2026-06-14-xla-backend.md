# XLA Backend

## Summary

Added the experimental `tenferro-xla` crate for static-shaped StableHLO
lowering, runtime PJRT plugin loading through environment variables, external
OpenXLA execution verification hooks, and online documentation for XLA/PJRT,
CUDA, and cuTENSOR setup. The online XLA tutorial uses a checked fixed-shape
N-ary einsum example that expands to standard `dot_general` operations through
the extension standard-op lowering hook.

## Context Read

- `AGENTS.md`
- `REPOSITORY_RULES.md`
- shared tensor4all common, Rust, performance, numerical, and docs rules
- `docs/superpowers/specs/2026-06-14-xla-backend-design.md`
- `crates/tenferro-runtime/src/graph/program.rs`
- `crates/tenferro-runtime/src/exec.rs`
- `crates/tenferro-core-ops/src/catalog.rs`
- `crates/tenferro-einsum/src/traced.rs`
- `crates/tenferro-internal-ops/src/ext_op.rs`
- `crates/tenferro-xla/src/lowering/program.rs`
- OpenXLA StableHLO and `run_hlo_module` examples under `/home/shinaoka/tensor4all/xla`

## Decisions

- Kept XLA as a peer executor crate instead of adding a `TensorBackend`.
- Added a narrow read-only `GraphProgram::lowering_view()` accessor for owner
  lowering integrations.
- Lowered only exact static shapes and `F32`/`F64` in the first subset.
- Rejected unsupported dtypes, dynamic upper-bound extents, and unsupported ops
  before PJRT.
- Used runtime plugin loading via `TENFERRO_PJRT_PLUGIN` and
  `TENFERRO_PJRT_GPU_PLUGIN`; no compile-time XLA link was added.
- Treated StableHLO tensor dimension order and PJRT host memory order as
  separate concerns. Physical host-order conversion stays in `tenferro-xla`.
- Inserted a transpose after batched StableHLO `dot_general` to preserve
  tenferro's batch-trailing logical result contract.
- Kept XLA lowering limited to standard ops. Fixed-shape N-ary einsum reaches
  XLA through `ExtensionOp::lower_to_standard_ops`; dynamic extension-runtime
  execution remains on the native runtime path.
- Added an external `run_hlo_module` execution test instead of claiming text
  snapshots prove generated StableHLO correctness. The test covers both the
  initial direct-op module and the fixed-shape N-ary einsum module generated
  through extension standard-op lowering.

## Verification

- `cargo test -p tenferro-runtime --test public_surface_contract graph_program_exposes_read_only_lowering_view_for_owner_crates`
- `cargo test -p tenferro-runtime --doc`
- `cargo test -p tenferro-internal-ops --doc`
- `cargo test -p tenferro-xla --doc`
- `cargo test -p tenferro-xla --tests`
- `cargo test -p tenferro-xla --test stablehlo_lowering lowers_concrete_nary_einsum_via_standard_ops -- --nocapture`
- `cargo test -p tenferro-xla --test stablehlo_lowering lowers_static_symbolic_nary_einsum_extension_via_standard_ops -- --nocapture`
- `cargo test -p tenferro-xla --test unsupported rejects_extension_without_standard_op_lowering -- --nocapture`
- `cargo test -p tenferro-einsum --test traced_extension -- --nocapture`
- `cargo test -p tenferro-xla --features pjrt --tests`
- `TENFERRO_XLA_RUN_HLO_MODULE=/home/shinaoka/tensor4all/xla/bazel-bin/xla/tools/run_hlo_module TENFERRO_XLA_RUN_HLO_PLATFORM=Host cargo test -p tenferro-xla --test xla_tool_execution -- --nocapture`
- `/home/shinaoka/.local/bin/bazelisk build --config=cuda --repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES=sm_80 //xla/tools:run_hlo_module`
- `CUDA_PATH=/usr/local/cuda-12.6 LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH TENFERRO_XLA_RUN_HLO_MODULE=/home/shinaoka/tensor4all/xla/bazel-bin/xla/tools/run_hlo_module TENFERRO_XLA_RUN_HLO_PLATFORM=CUDA cargo test -p tenferro-xla --test xla_tool_execution -- --nocapture`
- `cargo test -p tenferro-tutorial-code --release tutorial_binaries_run_successfully -- --nocapture`
- `python3 scripts/check-doc-snippets.py --check`
- `bash scripts/build_docs_site.sh`
- `python3 scripts/check-api-consistency.py --fail-on-findings`
- `python3 scripts/check-publish-layout.py`

External OpenXLA verification is environment-gated through
`TENFERRO_XLA_RUN_HLO_MODULE`. A local Bazelisk install built OpenXLA
`//xla/tools:run_hlo_module`, and the Host platform accepted and executed the
generated direct-op and N-ary einsum StableHLO modules.

CUDA platform verification was attempted with the Host-built tool and failed
because the tool registered only `Host` and `Interpreter`. A CUDA-enabled
OpenXLA rebuild was then attempted with `--config=cuda`; the first attempt
failed because the default CUDA architecture list included unsupported
`compute_35` for CUDA 12.9. The A100-specific rebuild with
`--repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES=sm_80` succeeded, and the CUDA
platform accepted and executed the generated StableHLO modules with the local
CUDA 12.6 and cuTENSOR library paths.

## Residual Risks

- Rust-side PJRT compile/upload/execute/download wrappers are not complete yet;
  the current Rust `pjrt` feature fixes plugin loading and leaves execution
  verification to OpenXLA's tool.
- CUDA execution through `run_hlo_module` was verified on this A100 machine,
  but remains dependent on local OpenXLA build success and CUDA/cuTENSOR
  dynamic-library configuration.
- The initial StableHLO subset is intentionally small and does not cover
  integer, boolean, complex, linalg, indexing, or extension operations without
  a fixed-shape standard-op lowering hook.
