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

## 2026-06-20 Phase 1 Elementwise Update

### Summary

Expanded the Phase 1 XLA path from the initial `Add`/`Multiply`/`Negate`
elementwise subset to real-floating analytic elementwise lowering and added a
Rust-side PJRT execution API boundary.

### Context Read

- `crates/tenferro-runtime/src/graph/lowering_view.rs`
- `crates/tenferro-runtime/src/graph/lowering_view/tests.rs`
- `crates/tenferro-xla/src/lowering/program.rs`
- `crates/tenferro-xla/src/pjrt/sys.rs`
- `crates/tenferro-xla/src/pjrt/plugin.rs`
- `crates/tenferro-xla/src/executor.rs`
- `docs/design/xla-backend.md`
- `docs/guides/xla.md`
- OpenXLA `xla/pjrt/c/pjrt_c_api.h` from the sibling `../xla` checkout

### Decisions

- Added `GraphOpView` variants for Phase 1 real-floating elementwise ops:
  `Divide`, `Abs`, `Exp`, `Log`, `Sin`, `Cos`, `Tanh`, `Sqrt`, `Rsqrt`,
  `Pow`, `Expm1`, and `Log1p`.
- Mapped those variants directly to StableHLO ops for exact static `F32`/`F64`
  programs.
- Kept `Compare`, `Select`, `Maximum`, `Minimum`, `Clamp`, `Sign`, `Conj`,
  integer, `Bool`, and complex support outside Phase 1.
- Added `XlaExecutor::run_with_inputs` and `run_many_with_inputs` as the public
  Rust-side execution boundary. The methods require a loaded PJRT plugin and
  return explicit typed errors when the `pjrt` feature or plugin is absent.
- Kept host memory order handling explicit in `tenferro-xla`: inputs pass
  column-major `byte_strides` to PJRT so upload can read compact tenferro host
  buffers directly, and outputs still convert the downloaded default host
  layout back to tenferro column-major tensors.
- Bound only the PJRT C API prefix needed for single-device Phase 1 execution:
  client creation, addressable-device lookup, MLIR compile, host-buffer upload,
  executable output count, execute, host download, and object/event cleanup.
- After testing against the prebuilt OpenXLA/JAX CUDA PJRT plugin, changed PJRT
  compile to pass a minimal serialized `CompileOptionsProto` with
  `num_replicas = 1` and `num_partitions = 1`. The CUDA plugin aborts in device
  assignment if those values are left at their protobuf default of zero.

### Verification

- `cargo test -p tenferro-runtime graph::lowering_view --lib`
- `cargo test -p tenferro-xla --test stablehlo_lowering`
- `cargo test -p tenferro-xla --test unsupported`
- `cargo test -p tenferro-xla --test public_api`
- `cargo test -p tenferro-xla --features pjrt --test public_api`
- `cargo fmt --all --check`
- `cargo test -p tenferro-xla --tests`
- `cargo test -p tenferro-xla --features pjrt --tests`
- `cargo test -p tenferro-xla --doc`
- `cargo test -p tenferro-xla --features pjrt --doc`
- `cargo clippy -p tenferro-xla --all-targets -- -D warnings`
- `cargo clippy -p tenferro-xla --features pjrt --all-targets -- -D warnings`
- 2026-06-21 external OpenXLA Host verification:
  - `git -C ../xla fetch origin && git -C ../xla pull --ff-only origin main && git -C ../xla rev-parse --short HEAD` -> `3b0ff804f2`
  - `/home/shinaoka/.local/bin/bazelisk build //xla/tools:run_hlo_module`
  - `TENFERRO_XLA_RUN_HLO_MODULE=/home/shinaoka/tensor4all/xla/bazel-bin/xla/tools/run_hlo_module TENFERRO_XLA_RUN_HLO_PLATFORM=Host cargo test -p tenferro-xla --test xla_tool_execution -- --nocapture`
  - Result: 3/3 execution tests passed, covering the direct static graph,
    fixed-shape N-ary einsum, and Phase 1 elementwise StableHLO graph.
- 2026-06-21 Rust-to-PJRT CUDA verification with prebuilt wheels:
  - `python3 -m pip download --only-binary=:all: --no-deps --dest /tmp/tenferro-openxla-prebuilt jaxlib jax-cuda12-plugin==0.10.2 jax-cuda12-pjrt==0.10.2 'nvidia-cudnn-cu12>=9.8,<10.0'`
  - `nm -D /tmp/tenferro-openxla-prebuilt/jax-cuda12-pjrt-unpacked/jax_plugins/xla_cuda12/xla_cuda_plugin.so | rg GetPjrtApi`
  - `TENFERRO_PJRT_PLUGIN=/tmp/tenferro-openxla-prebuilt/jax-cuda12-pjrt-unpacked/jax_plugins/xla_cuda12/xla_cuda_plugin.so LD_LIBRARY_PATH=/tmp/tenferro-openxla-prebuilt/nvidia-cudnn-cu12-unpacked/nvidia/cudnn/lib:/usr/local/cuda-12.5/targets/x86_64-linux/lib:/usr/local/cuda-12.6/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH cargo test -p tenferro-xla --features pjrt --test pjrt_execution -- --nocapture`
  - Result: 3/3 Rust E2E tests passed, covering fixed-shape N-ary einsum,
    Phase 1 elementwise ops, and a fixed-shape N-ary einsum followed by
    elementwise ops through compile, upload, execute, download, and value
    compare.

### Residual Risks

- The new Rust-side PJRT execution path was verified against the prebuilt CUDA
  PJRT plugin on one A100 machine, but not yet against a standalone CPU PJRT
  plugin.
- The PJRT binding mirrors the current OpenXLA C API prefix from the local
  sibling checkout. Future PJRT API drift should be caught with real-plugin
  smoke tests before expanding the execution subset.
