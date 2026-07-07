# Issues 1315-1320 Capability, Output, Integer, And Docs Batch

## Summary

This branch implements the accepted plans for #1315, #1316, #1317, #1318,
and #1320 as one reviewable batch:

- #1315 adds backend capability descriptors for CPU and CUDA/CubeCL and
  records output dtype, read-input, write-output, strided-output, and
  accumulation axes.
- #1316 adds a dtype-dispatch seam in `tenferro-tensor` and uses it to keep
  dtype policy checks centralized rather than copied through backends.
- #1317 defines output/write vocabulary and adds fallback overwrite APIs plus
  dot accumulation helpers without claiming native CUDA write-output support.
- #1318 reframes README/getting-started docs around direct typed tensor use
  first, opt-in AD second, and adds a runnable direct linalg quickstart.
- #1320 completes integer CPU/CUDA parity for the accepted batch-2 operation
  set, including `sub`, integer unary/order/reduction support, and checked
  integer `div`/`rem`/`pow` domain errors.

## Context Read

- `AGENTS.md`
- `REPOSITORY_RULES.md`
- shared tensor4all common, Rust, performance, numerical, and docs/test rules
- GitHub issue bodies and comments for #1315, #1316, #1317, #1318, and #1320
- `docs/design/gpu-backend-design.md`
- `docs/guides/devices-and-gpu.md`
- `crates/tenferro-core-ops/src/catalog.rs`
- `crates/tenferro-tensor/src/backend.rs`
- `crates/tenferro-tensor/src/capability.rs`
- `crates/tenferro-tensor/src/dispatch.rs`
- `crates/tenferro-cpu/src/elementwise.rs`
- `crates/tenferro-cpu/src/reduction.rs`
- `crates/tenferro-gpu/src/cubecl/mod.rs`
- `crates/tenferro-gpu/src/kernels/elementwise.rs`
- `crates/tenferro-einsum/src/concrete.rs`

## Decisions

- Keep capability descriptors informational and queryable. They describe the
  current public backend surface but do not become a second dispatch path.
- Represent fallback output writes explicitly as `SupportLevel::FallbackCopy`.
  The default `_into` methods compute into a temporary and copy into the
  caller output; backend tables should not label those paths as native.
- Make integer overflow semantics explicit two's-complement wrapping for CPU
  and CUDA. Integer division by zero and negative integer exponents return
  typed errors; `MIN / -1` and `MIN % -1` use wrapping Rust/CUDA-compatible
  semantics instead of panicking.
- Keep `rem` out of CPU elementwise fusion for now. The backend executes it as
  a normal elementwise operation and rejects fusion plans containing remainder
  until a strided-kernel remainder primitive exists.
- Add a runnable direct linalg quickstart as the source of truth for copied
  README and docs snippets. The quickstart uses typed tensors and explicit
  operation crates instead of implying a root facade crate.

## Verification Performed

- Focused tensor/output/einsum tests for `_into`, dot accumulation, capability
  descriptors, integer dtype propagation, and exec dispatch.
- Focused CUDA tests for integer add/mul/order/select parity, checked integer
  domain errors, and descriptor parity smoke cases with CUDA 12.6.
- `cargo test -p tenferro-tutorial-code tutorial_binaries_run_successfully -- --nocapture`
- `cargo run -p tenferro-tutorial-code --bin direct_linalg_quickstart`
- `python3 scripts/check-doc-snippets.py --check`
- `python3 scripts/check-docs-site.py`
- `cargo test -p tenferro-tensor --doc`
- `CARGO_BUILD_JOBS=1 RUST_TEST_THREADS=1 cargo test -p tenferro-einsum --doc -- --test-threads=1`
- `cargo fmt --all --check`
- `CARGO_BUILD_JOBS=2 cargo clippy --workspace --all-targets -- -D warnings`
- `CARGO_BUILD_JOBS=2 cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
- `CARGO_BUILD_JOBS=2 cargo test --workspace --release`
- `CARGO_BUILD_JOBS=2 cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `CARGO_BUILD_JOBS=2 cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`
- `python3 scripts/repository-rules-review.py --base origin/main --head HEAD --output-json /tmp/repository-rules-review.json`

## Residual Risks

- CUDA verification used the installed `/usr/local/cuda-12.6` toolkit rather
  than the documented CUDA 12.8 example path. The targeted CUDA tests passed
  with the installed toolkit and driver.
- `_into` elementwise defaults are intentionally fallback-copy paths. Native
  backend-specific output writes remain future optimization work and should be
  reflected as `Native` only after backend implementations land.
- `rem` is not fused in CPU elementwise plans. If a future strided-kernel
  remainder op lands, the current explicit filter and tests should be updated
  with that backend capability change.
