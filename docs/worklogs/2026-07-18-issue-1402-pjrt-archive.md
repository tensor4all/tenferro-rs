# Issue #1402 PJRT Hosted Archive

## Session summary

Finished the remaining #1402 GPU contract gap: build `tenferro-xla --features
pjrt` on the hosted Ubuntu 22.04 archive job as a separate nextest archive, and
run PJRT E2E tests on RunPod / fork `CI_gpu` from that archive instead of
`cargo test`.

## Context read

- [#1402](https://github.com/tensor4all/tenferro-rs/issues/1402) acceptance:
  CUDA and PJRT binaries in the hosted artifact; no Cargo on RunPod
- Post-merge thin-profile RunPod evidence (CUDA archive already on `[profile.ci]`)
- `runpod-gpu-test.yml` / `CI_gpu.yml` PJRT steps still compiling on the GPU node

## Chosen design

- Two archives (`cuda-tests.tar.zst` and `pjrt-tests.tar.zst`) rather than one
  multi-feature archive, so `--features cuda` and `--features pjrt` stay
  isolated.
- Keep PJRT plugin / cuDNN / NVCC wheel download on the GPU node (runtime-only).
- Filter nextest with `-E 'test(pjrt_execution)'` to match the previous
  `cargo test ... pjrt_execution` scope.
- Bump archive cache keys (`v11` / `v6`) and rust-cache prefixes.

## Rejected alternatives

- Single archive with package-qualified features: more fragile feature
  unification for little operational gain.
- Shipping the OpenXLA plugin inside the Rust archive: unnecessary; wheels are
  already fetched at runtime.

## Residual risks

- First cold archive after merge builds both CUDA and PJRT; wall-clock may rise
  slightly versus CUDA-only even though RunPod compile (~1.5–2 min) disappears.
- nextest name filter must keep matching `pjrt_execution::*` tests if modules
  are renamed.
