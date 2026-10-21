# Adopt the secular-root SVD correction

## Decisions

- Adopt published t4a-faer 0.24.8 and tlinalg 0.4.4 with compatible Cargo ranges. The older tlinalg release itself pinned the faulty faer, so both requirements must move. Leave the unrelated BLAS provider requirement unchanged.
- Repair the CPU CI blockers in the same finished adoption change: architecture-gate existing AVX2 softmax calls while retaining the scalar fallback; reuse the engine's existing buffer pool for realfft lanes and provider scratch.
- Do not replace the stale complex-FFT allocation expectation with an uninformative zero. Count real-sized lanes too, require zero warm large allocations in both real transform directions, and preserve inputs and numerical results.
- Prime every test-owned Rayon worker before the process-global allocation window. Tiny caller-only operations cannot synchronize asynchronous worker startup. Keep all-thread counting, the two-worker execution cases, and the original zero-allocation budgets; no sleeps or retries.

## Verification conclusions and constraints

- The stronger FFT regression detects two fresh working buffers (65,552 bytes) before the production correction and passes after reuse, including inverse reconstruction and source preservation. The existing FFT numerical/parallel/managed-buffer suite and native typed-allocation checks also pass.
- The aarch64 Apple Accelerate cross-check rejects the old softmax call sites and accepts the architecture-gated calls. Linux cross-compilation does not establish Apple runtime or Metal behavior; the required macOS lane remains authoritative.
- Native (3,443 tests, 311 skipped) and system-OpenBLAS (3,255 tests) workspace suites, both doctest suites, scalar-composition object verification, BLAS interop, and three provider-injection cases pass. Full-suite validation sets the ambient Rayon width to four; the allocation fixtures independently assert one/two-worker budgets. Local UI snapshots are checked without kache's diagnostic-path remapping.
- Exact-commit local A100 evidence is recorded with the PR. Preserve required checks and the existing locally validated GPU route; no RunPod execution, workflow changes or protection changes are part of this correction.
- Allocation counts establish buffer reuse, not a timing speedup. No performance benchmark or full CUDA matrix is claimed.
