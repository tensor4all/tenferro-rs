# Order and gate for contributing the ports upstream

Status: agreed sequencing. Nothing is published and nothing is filed upstream
yet.

## Order

1. **Mac-side work first** (`#2114`): the Apple/Metal host-visible path is part
   of the port set, so the ports cannot be presented upstream until it exists.
   The Mac side produces:
   - the host-visible work ported onto the 0.11 line (the fork's
     `588e2ee4c` + `11b52669f` live on the 0.10 line today), as its own
     feature-scoped commit(s) with no `t4a-` packaging;
   - the wgpu interop surface exported so tenferro can name the wgpu server and
     resource instead of the CUDA ones (that CUDA naming is a tenferro migration
     mistake, corrected in #2114);
   - Metal evidence: `apple_cpu`, `webgpu_metal_fft`, `apple_context`, the
     storage-provider WebGPU tests, and `cargo check -j 4 -p tenferro-gpu
     --features cuda,webgpu`.
2. **Port tidying on Linux**, once the port set is complete:
   - split the interop port's single commit into two (server interop;
     complex plane reductions);
   - rebuild the fix series as four independent single-file commits —
     `dialect/math.rs` (sound folds), `types/scalar.rs` (`SizedType for BoolType`),
     `opt/passes/inst_combine.rs` (FMA stays real), `frontend/polyfills.rs`
     (accurate `log1p`) — dropping the intermediate `log1p` lowering toggle in
     `cubecl-cpp/src/shared/unary.rs` that the polyfill change reverses;
   - rebase every port onto the then-current upstream `main` (the cubecl ports sit
     39 commits behind, with upstream changes in the same files:
     `cubecl-cuda/src/compute/server.rs` ×4, `cubecl-runtime/src/client.rs` ×6;
     the cubek ports are clean apart from `Cargo.lock`).
3. **Only then** the release branches are regenerated from the rebased ports and
   published, and the pins move. Publishing before the rebase would force a
   0.11.1 release.

## Gate before any upstream contribution

Contributing a port upstream is only safe once it is shown to carry tenferro's
own tests, not just its own:

- `cargo check -j 4 -p tenferro-gpu --features cuda` — 0 errors, 0 warnings;
- A100 suite against the **rebased** ports (the current 209 passed / 0 failed lib
  and 45 passed / 0 failed integration results were measured against the
  pre-rebase base, so they must be re-run);
- the webgpu feature compiles and its macOS-gated tests pass on the Mac,
  including the tenferro-side fixes (wgpu server naming, cubek `Strategy` arm);
- `python3 scripts/ci/run_profile.py fmt` and the repo's clippy invocation stay
  clean;
- the port series is split as above, so each commit is independently reviewable.

A port that cannot be shown to carry the tests is not contributed; it is
reported as a known gap instead.
