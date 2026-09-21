# #1855: publish the fork FFT crate as t4a-cubek-fft

## Decisions

- Publish the fork's FFT crate under a fork-owned name and depend on that name.
  The crates.io name `cubek-fft` belongs to upstream, and the fork built its
  copy with `publish = false`, so a published `tenferro-gpu` resolved upstream's
  incompatible crate for the `webgpu` feature. Publishing as `t4a-cubek-fft`
  keeps the pinned implementation and its CI coverage and makes the published
  manifest resolve content that matches the pin.
- Rejected alternatives from #1855: adopting upstream `cubek-fft` with upstream
  `cubecl` (would put two `cubecl` families in one graph and mix them across the
  tenferro boundary) and dropping `webgpu` from the published feature set
  (removes a shipped feature instead of fixing its resolution).
- Keep the library name `cubek_fft` via `[lib] name`, so tenferro's
  `use cubek_fft::` and the `cubek-fft` dependency key are unchanged; only the
  package name moves. This follows the existing `t4a-cubek-matmul` and
  `t4a-cubek-std` pattern.
- Pin every cubek crate to the single revision that carries the rename. Mixing
  the 0.2.1 tag with the rename commit made cargo build two cubek sources, and
  `tenferro-gpu`'s `cubek_std` types then failed to unify (observed E0308 on
  `InputBinding` with two `cubek_std` instances in the graph).

## Verification conclusions and constraints

- `cargo publish -p t4a-cubek-fft --dry-run` and the real `0.2.1` publish
  succeeded from the rename commit, which is now on the fork's
  `release/t4a-0.2.1` line.
- With every git pin resolved from crates.io (the state a published crate sees),
  `cargo check -p tenferro-gpu --features webgpu` compiles; before the rename
  the same configuration failed in upstream's `cubek-fft` with
  `cannot find std in cubecl`. The git-pin configuration compiles too.
- `scripts/check-git-pin-content.py` passes with no exception, so
  `scripts/git-pin-content-exceptions.toml` carries no entries again.
- Not verified here: WebGPU runtime execution, which needs WebGPU hardware. CI
  covers CUDA on RunPod; the wgpu path keeps its existing compile-level CI
  coverage.
