# Order and gate for contributing the ports upstream

Status: agreed sequencing. The same procedure, with owners and current state,
is tracked in the umbrella issue #2116. Nothing is published and nothing is
filed upstream yet.

## Order

Base: upstream `v0.11.0` (`52c5086d3`) and `v0.3.0` (`b6ac1f2ee`) are published.
Verified at `v0.11.0`, the five fixes and the four ports are still absent, so the
whole extension set is contributed as 0.12.0-line work.

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
   - rebase every port onto the stable upstream tags — cubecl `v0.11.0`
     (`52c5086d3`), whose 16 commits since the port base changed 68 files in the
     port-target crates; cubek `v0.3.0` (`b6ac1f2ee`), 8 commits with the target
     crates untouched. The contribution then targets the 0.12.0 line, since
     0.11.0 is already released.
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

## Process for every PR candidate

1. An independent review by `openai/gpt-6.1-sol` (read-only, `independent-review`
   skill), with its findings verified against the source before anything is acted
   on.
2. The user's explicit approval.
Nothing is pushed to `tracel-ai/*` or filed before both. The arithmetic fix
series is already in that shape and pushed as
`tensor4all/cubecl` `feat/tenferro-upstream-arith-fixes` (`daf31e033`, four
single-file commits on `6bba5fdf`).

## Gate met: the rebased run

Base decided as upstream `main` `6bba5fdf` (the Mac ports are built on it; the
`v0.11.0` tag is 25 commits behind with 105 files changed in the port-target
crates). Against `local/ports-011-combined` @ `2747c68a1` — `6bba5fdf` + three
wgpu host-visible commits + the CUDA-interop port + two arithmetic fixes + sized
`BoolType` + native `log1p` — on the A100:

- `cargo check -j 4 -p tenferro-gpu --features cuda` — 0 errors
- `cargo check -j 4 -p tenferro-gpu --features cuda,webgpu` — 0 errors
- CUDA lib tests **209 passed / 0 failed**; integration **45 passed / 0 failed**

Consumer-side fixes are on `migration/cubecl-0.11-upstream` @ `a8ae113d1`.
Metal evidence: #2114.

## Candidates and what each still needs

| # | branch | content | open item |
| --- | --- | --- | --- |
| 1 | `feat/tenferro-upstream-arith-fixes` `5941a32b9` | sound folds; FMA stays real | a test per commit |
| 2 | (from the Mac stack) | CUDA interop | split into server interop + complex plane reductions (`with_server` hunk already absent) |
| 3 | `feat/tenferro-upstream-log1p-native` `c02cc079a` | native `log1p` in C++/LLVM, polyfill only for Metal | upstream accuracy test |
| 4 | `feat/tenferro-upstream-bool-buffers` `5d0fa26b5` | sized `BoolType` | part 2: the refusal (narrow the wgpu tables, shared compile-time check, tests) |
| 5 | `feat/tenferro-upstream-host-visible` `3098a1558` | completion, primary memory, device-server accessor | — |
| 6 | cubek `feat/tenferro-upstream-kernels`, `...-interleaved-fft` | matmul C32, interleaved FFT | — |

Each goes through the gpt-6.1-sol review and then the user's approval before
filing.

