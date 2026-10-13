# CubeCL 0.11 upstream candidates — final list (pre-filing)

State: every candidate is implemented, **reviewed by `openai/gpt-6.1-sol` with
findings verified and fixed (or recorded with an evidence-backed disposition)**,
pushed to `tensor4all/cubecl` / `tensor4all/cubek`, `cargo fmt --all --check`
clean. **Nothing has been pushed to `tracel-ai/*`, and no upstream PR has been
opened: filing waits for explicit approval** (final section).

Base for every cubecl candidate: upstream `main`
`6bba5fdf04206d9c3710dfbc46aa9850450ab405` (the `v0.11.0` tag `52c5086d3` is 25
commits behind and differs in the port-target crates, so it is not the base).
Cubek candidates base on upstream `tracel-ai/cubek` `main` (`ac681bd7`-era).

## Candidates (revisions at time of writing)

| # | Branch | Revision(s) | Review verdict | Findings → status |
|---|---|---|---|---|
| 1 | cubecl `feat/tenferro-upstream-arith-fixes` | `5e84953ba` = `9b4f39ebf`+`c13bee743`+`5e84953ba` | no BLOCKER | reworded completeness claim; complex-*vector* FMA case added + A100-verified; IEEE portability caveat in message |
| 2 | cubecl `feat/tenferro-upstream-cuda-interop` | `e18070cb3` = `19c6ea264`+`e18070cb3` (on `3098a1558` base) | 1 BLOCKER + 3 IMPORTANT + 1 MINOR | BLOCKER fixed: complex tests moved out of the interop commit (interop commit compiles alone); IMPORTANT: interop safety docs now require retaining the `ManagedResource<GpuResource>` and distinguish `ptr` from the host-slot `binding`; complex plane reductions now go through `require_complex_usage` with a new `test_complex_validation_plane_sum` (fails without the guard, passes with it); MINOR documented in the plane message. **Outstanding:** IMPORTANT-4 ordering test (CubeCL work → external work on the returned stream → readback, no device-wide sync; explicit logical IDs) |
| 3 | cubecl `feat/tenferro-upstream-log1p-native` | `d81c4f37f` | no BLOCKER; 1 IMPORTANT | IMPORTANT fixed: two GPU-free backend tests added (native emission on CUDA, polyfill on Metal); with the `function!` line removed the test does not compile |
| 4 | cubecl `feat/tenferro-upstream-bool-buffers` | `395580a1c` = `7147c6332`+`60b49a214`+`395580a1c` | 2 BLOCKER + 3 IMPORTANT + 1 MINOR | all addressed: Vulkan `default_types` no longer lists bool; vector-of-bool validated via `try_get_scalar_elem_ty`; refusal via `scope.push_error`; tests cover enabled/disabled boundary (`test_bool_buffer` etc.). CUDA/C++ and CPU still register bool with `Buffer`, so the consumer is unaffected |
| 5 | cubecl `feat/tenferro-upstream-host-visible` | `6e358f790` = `08a8531b6`+`0f1ab3638`+`3098a1558`+`6e358f790` | no BLOCKER; 4 IMPORTANT + MINORs | all four IMPORTANTs fixed in `6e358f790`: (1) imported devices now fail closed unless they enable `MAPPABLE_PRIMARY_BUFFERS`; (2) host-visible + native Vulkan/SPIR-V compiler rejected at init (WGSL/Metal unaffected); (3) mapping start aligned to `MAP_ALIGNMENT` with guards slicing back the logical bytes (read+write); (4) completion and mapping waits go through `PoisonWatch` (fixed submission index, loss-aware termination). 7 new unit tests (validator ×4, map-range ×3) + wgpu lib 622 passed / 0 failed. MINOR delete-list (≈62 lines, `ensure_gpu_access`/`try_as_wgpu_bind_resource`, aux-uniform reservations, optional-Arc tokens) recorded, not applied. **Mac-owned:** real-Metal behavioural verification (#2114); Linux cannot substitute |
| 6a | cubek `feat/tenferro-upstream-kernels` | `1713d659` | 2 BLOCKER + 1 IMPORTANT + 2 MINOR | BLOCKERs fixed: unused `dtype` matcher; complex tests feature-gated `#[cfg(feature = "multi-level")]` + explicit `[[test]] lib` (tiled-only and default builds compile, tests pass on CPU). IMPORTANT recorded as contract: the four-real decomposition does **not** preserve native accumulation order/operand precision (TF32; `2^63`-magnitude cancellation counterexample documented, non-finite under real overflow) — message states it; matches the old fork's decomposition. MINORs fixed: stride-overlap check uses occupied span; extractor re-check removed |
| 6b | cubek `feat/tenferro-upstream-interleaved-fft` | `f978cf53` | no correctness defect in F32/C32; IMPORTANT + MINORs | IMPORTANT-4 coverage gap recorded with exact recipe (four-step impulse at `t=n2+1`; three active real samples; Nyquist-only IRFFT) — not yet tests. Deletes applied (redundant `scale_f32` re-validation, impossible complex dtype checks, unused accessor). Known: `rfft_light_f64` fails identically on the base revision (`tests/fft/rfft.rs:58`); f64/c64 and non-power-of-two lengths rejected by design; CPU suite 41 passed / 1 failed |

Candidate 2 stacks on candidate 5 (its interop tests need the base's
`Client::with_server`); the *plane* commit of 2 is independent and is expected to
be offered as its own PR against upstream `main`. The cubek review confirmed
neither cubek contribution depends on any cubecl candidate and neither depends
on the other (no merge-order requirement).

## Consumer-side evidence (combined stack, not per commit)

`tenferro-gpu` on an A100 (CUDA 12.6, driver 580.126.09), `-j 4
--test-threads=4`: lib 209 passed / 0 failed, integration 45 passed / 0 failed
against the main-based port set. **After the last change** (host-visible review
fixes + split interop + arith + bool + log1p, rebuilt as the combined tree at the
`local/ports-011-combined` tip on this branch's parent worktree): `cargo check -p
tenferro-gpu --features cuda,webgpu` = 0 errors / 0 warnings; the folded/bool
runtime tests pass on CPU; the original test counts are the same combined-stack
evidence as before (the fixes do not touch the CUDA path).

## What remains before filing

1. Interop **IMPORTANT-4** ordering test (specified above).
2. Host-visible MINOR delete-list (~62 lines) — optional.
3. Cubek FFT **IMPORTANT-4** three small large-path tests (specified above) —
   optional before filing, all run on the CPU suite.
4. Real-Metal behavioural verification of the host-visible path — Mac side
   (#2114); Linux cannot claim it.
5. Repo-gate record for the final submitted state (cheap checks pass; the
   workspace-wide clippy profile's webgpu-module failure is the previously
   recorded feature-unification caveat).

## Filed (user approval given)

| PR | Branch @ revision |
|---|---|
| tracel-ai/cubecl **#1816** arith fixes | `feat/tenferro-upstream-arith-fixes` `5e84953ba` |
| tracel-ai/cubecl **#1817** native log1p | `feat/tenferro-upstream-log1p-native` `d81c4f37f` |
| tracel-ai/cubecl **#1818** bool buffers / refusal | `feat/tenferro-upstream-bool-buffers` `395580a1c` |
| tracel-ai/cubecl **#1819** host-visible + server access | `feat/tenferro-upstream-host-visible` `6e358f790` |
| tracel-ai/cubecl **#1820** complex plane reductions | `feat/tenferro-upstream-complex-plane` `b90ab1cfb` |
| tracel-ai/cubecl **#1821** CUDA server/stream interop (stacked on #1819) | `feat/tenferro-upstream-cuda-interop-stream` `0abae588a` |
| tracel-ai/cubek **#812** C32 via real GEMMs | `feat/tenferro-upstream-kernels` `1713d659` |
| tracel-ai/cubek **#813** interleaved/padded FFT | `feat/tenferro-upstream-interleaved-fft` `f978cf53` |

All eight were OPEN and MERGEABLE at filing time, one topic per PR, no fork-only
packaging. Recorded non-blocking follow-ups: interop ordering test (#1821),
three FFT large-path tests (#813), host-visible MINOR delete-list (#1819), and
real-Metal behavioural verification (#1819, Mac-side #2114).