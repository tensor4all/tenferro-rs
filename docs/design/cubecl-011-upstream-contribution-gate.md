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
| 1 | `feat/tenferro-upstream-arith-fixes` `23b6cd7a5` | sound folds; FMA stays real | **done**: each commit carries a device-verified test |
| 2 | `feat/tenferro-upstream-cuda-interop` `9344bd4e0` | CUDA interop, split | **done**: two topic commits, content-identical to the Mac's single commit (`git diff 1883fd93b HEAD` empty) |
| 3 | `feat/tenferro-upstream-log1p-native` `5fc8da47e` | native `log1p` in the C++ backend, polyfill only for Metal | upstream accuracy test: needs a dtype-correct shape first (the LLVM backend has no native `log1p` lowering, so a generic test would fail there) |
| 4 | `feat/tenferro-upstream-bool-buffers` `f4afe68f0` | sized `BoolType`; the wgpu tables stop claiming bool buffers | **the refusal call site** (see below) |
| 5 | `feat/tenferro-upstream-host-visible` `3098a1558` | completion, primary memory, device-server accessor | — |
| 6 | cubek `feat/tenferro-upstream-kernels`, `...-interleaved-fft` | matmul C32, interleaved FFT | — |

Each goes through the gpt-6.1-sol review and then the user's approval before
filing.

## Verification of the new candidate tests (this session)

- fold test (`test_ieee_constant_folds`, in the folds commit): CPU backend —
  **passes with the change, fails without it** (`cargo test -p cubecl-cpu --lib --
  test_ieee_constant_folds`, the pre-fix failure is the `v % 1.0 -> v` assertion).
- complex mul-add test (`test_complex_mul_add_cf32/cf64`, in the FMA commit):
  A100 with `--features cpp` — **both pass with the change and fail to launch
  without it** (`cargo test -p cubecl-cuda --features cpp --lib --
  test_complex_mul_add`; the pre-fix run reports "A launch error happened").
- `log1p`: the consumer's own `test_log1p_small_x_f32_precision` is the reproducer
  (fails against the pre-fix revision, passes with the C++ native lowering), and
  it is cited in the commit. A portable upstream accuracy test is not yet written
  for the reason above.

## bool part 2: what is done and what the refusal still needs

Done and pushed (`feat/tenferro-upstream-bool-buffers` @ `f4afe68f0`,
`local/ports-011-combined` @ `93db202c3`):

- both wgpu registrations (`backend/wgsl.rs`, `backend/vulkan.rs`) no longer claim
  `TypeUsage::Buffer` for `ElemType::Bool`, with the reason written down;
- `Features::supports_buffer(elem)` (and `DeviceProperties::supports_buffer_type`)
  answers exactly that question, with a unit test (`cargo test -p cubecl-ir --lib
  -- supports_buffer`: a type usable as a value is not necessarily storable);
- `cargo check -p cubecl-wgpu` and the consumer's `--features cuda,webgpu` check
  both pass with the change.

**The refusal itself is not implemented.** Where it has to go was located by
reading the two call paths:

- `WgslCompiler::compile_kernel(kernel, definition, options)` has the IR but no
  `DeviceProperties`; `validate_ir(&self, repr, props)` has the properties but only
  the compiled representation, so neither has both.
- Kernels reach compilation through `KernelLoader<WgpuPipelines<C>>::compile_queued`
  (`crates/cubecl-wgpu/src/compute/server.rs:89,603`), which is backend-agnostic and
  has neither.
- So the check belongs at the wgpu launch/compile boundary where the server holds
  the device and the kernel: a pure `fn refuse_unsupported_buffers(ctx, entry_func,
  props) -> Result<(), String>` that walks the kernel's buffer arguments exactly as
  `cubecl-opt/src/passes/annotate_buffer_visibility.rs:32-44` does (find
  `BufferBindingAttr` args, take each value's element type, ask
  `supports_buffer_type`), called from the wgpu compile path. Being a pure function
  over hand-built IR and a hand-built feature table, it is unit-testable without a
  device, which is what makes it verifiable here rather than only in wgpu CI.
- Still owed: that call site, its unit test, and a positive runtime test for a
  `&[bool]` kernel on the backends that can store it (CUDA/CPU), guarded by the
  same usage check the existing tests use.

