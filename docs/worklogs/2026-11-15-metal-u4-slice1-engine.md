# Metal backend U4 slice 1: the engine, the shared domain and the elementwise family (#2402)

## Decisions

- U4 lands in three slices, in this order, and each refusal names the slice that serves it.
  Slice 1 (this one) is the engine boundary the rest of the work attaches to: probe-first
  discovery, `MetalContext` (one device, one queue, one shared allocation domain, one paired
  `CpuEngine`), the `MetalStorage` host admission, explicit `upload`/`download` with byte
  accounting, the flat elementwise family the Metal catalogue serves for f32 and Complex32
  (`add`, `sub`, `mul`, `div` on f32, `neg`, `abs`, `sign`, `conj`, and the in-place form over
  a caller-owned output), and a same-dtype `cast` (one exact copy). Slices 2 and 3 follow with
  the structural/materialization kernels and then MPS reductions; U5 is MPS GEMM and U6 the
  MSL fusion emitter over the existing fusion planner.
- f64 and Complex64 are refused by the *engine*, not by the allocator. `MetalAllocationDomain`
  serves every `DType` so a CPU result allocated in the same domain chains back into a Metal
  operation with no transfer (design §4.3, §9); it is `MetalEngine` that returns
  `Unsupported` naming `MetalContext::cpu_engine`, because MSL rejects `double`.
- One `Mutex<Runtime>` serialises allocation, MSL compile, pipeline bind, launch encoding,
  commit and the host-admission check. Waits do not happen under it: a submission records its
  `Completion` on every allocation it named (read or write) and the host touch waits on the
  newest one, which on an in-order queue is waiting for everything earlier. `Runtime` had to
  become `Send` for the context to be `Sync`; that is the sibling change in
  `t4a-gpu-backend` (`t4a-metal-runtime` 0.1.2, published from
  [t4a-gpu-backend@1a8e409](https://github.com/tensor4all/t4a-gpu-backend/commit/1a8e4096b632fd176a29bbb73fb49ee246810e99),
  PR [t4a-gpu-backend#17](https://github.com/tensor4all/t4a-gpu-backend/pull/17); 0.1.1
  shipped before the `Send` impl).
- A GPU submission is admitted only when no live `map_read`/`map_write` guard names an operand;
  otherwise it returns `MappedForHost` and commits nothing. The admission check and the commit
  hold the same lock, so no guard can appear between them. `shared_host_slice` cannot raise
  that count (it has no guard to drop), so this slice removes that window structurally instead:
  every GPU write targets an allocation created inside the same call and handed out only after
  the commit, and a GPU read concurrent with a host borrow is harmless. Slice 2's in-place
  materialization kernel has to revisit it.
- No build-time framework link and no `newBufferWithBytesNoCopy`. The frameworks are `dlopen`ed
  on first use; `scripts/check-macos-no-framework-link.sh` proves it with `otool -L` on the
  macOS lane. Every allocation is a genuine shared-mode buffer the domain owns.
- The capability table is `const` data in `crates/tenferro-gpu/src/metal/capability.rs`, and
  the engine's `backend_id()` is `BackendId::Other("metal")`: adding a `BackendId` variant is a
  public enum change and belongs to a design decision, not to this slice.

Review: the plan was reviewed by `ollama-cloud/glm-5.3-flash` before implementation
(VERDICT: CHANGES, seven findings, all addressed) and the finished diff by `gpt-6.1-sol`
(VERDICT: CHANGES, five IMPORTANT findings, all fixed): the host-admission check moved inside the
runtime lock, host waits moved outside it, the framework-link check now finds the `ci`-profile
binary, the temporary git pins gave way to registry requirements once the runtime and catalogue
releases landed, and the
synchronization bars now prove what they claim (a wait counts only when there was a completion to
wait for, and a host overwrite after a read-only launch is checked by value). The review also led
to refusing overlapping host mappings (`OverlappingHostMapping`), which the draft allowed.

## Verification conclusions and constraints

- `TENFERRO_REQUIRE_METAL=1 cargo nextest run --cargo-profile ci -p tenferro-gpu --features
  metal --test metal_engine` (the hosted-lane command): 19 passed on the Apple M5 Max. The bars that matter: bit-exact
  round trips at lengths {0, 1, 3, 255, 256, 257, 4096, 100_003}; elementwise equality with the
  CPU reference (0 ulp for add/sub/neg/conj on the same inputs, bit-exact division); a live
  host guard refusing a submission with `MappedForHost`; a host map waiting on the launch that
  read the allocation; f64 on the paired `CpuEngine` over the shared domain with no upload; a
  foreign context's allocation refused; a second host mapping refused as overlapping; no deadlock
  across two threads × 16 allocate/launch/download rounds with the computed values checked; an
  empty operand allocating without a submission.
- Complex32 `abs` is the baseline's scaled hypot (`scale * sqrt(r*r + i*i)` after dividing by
  `max(|re|, |im|)`), so its agreement with the platform `hypot` is asserted within 4 ulp, not
  bit-exactly. That is a property of the catalogue entry, not of this engine.
- Benchmark (M5 Max, shared buffers, one op then one wait; `cargo bench -p tenferro-gpu
  --features metal --bench metal_engine`, rerun at the registry-pinned head). f32 add 1e7
  elements: 3.28 ms Metal vs 0.91 ms CPU on the same buffers (3.05 vs 11.02 GFLOP/s); ten
  submissions then one wait: 9.10 ms, i.e. 0.91 ms per op. Single-op latency below about 1e6
  elements is dominated by roughly 0.5 ms of submission/wait overhead (1e3 elements: 0.52 ms
  Metal vs 0.5 µs CPU). Upload 17–19 GB/s, download 72–76 GB/s at 1e6–1e7 elements. Elementwise
  work is therefore not where this backend wins on a shared-memory architecture; it is the
  correctness boundary the MPS GEMM/FFT/reduction slices (U5 in part, U4 slice 3) and fusion
  (U6) build on.
- Remaining residuals, each recorded in the plan: f64/BF16 are out of scope by the issue;
  non-compact operands and borrowed views are refused rather than staged (slice 2); Metal's
  fast-math subnormal flush needs a cross-repo compile-options seam to change and is not
  measured here; `Runtime` is `Send` but deliberately not `Sync`.
- Repo gate now passes: `scripts/check-git-pin-content.py` reports `0 warning(s), no content
  drift` with the registry requirements `t4a-metal-runtime =0.1.2`,
  `t4a-cuda-kernels =0.3.0` and transitive `t4a-metal-sys =0.1.0`. The release chain that was
  the follow-up on #2402 has run: [`t4a-metal-runtime 0.1.2`](https://crates.io/crates/t4a-metal-runtime/0.1.2)
  was published from `t4a-gpu-backend@1a8e409` (the `Send` revision plus the version bump), and
  the temporary exception entries are gone.
- Not checked here: the hosted macOS lane (it runs on PR, this is macOS with a device but the
  workflow's `macos-15` runner is not exercised locally), and the GPU pods, which this diff does
  not touch.
