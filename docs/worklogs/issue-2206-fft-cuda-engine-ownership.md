# FFT on caller-held CUDA engines (#2206)

## Decisions

- **cuFFT plans belong to the engine, not to the cache handle the caller
  passes.** A cuFFT plan is device-bound, so a caller-owned `FftExecutor`
  cache (which may outlive an engine and be shared by several engines) never
  receives one. A direct call borrows the engine's own extension-cache store
  for the call with `ExecutionEngine::take_extension_caches` /
  `restore_extension_caches`; eager and traced execution receive the same
  engine-owned store from the runtime for the run. One engine owns exactly one
  logical stream, so the store plus the runtime identity in `CufftPlanKey`
  identify the plan's engine and stream together.
- **Plan retirement never waits on the stream.** Every execution already
  synchronizes the bound stream inside the raw session before releasing its
  session-scoped work area (the retained cuFFT lifetime obligation of #967).
  A cached plan therefore has no queued work that references it, and
  `CufftPlanEntry::drop` only activates the retained context and calls
  `cufftDestroy`. The alternative - a synchronize inside `Drop` - was rejected:
  it blocks a non-reporting path that the engine's own drop barrier already
  covers, and repeats per evicted plan.
- **Unproven completion leaks instead of destroying.** A failed execution
  barrier marks that entry unproven (the work area and retention guards are
  forgotten as before), and its `Drop` then leaks the plan handle with its
  library and runtime witnesses rather than destroying a handle whose queued
  work may still be active. The same leak applies when the retained context
  cannot be activated or `cufftDestroy` reports a status.
- The CUDA FFT test targets drive a caller-held `CudaEngine`
  (`Runtime::prepare` / `Runtime::run` for traced graphs, direct calls
  otherwise) and observe cuFFT plans through the engine's own cache store.
  `tenferro-fft` now joins the CUDA nextest archive, so the CUDA partition
  classifies every FFT test.

## Verification conclusions and constraints

- The design intent is recorded in
  [FFT Backend Execution](../design/fft-backend-execution.md#cudacufft-execution).
- The three eager CUDA FFT cases self-skip: `EagerRuntime::ensure_engine`
  registers CPU engine kinds only in this tree
  (`crates/tenferro-ad/src/eager_backend.rs`, `TODO(#2206 gpu)`), so an eager
  CUDA operation cannot run until that port lands. They start running
  unchanged as soon as it does; every other error still fails them.
- The CUDA partition rows were generated from the FFT package inventory under
  the archive feature set. The full four-package archive cannot be listed in
  this tree while `tenferro-linalg`'s CUDA port is unfinished, so the
  end-to-end archive validation belongs to the integration pass.
