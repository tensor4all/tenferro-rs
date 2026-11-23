# FFTW runtime provider

## Decisions

- Select FFTW per CPU engine and reuse the existing lane execution and plan cache. See the [FFT backend design](../design/fft-backend-execution.md).
- Reuse plans across lane buffers by using FFTW’s unaligned mode. This can limit SIMD selection. Serialize native planning and destruction while keeping execution concurrent across separate buffers.
- Threaded providers require FFTW 3.3.9 or later. Every provider installs its own FFTW planner hooks (`fftw_set_planner_hooks`), after `fftw_make_planner_thread_safe` when the thread API exists so that later calls of it cannot replace them. The hooks serialize all plan creation and destruction in the process. Each tenferro plan sets the planner thread count and restores it inside that lock, using a request that the planning thread records just before calling FFTW. A plan inside FFTW's planner reads and rewrites the count (`invoke_solver` in `kernel/planner.c`), so setting it outside the lock races with another user's planning.
- Single-threaded providers load the thread API when the installation has one, and plan with one thread whatever count another FFTW user set.
- FFTW worker threads keep the CPU mask of the thread that first starts them. Threaded planning (including `Measure`'s timed candidates) and threaded execution on a team engine run through `Exec::external`. A team's one-slot token runs single-threaded plans.
- Threaded providers parallelize across lanes, with single-threaded plans, when the lanes can occupy the engine's thread budget, and use FFTW threads in one job otherwise. Release medians on an Apple M3 Pro, 4-thread engine and 4-thread provider, complex f64: 262,144 length-16 lanes 7.0 → 3.0 ms; four 2^20 lanes 22.3 → 16.6 ms; one 2^22 lane 26 ms either way.

## Verification conclusions and constraints

- Runtime coverage checks both precisions, odd and even lengths, concrete and traced execution, eager in-place execution, and reuse of threaded plans.
- Local runtime testing uses macOS FFTW. The Linux CI job exercises the separate threading libraries and the worker CPU masks (`tests/fftw_placement.rs`, Linux only).
- `fftw::tests` reads thread counts from `fftw_sprint_plan`: a 2^20 estimate plan has threaded stages whenever the planner count allows them. A foreign thread planning concurrently through the FFTW API keeps its own count, and single-threaded tenferro plans have no threaded stage under an ambient count of three.
