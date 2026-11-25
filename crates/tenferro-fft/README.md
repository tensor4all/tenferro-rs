# tenferro-fft

FFT operations for tenferro tensors. See the [FFT guide](https://github.com/tensor4all/tenferro-rs/blob/main/docs/guides/tenferro-fft.md) for the concrete and traced APIs.

## Optional FFTW CPU provider

Enable the `fftw` Cargo feature and install an FFTW 3 runtime for both single and double precision. Select the provider on the CPU engine that will execute the transforms:

```rust
use std::num::NonZeroUsize;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{Fftw, FftwPlanning};

let mut engine = CpuEngine::serial();
let fftw = Fftw::load(FftwPlanning::Estimate, NonZeroUsize::MIN)
    .expect("install the FFTW single- and double-precision runtimes");
fftw.install(&mut engine);
```

A new CPU engine uses RustFFT until a provider is explicitly installed. Use
`Fftw::load_from` to choose a library directory. Choose `Measure` to let FFTW
time candidate plans. A thread count above one requires FFTW 3.3.9 or later
with threading support, either in the base library or in the FFTW threads
companion library.

Loading a provider calls `fftw_make_planner_thread_safe` once per FFTW
installation, so FFTW's own lock serializes every plan creation and
destruction in the process, including those of other libraries that use the
same FFTW. tenferro installs no planner hooks of its own. It holds a private
lock around `fftw_init_threads`, `fftw_destroy_plan`, and each plan creation
together with the `fftw_plan_with_nthreads` call before it: a tenferro plan
sets FFTW's process-wide planner thread count to the count it needs, one for
plans that run inside parallel lanes, and leaves it there: other code that
plans with FFTW afterwards should set its own count first. A single-threaded
provider can set the count only when the installation exports FFTW's thread
API.

Two boundaries follow from this:

- `fftw_make_planner_thread_safe` requires FFTW 3.3.6 or later and lives with
  FFTW's thread API. Without it (an older FFTW, or an installation without a
  threading library) loading still succeeds and tenferro's lock alone
  serializes tenferro's planning, but other libraries' planning in the process
  is not serialized with tenferro's and must not run concurrently with it.
- FFTW's lock covers each call, not the pair of setting the thread count and
  planning, and `fftw_plan_with_nthreads` takes no lock. Other code that calls
  it while a tenferro plan has its count set changes the count tenferro plans
  with, and a plan other code creates in that window uses tenferro's count.
  Planning by other code with an explicit thread count, concurrently with
  tenferro in the same process, is unsupported. Other code that plans with
  FFTW's default count may run concurrently.

Load the provider before other threads start planning with FFTW, since FFTW
does not install its lock atomically. FFTW's lock does not cover
`fftw_init_threads` or the creation of FFTW's global planner, which tenferro's
first plan of an installation can trigger: other code must call
`fftw_init_threads`, or create a plan, before it plans concurrently with
tenferro.

Plans use FFTW's unaligned mode so cached plans can execute on Rust lane
buffers of different alignments, which can limit FFTW's use of SIMD.
FFTW threads and tenferro lane parallelism never nest. A transform whose lanes
can occupy every engine thread runs them in parallel with single-threaded FFTW
plans; otherwise it runs as one job with FFTW's threads. FFTW starts its
worker threads from the thread that plans or executes, and they keep that
thread's CPU mask. On a team engine, threaded planning and execution therefore
run through `Exec::external`, off the one-CPU placement of a team leader or
member. On a serial engine they run on the calling thread, whose mask
tenferro does not change.

FFTW is GPL-2.0-or-later and commercial licenses are available. The feature is
off by default and loads an independently installed runtime. Runtime loading
does not by itself exempt a combined work from the GPL. Evaluate FFTW's license
terms before distributing software that enables the FFTW provider. See [NOTICE](NOTICE).
