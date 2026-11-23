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

Loading a provider installs planner hooks in FFTW that serialize every plan
creation and destruction in the process, including those of other libraries
that use the same FFTW. Inside that lock each tenferro plan sets FFTW's
process-wide planner thread count to the count it needs, one for plans that
run inside parallel lanes, and restores the previous count afterward, so
other FFTW users never plan with tenferro's count. A single-threaded provider
can override another user's count only when the installation exports FFTW's
thread API. `fftw_plan_with_nthreads` itself takes no lock: other code must
not call it while another thread plans. Load the provider before other threads
start planning with FFTW, and do not install other planner hooks afterward.

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
