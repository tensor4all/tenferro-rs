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
companion library. Threaded providers make FFTW's planner thread safe and
restore FFTW's process-wide planner thread count after each plan. Plans use FFTW's unaligned mode so cached plans can execute on Rust
lane buffers of different alignments, which can limit FFTW's use of SIMD.
FFTW threads and tenferro lane parallelism are used separately. Use
`CpuEngine::serial()` for FFTW-owned threading on Linux, where FFTW pthread
workers can inherit the affinity of a pinned tenferro worker.

FFTW is GPL-2.0-or-later and commercial licenses are available. The feature is
off by default and loads an independently installed runtime. Runtime loading
does not by itself exempt a combined work from the GPL. Evaluate FFTW's license
terms before distributing software that enables the FFTW provider. See [NOTICE](NOTICE).
