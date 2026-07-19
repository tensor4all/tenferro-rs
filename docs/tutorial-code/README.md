# tenferro Tutorial Code

Runnable source code for the tenferro tutorial pages.

The online tutorials quote these binaries as their source of truth. Run them
with:

```bash
cargo test -p tenferro-tutorial-code --release
```

In CI, this package is executed by the existing workspace test command. Do not
add a separate tutorial workflow that recompiles tenferro after unit tests.

The Apple shared-allocation tutorials are feature-gated so ordinary tutorial
tests do not compile the WebGPU stack. Compile and run them on macOS with:

```bash
cargo test -p tenferro-tutorial-code --no-default-features \
  --features cpu-faer,apple-shared --test tutorial_binaries
```

`apple_shared_fft` selects RustFFT, CubeK Metal, and RustFFT again over one
managed input, then checks allocation/domain and transfer-counter invariants.
It also demonstrates C64 CPU support and the typed Metal capability error.
`apple_shared_cholesky` selects the paired CPU backend for the initial mapped
rank-2 linalg operation and verifies an SPD reconstruction residual. Both
binaries are compile-gated to macOS and surface Metal initialization failures
instead of treating them as a successful run.
