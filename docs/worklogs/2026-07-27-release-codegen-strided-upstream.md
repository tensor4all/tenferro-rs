# CPU Release Codegen And Strided Upstream Boundary

This worklog records the #1486 fresh timing pass and the post-measurement
decision. No additional `tenferro-cpu` crate split should land from this issue
until the generic CPU-kernel ownership question is settled upstream in
`strided-kernel`.

## Measurement Setup

All timings used Rust/Cargo 1.97.1, `CARGO_BUILD_JOBS=64`,
`CARGO_INCREMENTAL=0`, no `RUSTC_WRAPPER`, and a fresh target directory:

```text
/tmp/tenferro-1486-baseline-target.TjMbCR
```

The measurements were run from `origin/main` before any new #1486 code or
crate-boundary changes.

## Results

Clean workspace release build:

```text
command: cargo build --workspace --release --timings
wall: 3m40.70s
user: 1507.99s
max RSS: 4.68 GiB
timing report: /tmp/tenferro-1486-baseline-target.TjMbCR/cargo-timings/cargo-timing-20260727T005704758Z-4d6b3f7999bda312.html

Top units:
tenferro-internal-cpu-kernels 212.2s total, 8.4s frontend, 203.8s codegen
tenferro-cpu                  141.6s total, 8.2s frontend, 133.4s codegen
faer                           30.0s total, 29.8s frontend, 0.2s codegen
tenferro-linalg                11.9s total, 4.0s frontend, 7.8s codegen
tenferro-fft                   11.2s total, 0.8s frontend, 10.4s codegen
tenferro-runtime                6.8s total, 3.7s frontend, 3.2s codegen
```

Provider-only rebuild:

```text
command: touch crates/tenferro-cpu/src/provider.rs
command: cargo build -p tenferro-cpu --release --timings
wall: 2m16.94s
user: 524.98s
max RSS: 3.65 GiB
timing report: /tmp/tenferro-1486-baseline-target.TjMbCR/cargo-timings/cargo-timing-20260727T010131006Z-4d6b3f7999bda312.html

Top units:
tenferro-cpu                  133.1s total, 7.8s frontend, 125.3s codegen
tenferro-runtime                6.4s total, 3.3s frontend, 3.1s codegen
tenferro-internal-ops           0.6s total, 0.3s frontend, 0.3s codegen

Observation:
tenferro-internal-cpu-kernels was reused.
```

Kernel-touch rebuild:

```text
command: touch crates/tenferro-internal-cpu-kernels/src/elementwise.rs
command: cargo build -p tenferro-cpu --release --timings
wall: 3m31.19s
user: 1203.19s
max RSS: 4.62 GiB
timing report: /tmp/tenferro-1486-baseline-target.TjMbCR/cargo-timings/cargo-timing-20260727T010406469Z-4d6b3f7999bda312.html

Top units:
tenferro-internal-cpu-kernels 211.0s total, 7.7s frontend, 203.2s codegen
tenferro-cpu                  141.1s total, 8.1s frontend, 133.0s codegen

Observation:
kernel edits rebuild both the internal kernel unit and the public CPU crate.
```

Focused release test no-run:

```text
command: cargo test -p tenferro-cpu --release --lib --no-run --timings
wall: 5m46.28s
user: 1288.18s
max RSS: 4.64 GiB
timing report: /tmp/tenferro-1486-baseline-target.TjMbCR/cargo-timings/cargo-timing-20260727T010815397Z-4d6b3f7999bda312.html

Top units:
tenferro-internal-cpu-kernels        204.8s total, 8.1s frontend, 196.7s codegen
tenferro-cpu tenferro_cpu lib test   138.4s total
faer                                  29.9s total, 29.7s frontend, 0.2s codegen
criterion                              3.1s total, 1.6s frontend, 1.6s codegen
strided-kernel                         1.2s total, 0.9s frontend, 0.3s codegen
```

These fresh numbers reproduce the #1472 closeout observation: the first
internal kernel split lets provider/runtime-only edits reuse the internal CPU
kernel artifact, but release codegen remains dominated by
`tenferro-internal-cpu-kernels` and by the downstream `tenferro-cpu` unit.

## Ownership Boundary

The current `tenferro-internal-cpu-kernels` crate mixes two distinct layers:

- backend-neutral strided scalar execution built on `strided-kernel`
  (`map_into`, `zip_map*_into`, `fused_elementwise_into`,
  `broadcast_mul_into`, `batched_outer_product_into`);
- tenferro-specific adaptation: `DType` dispatch, promotion/error policy,
  divide-by-zero checks, unsupported complex ordered-op policy, `TensorRead`
  and backend-buffer validation, output allocation, and buffer-pool ownership.

Splitting that file again inside tenferro would isolate some artifacts, but it
would also make a second tenferro-owned layer for generic shape/stride scalar
kernels. That is the wrong default if those kernels can belong to
`strided-kernel`, which is already the upstream owner for generic strided CPU
execution.

The upstream decision is now tracked in
<https://github.com/tensor4all/strided-rs/issues/148>. That issue is a sibling
to `strided-rs` #139, whose stage-1 `CopyPlan` already merged as `strided-rs`
#142. The next design question is whether prepared/raw map and runtime-DAG
fused elementwise kernels should become `strided-kernel` APIs before tenferro
adds more internal CPU kernel crates.

## Decision

Do not implement a second #1486 crate split in tenferro now.

Treat #1486 as resolved by measurement plus an upstream-first boundary
decision:

- keep public `tenferro-cpu`;
- keep tenferro-owned cache and buffer-pool lifetime management in tenferro;
- keep tenferro dtype/error/tensor adapter semantics in tenferro;
- move only backend-neutral prepared/raw strided scalar-kernel machinery
  upstream if the `strided-kernel` maintainers accept the API direction;
- require fresh compile-time and runtime benchmark evidence before any later
  tenferro crate split or upstream specialization lands.

This avoids adding a tenferro-specific internal split that would need to be
undone or mirrored after `strided-kernel` grows the right prepared execution
surface.
