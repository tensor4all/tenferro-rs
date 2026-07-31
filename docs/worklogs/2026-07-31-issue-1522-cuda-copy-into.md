# 2026-07-31 CUDA `copy_read_into` destination reuse (#1522)

## Scope

The public caller-owned-destination CUDA `copy_read_into` path now shares the
existing cuTENSOR permutation executor and `CudaBackend`-owned permutation plan
cache when the source is compact column-major and the destination is a valid
nonnegative-stride view of a supported `F32`, `F64`, `C32`, or `C64` allocation.
No new cache or pointer-global state was added.

Integer and Bool copies continue to use the existing CubeCL path. Negative
destination strides also continue to use that path because cuTENSOR 2.x does
not accept negative-stride descriptors. This is a layout capability boundary,
not a missing-library fallback. For supported cuTENSOR layouts, loader and plan
errors remain typed errors and are never silently routed to CubeCL.

The cuTENSOR path repeats the existing public contract checks: equal shape,
device residency, distinct source/destination allocations, compact full-source
coverage, and valid destination reachable range. Destination view construction
continues to enforce internal non-overlap.

## Tests

- f64 2D destination reuse matches the expected column-major permutation.
- C32 3D destination reuse matches the expected column-major permutation.
- Mutating the source after `copy_read_into` leaves the caller-owned
  destination unchanged.
- Existing aliasing, noncompact-source, dtype, and typed cuTENSOR-loader tests
  remain in place.

Focused A100 command:

```bash
CUBECL_DEBUG_LOG=0 CUDA_PATH=/usr/local/cuda \
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcutensor/12:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
CUDA_VISIBLE_DEVICES=0 CARGO_BUILD_JOBS=64 \
cargo test -p tenferro-gpu --features cuda \
  cubecl::tests::structural_tests::cuda_runtime_copy_into_cutensor \
  -- --ignored --nocapture
```

The focused f64/C32 differential and mutation tests passed on an NVIDIA A100
80GB PCIe.

## A100 measurement

Machine: NVIDIA A100 80GB PCIe, CUDA device 0. The focused runner used three
warmups and seven timed calls per row, with `CudaBackend::copy_read_into`
followed immediately by `backend.runtime().synchronize()` in every timed call.
The source and destination were allocated once outside the timed region.

| row | origin/main public path | this worktree public path | raw cuTENSOR control from #1522 | new/raw |
| --- | ---: | ---: | ---: | ---: |
| 2D `[32768,16384]`, transpose | 25.893 ms | 5.593 ms | 5.403 ms | 1.035x |
| 3D `[1024,1024,512]`, `[1,2,0]` | 84.692 ms | 5.577 ms | 5.340 ms | 1.044x |

The origin/main values and this-worktree values were collected with the same
fixture, three warmups, seven timed calls, and host synchronization after each
call. The raw cuTENSOR values are the exact controls recorded by accepted issue
#1522 under the same fixture and destination-reuse protocol. Both rows are
within the 20% target.

The first public-path run was 6.598 ms for 2D because the descriptor advertised
only the scalar-size alignment (`8` for f64), producing a different cuTENSOR
plan from the raw 256-byte-aligned control. The path now resolves the actual
device-region pointer alignment and includes it in the existing cache key. The
follow-up A100 run selected the aligned plan and reduced 2D to 5.593 ms and 3D
to 5.577 ms; both rows are within the 20% raw-control target. Nsight Systems
also showed that the steady-state public host-side lookup/validation is only
about 0.1--0.25 ms, while the cuTENSOR kernel is about 6 ms, so the original
gap was primarily plan/kernel selection rather than an unbounded public API
overhead.

The worktree's final A100 samples were 2D `[5.5621, 5.5738, 5.5894, 5.5928,
5.6038, 5.6145, 5.6221]` ms and 3D `[5.4205, 5.4265, 5.5678, 5.5774,
5.5893, 5.5906, 5.6091]` ms. The medians above are used for the gate.

This focused test does not expose a CPU thread-count dimension: CUDA work is
submitted to the backend stream and synchronized at the host boundary. The
public API benchmark harness currently has no `copy_read_into` 2D/3D rows, so
the required threads-1/threads-4 CPU publication gate is not applicable to this
CUDA-only issue; no such result is being inferred here.

## Public usage

The CUDA quickstart example exercises `TensorStructural::copy_read_into` with
an already allocated destination. CUDA requires the NVIDIA library stack for
cuTENSOR-supported layouts and reports typed provider/load errors when it is
not available.
