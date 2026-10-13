# A full reduction uses one warp (#2134): split the plane reduction over the axis

Status: design accepted, implementation pending. This is the record #2112 item 9
asks for: the root fix and its acceptance, written before the code.

## The defect

`launch_with_plane_settings` (`crates/tenferro-gpu/src/kernels/reduce/launch.rs`)
sets

```rust
cube_count: CubeCount::Static(problem.reduce_count.max(1), 1, 1),
cube_dim:   CubeDim::new_1d(plane_width),
```

one cube per **output element**, one warp per cube, each warp walking the whole
reduce axis with a stride of `plane_width`. For a full reduction
`reduce_count == 1`, so a single warp reduces the entire axis: the cost is linear
in the axis length at roughly 3.5-3.8 ns/element, far below memory bandwidth.

Measured by the reporter on an A100 (nsys, release, warm, median of 31), full
`reduce_sum` over a rank-1 tensor:

| len | f32 | f64 |
| --- | --- | --- |
| 1e3 | ~5 us | ~4 us |
| 1e6 | ~1.95 ms | ~1.8 ms |
| 1e7 | ~35.4 ms | ~38 ms |

The `sub`/`abs` elementwise kernels over the same 1e7 elements take about 0.1 ms
(f64), so the reduction is 200-350x slower than a bandwidth-bound pass.

**Blast radius** is wider than the full reduction: the geometry is one cube per
output element, so any reduction with a long axis and few outputs is affected.
`[1e7, 8]` reduced over axis 0 is 8 cubes x 32 threads over 1e8 elements.

## Root cause

Two things are conflated. The plane kernel derives *which output element* it
serves from `CUBE_POS_X`, and it assumes that one cube can walk the whole axis.
Splitting the axis needs (a) each cube to know a *slice* of the axis, and (b) a
place to put the per-slice partial results, because a warp can only combine what
it has already read.

## Design: two passes, and the combine pass reuses the existing kernels

### Pass 1 - chunked partial reduction (new kernels)

Per op, a kernel with the same body scaffolding as the existing plane kernel but:

- signature `(input, partials, axis, output_len, splits, chunk_len)`;
- `cube_count = reduce_count * splits`, `cube_dim = plane_width`, and
  `let output_index = CUBE_POS_X as usize / splits; let split = CUBE_POS_X as usize % splits;`
- the decomposition walks the **input's** dims and *skips the reduce axis*, which
  is exactly what the existing decomposition does in effect: in the keepdims
  output that axis has length 1, so its coordinate is always zero and it
  contributes nothing to the base offset. `partials_offset` accumulates
  `split * partials.stride(axis)` in that step.
- the reduction walks `reduce_index in [split * chunk_len, min((split+1)*chunk_len, axis_len))`
  with the same `plane_width` stride, and writes
  `partials[partials_offset]` at `UNIT_POS == 0`.

Pass 1 therefore needs no output-tensor binding and no keepdims metadata: the
partials' own shape and strides carry the layout.

### Pass 2 - combine (the existing kernels, unchanged)

The partials are a freshly allocated tensor of shape **keepdims shape with the
reduce axis set to `splits`** (contiguous, the split dim innermost). Reducing
that axis is an ordinary single-axis reduction whose input is the partials and
whose output is the real one, so the **existing** `reduce_*_plane` kernel does it
with `axis = rank`:

- the kernel's decomposition loop walks `output.rank()` dims, which is the
  keepdims rank, and reads `input.shape(dim)` / `input.stride(dim)` from the
  partials - aligned for every dim below `rank`;
- `axis = rank` then has `input.shape(rank) == splits` and
  `input.stride(rank) == 1`;
- the strategy for this pass is the *existing* unit/plane selection on
  `reduce_len = splits`, so a small `splits` uses the unit kernel and no plane
  shuffle.

No combine kernel has to be written, and no existing kernel changes.

### Choosing `splits`

The split must not introduce a per-thread identity element. The existing plane
kernel has none - it starts from each thread's own first element and relies on
the plane strategy only being chosen when `reduce_len > plane_width`, so every
thread of a cube is guaranteed at least one element. A split that leaves some
thread with an empty slice would need `0`/`1`/`-inf`/`+inf` per op, and would
have to get the NaN policy right for the max/min helpers as well.

Two rules keep the existing shape instead, and they are what the code uses:

```text
MIN_CHUNK = plane_width * 8          // 256 elements of real work per cube
splits    = (reduce_len / MIN_CHUNK).clamp(1, MAX_SPLITS)
chunk_len = reduce_len / splits      // integer division, floor
```

- `splits > 1` only when the axis is worth splitting (`reduce_len >= 2 * MIN_CHUNK`),
  so short axes keep the unsplit kernel and nothing changes for them;
- `splits <= reduce_len / MIN_CHUNK` makes `chunk_len = floor(reduce_len / splits)`
  at least `MIN_CHUNK`, and the *last* chunk is `reduce_len - (splits - 1) * chunk_len`,
  which for a floor split is at least `chunk_len`. Every chunk therefore holds at
  least `MIN_CHUNK >= plane_width` elements, and each thread reads its own first
  element exactly as the existing kernel does.
- `MAX_SPLITS` bounds the grid and the partials; the acceptance measurement is
  what tunes it, not a number chosen in advance.

Example: `reduce_len = 1e7`, `plane_width = 32`, `MAX_SPLITS = 4096` gives
`splits = 4096`, `chunk_len = 2441` (about 76 plane passes per thread), a 4096-element
partials buffer, and a combine pass that one warp finishes in about 128
iterations.

### Where the partials buffer comes from

The launch functions receive `TensorBinding`s and do not allocate: `launch_*` is
called from `BackendSession::launch_reduce_axis_typed`, which owns `alloc_output`
and the binding construction. Two options were weighed, and the first was
rejected after reading the binding path:

- **allocate inside the reduce layer** and build the `TensorBinding` by hand from
  `client.empty(bytes)` plus `Handle::binding()`. `TensorBinding`'s fields are
  public, so this compiles - but it bypasses `prepare_device_read` and
  `prepare_device_write`, which are where tenferro keeps its device-read
  ownership, memoized-address and claim discipline, and where a write binding
  invalidates the memoized device address (`#1868`, `#1875`). A scratch buffer
  built behind that layer's back is exactly the kind of bypass the discipline
  exists to prevent.
- **allocate in the backend layer** and thread the binding through the launch
  closure. That reuses `alloc_output` (so the scratch is pool-backed like every
  other output) and stays inside the prepared-access contract.

Chosen: the backend layer. The cost is an API change - a reduce plan that carries
both the resolution and the split, and one more binding parameter on the twelve
`launch_*` functions - which is why the plan is computed once, in the reduce
module, and the backend layer only allocates what it asks for:

```rust
// kernels/reduce/routines.rs
pub(crate) struct ReduceLaunchPlan {
    pub kind: ResolvedReduceStrategy,
    pub cube_count: CubeCount,
    pub cube_dim: CubeDim,
    pub axis: usize,
    pub output_len: usize,
    pub split: Option<ReduceSplit>,   // None = run the unsplit kernel
}

/// Validate, resolve the strategy and decide the split, without launching.
pub(crate) fn plan(client: &Client, input: &TensorBinding, output: &TensorBinding,
                   axis: usize, strategy: ReduceStrategy) -> Result<ReduceLaunchPlan>

/// The partials tensor this plan needs, as a keepdims shape whose reduce axis is
/// `splits`. The caller allocates it and passes the binding to `launch_*`.
pub(crate) fn partials_shape(plan: &ReduceLaunchPlan, output_shape: &[usize]) -> Option<Vec<usize>>
```

`launch_reduce_axis_typed` then allocates `partials_shape(...)` through
`alloc_output` when it is `Some`, and each `launch_*` runs pass 1 over that
binding and pass 2 over the same binding as its input.

### Determinism and numerics

The split order is fixed by `split = CUBE_POS_X % splits`, and pass 2 reduces the
partials in a fixed order, so the result is deterministic - unlike an atomic
accumulation into the output, which is why atomics are **rejected** here. The
float order does change relative to the unsplit kernel, so the existing
tolerance-based parity tests still apply and exactness is preserved where the
inputs are exact (`reduce_sum(sub(x, x))` stays exactly zero).

The max/min family keeps its NaN policy: pass 1 and pass 2 both use the existing
`nan_propagating_max`/`nan_propagating_min` helpers, so a NaN anywhere in the
axis still wins. Integer ops keep the `wrapping_add`/`wrapping_mul` helpers.

## Acceptance

- **#2134's arm**: a 1e7-element f64 full `reduce_sum` from ~38 ms toward ~0.1 ms
  (the elementwise pass over the same data), measured on the A100 with the
  repository's own command. The f32 row and the 1e6 rows move with it.
- **No regression**: the existing reduction suite (`crates/tenferro-gpu`, 216
  device tests) passes unchanged, including the complex plane path pinned by
  `test_cubecl_complex_full_reduction_compiles_and_matches_cpu` (#2133) and the
  unit/plane boundary cases.
- **A new device test** asserts a large-axis full reduction matches the CPU, so
  the split path is covered beyond the acceptance measurement.

## Out of scope

- Per-operation *host* cost (#2079, #2112 item 7). This is a device-side kernel
  change; the launch count does grow from one to two for a split reduction, which
  the measurement above will show against the kernel time it saves.
- Sequence-level execution, capture/replay, fusion: #1938 section 5.
