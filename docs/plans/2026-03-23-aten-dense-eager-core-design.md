# ATen Dense Eager Core Compatibility Design

## Goal

Start a new `ATen`-aligned phase for tenferro by building a reusable dense eager core across `tenferro-device`, `tenferro-tensor`, and `tenferro-prims`, then use that core to remove the remaining host-side metadata cleanup paths from `tenferro-linalg`.

This phase is not "full PyTorch parity." It is specifically "dense eager core compatibility" for the subset of tensor semantics that unblock linalg and decomposition cleanup without adding more ad hoc helpers.

## Why This Phase Exists

The current branch has already proven that substrate-first cleanup works better than op-local fixes:

- logical copy / `resolve_conj`
- GPU `cat` / `stack`
- complex-to-real unary substrate
- metadata family phase 1
- LU metadata tensorization in `tenferro-linalg-prims`

The next blocker is also substrate-shaped. `det`, `slogdet`, and `lu_solve` still want to reconstruct parity/sign metadata on the host because tenferro does not yet have a dense eager bool/int core comparable to `ATen`.

The recommendation is therefore to stop adding linalg-specific helpers and instead build the missing low-level dense eager substrate directly.

## Compatibility Target

This phase targets `ATen`-style dense eager behavior:

- dense tensors only
- eager execution
- no JIT / dispatcher / tracing requirement
- no sparse / quantized / named / nested support
- shared CPU and CUDA semantics

The intent is to make low-level tensor composition expressive enough that `torch.linalg`-style cleanup code can be written tensor-natively in `tenferro-linalg`.

## Scope

### In Scope

- deterministic constructors
  - `empty`
  - `empty_strided`
  - `full`
  - `zeros`
  - `ones`
  - `*_like`
  - `arange`
  - `linspace`
  - `eye`
- RNG constructors
  - `rand`
  - `randn`
  - `randint`
  - `*_like`
- dense eager pointwise and reduction substrate across:
  - bool metadata
  - `i32` metadata
  - `f32`
  - `f64`
  - `Complex32`
  - `Complex64`
- cast / bridge / select substrate:
  - bool metadata -> scalar tensor
  - `i32` metadata -> scalar tensor
  - metadata/scalar `where`
  - promotion policy sufficient for linalg cleanup closure
- shape/view/materialize substrate:
  - `contiguous`
  - `broadcast`
  - `reshape`
  - `cat`
  - `stack`
  - `view_as_real`
  - `view_as_complex`
  - logical conjugation materialization
- tensor-native linalg metadata composition:
  - LU pivots/info as tensors
  - `det`
  - `slogdet`
  - `lu_solve`
- public LU surface alignment toward `PyTorch`

### Explicitly Out Of Scope

- sparse tensors
- quantized tensors
- named tensors
- nested tensors
- JIT / dispatcher / serialization parity
- neural-network-specific operators
- autograd expansion beyond current linalg cleanup needs
- advanced indexing / sorting / top-k / scatter families
- random distributions beyond:
  - uniform
  - normal
  - integer range

## Layering And Dependency Rules

Crate dependency order must remain:

1. `tenferro-device`
2. `tenferro-tensor`
3. `tenferro-prims` and `tenferro-linalg-prims`
4. `tenferro-linalg`

That implies:

- `tenferro-tensor` must not depend on `tenferro-prims`
- `Tensor` object/view/materialize concerns stay in `tenferro-tensor`
- family protocol and dtype-crossing execution stay in `tenferro-prims`
- low-level kernels and runtime state stay in `tenferro-device`
- `tenferro-linalg` consumes substrate; it does not define replacement substrate

## DRY / KISS Guardrails

- No new linalg-only parity/sign helper if the behavior can be expressed with generic metadata ops.
- No new CPU fallback in `tenferro-linalg`.
- No GPU payload fallback to host for normal execution.
- No temporary API that is knowingly misaligned with the target `ATen` direction unless it is explicitly documented as a short-lived bridge.
- Any new substrate should be justified by at least two consumers or by clear `ATen` first-class status.

## Architecture

### `tenferro-device`

Owns raw dense eager runtime substrate:

- device allocation and transfer
- `empty` / `empty_strided`-class raw allocation support
- fill / iota kernels
- metadata kernels:
  - bool
  - `i32`
  - compare
  - logical combine
  - reduction
  - select
- scalar and complex launch substrate reused by family execution
- RNG runtime:
  - CPU generator support through host-side engine integration
  - CUDA generator support through per-device `Philox`
  - state/seed/offset handling

### `tenferro-tensor`

Owns tensor object API and dense eager materialization:

- public constructors
- `*_like` wrappers
- `arange` / `linspace`
- `cat` / `stack`
- `view_as_real` / `view_as_complex`
- shape/view/materialize operations
- transfer and layout normalization

It must stay free of family execution logic.

### `tenferro-prims`

Owns family execution and dtype-bridge semantics:

- metadata family
- scalar family
- analytic family
- complex-real family
- RNG family
- metadata/scalar cast bridge
- select / `where`
- promotion policy used by linalg cleanup

### `tenferro-linalg-prims`

Owns backend contracts that consume tensor metadata directly:

- `pivots` stay tensor-native
- `info` stays tensor-native
- no reintroduction of `Vec<i32>` metadata escape hatches

### `tenferro-linalg`

Consumes the dense eager core and performs cleanup:

- `det`
- `slogdet`
- `lu_solve`
- later:
  - `pinv`
  - `norm`
  - `matrix_exp`

No new low-level helper layer should be created here unless it is a thin composition helper over already-existing substrate.

## Dense Eager Core Surface

### Constructors

The target constructor set for this phase is:

- shape-driven
  - `empty`
  - `empty_strided`
  - `zeros`
  - `ones`
  - `full`
  - `arange`
  - `linspace`
  - `eye`
- tensor-driven
  - `empty_like`
  - `zeros_like`
  - `ones_like`
  - `full_like`
  - `rand_like`
  - `randn_like`
  - `randint_like`

`zeros`, `ones`, and `eye` already exist, but the API family is incomplete and not yet organized as an ATen-like constructor set.

### Metadata Core

The minimum bool/int metadata closure is:

- generate
  - `iota`
- binary compare
  - `eq`
  - `ne`
- binary arithmetic / logical
  - `add`
  - `sub`
  - `mul`
  - `bitand`
- ternary select
  - `where`
- reduction
  - `sum`
  - `all`
  - `any`

### Mixed-Dtype Bridge

The bridge closure needed by linalg is:

- bool metadata -> scalar same-shape tensor
- `i32` metadata -> scalar same-shape tensor
- metadata mask -> scalar/complex `where`
- promotion and cast rules conservative enough for:
  - LU sign/parity reconstruction
  - thresholding
  - `pinv`
  - `matrix_rank`
  - `matrix_exp`

### Representation Helpers

The minimum extra representation substrate is:

- `view_as_real`
- `view_as_complex`
- resolved logical-conjugation materialization

These should match `ATen` directionally, even if the first implementation is narrower.

## RNG Policy

### Canonical Design

RNG is included in this phase and should be designed in an `ATen`-aligned way from the start.

- CPU:
  - stateful generator
  - `MT19937`-class engine
  - normal-sample cache allowed
- CUDA:
  - `Philox 4x32-10`
  - counter-based per-device generator state
  - kernel launches receive generator-derived state, not ad hoc random state objects

### Why Not Delay RNG

RNG is not just a constructor detail:

- generator state
- seeding
- replayability
- offset handling
- backend-specific execution

If this phase is supposed to define the `ATen`-like dense eager core, excluding RNG would leave a major gap in the constructor layer and force later design backtracking.

### RNG Contract Expectations

- common public API across CPU and CUDA
- CPU-first semantics tests
- CUDA parity tests
- deterministic replay under fixed seed
- no demand for full CPU/CUDA bitwise identity unless explicitly designed for it

## CPU And CUDA Policy

This phase is not CUDA-only. Every substrate family must be designed CPU/CUDA together.

The implementation order is:

1. shared contract
2. CPU tests
3. CPU implementation
4. CUDA tests
5. CUDA implementation

This keeps semantics stable and prevents CUDA-first APIs that later become awkward for CPU.

## Testing Strategy

- contract-first tests for each new family / constructor
- CPU-first RED/GREEN
- CUDA parity second
- focused source-level regressions for cleanup paths
- RNG tests split into:
  - replayability
  - shape/dtype semantics
  - statistical sanity

Host transfers are allowed in tests when comparing results, but not as implementation strategy.

## Rollout Plan

### Phase 1: Deterministic Dense Eager Constructors

- `empty`
- `empty_strided`
- `full`
- `*_like`
- `arange`
- `linspace`

### Phase 2: Metadata Phase 2

- metadata arithmetic
- metadata logical combine
- metadata `where`
- metadata reductions

### Phase 3: Cast / Select / Promotion Bridge

- metadata-to-scalar bridge
- metadata/scalar `where`
- conservative promotion rules

### Phase 4: Representation Helpers

- `view_as_real`
- `view_as_complex`

### Phase 5: RNG Core

- generator abstraction
- CPU engine
- CUDA `Philox`
- `rand`
- `randn`
- `randint`
- `*_like`

### Phase 6: Linalg Cleanup

- `det`
- `slogdet`
- `lu_solve`
- LU public surface alignment
- removal of remaining host metadata bridges

## Recommendation

Treat this as a single named program of work:

`ATen dense eager core compatibility`

Do not restart from individual linalg blockers. Build the core once, then clean up linalg on top of it.
