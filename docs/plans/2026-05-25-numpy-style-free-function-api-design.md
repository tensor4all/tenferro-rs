# NumPy-Style Free-Function Tensor API Design

## Context

Issue #892 tracks the public API cleanup needed to make tenferro easier to use
and easier for agents to target. The current backend primitive surface is broad,
but the public API is uneven:

- `tenferro::traced_tensor` already exposes a small free-function namespace.
- `tenferro::eager_tensor`, `tenferro::tensor`, and
  `tenferro::typed_tensor` mostly re-export types.
- NumPy-style broadcasting currently lives in traced helpers, while eager and
  concrete tensor APIs often route directly to primitive backend operations.
- Standard operation crates expose some family-specific namespaces, but they do
  not yet follow one consistent API policy.

The goal is to make module free functions the canonical NumPy-style public API
for the four core tensor families, while keeping backend primitives simple and
keeping standard extension crates outside the `tenferro` facade.

## Scope

This design covers a helper-first implementation of issue #892 through the
initial binary and ternary public API set, plus a final extension-crate cleanup
pass.

The core operation set for the first implementation pass is:

- Arithmetic: `add`, `sub`, `mul`, `div`, `pow`
- Unary elementwise helpers where the primitive exists: `neg`, `abs`, `sign`,
  `conj`, `exp`, `log`, `sin`, `cos`, `tanh`, `sqrt`, `rsqrt`, `expm1`,
  `log1p`
- Comparison and selection: `maximum`, `minimum`, `compare`, `where_select`,
  `clamp`
- Core contraction: `matmul`, `dot_general`

Shape and reduction helpers are included in the shared normalization layer so
later phases can add `broadcast_to`, `squeeze`, `expand_dims`, `flatten`, and
`sum`/`prod`/`max`/`min` with negative axes and `keepdims` without inventing a
second policy.

Out of scope for the core facade:

- No `tenferro::linalg`, `tenferro::einsum`, or `tenferro::fft` modules.
- No facade dependency from `tenferro` to standard operation crates.
- No broad method mirror for the free-function API.
- No GPU-specific broadcasting semantics in backend primitive methods.

## Architecture

Use this layering:

```text
public module free function
  -> shared pure normalization helpers
  -> tensor-family-specific lowering
  -> primitive tensor op
  -> CPU / CUDA / future backends
```

Shared helpers live in `tenferro-internal-ops` because they are pure semantic
helpers and do not require concrete tensor execution. They should be small,
well-tested modules:

- `axis`: normalize one axis, optional axes, and axis lists with duplicate
  validation where needed.
- `broadcast`: compute NumPy-compatible output shapes and source dims for
  `BroadcastInDim`.
- `reduction`: compute reduced output shapes with and without `keepdims`.

Concrete execution helpers can live near the owning tensor layer when they need
backend execution. For example, concrete tensor public functions can lower
through `broadcast_in_dim` followed by `add` or `select` on the provided
backend. Traced functions reuse the same pure helpers but emit graph ops.

Backend primitive methods remain primitive-oriented. They can accept already
broadcasted inputs, but they do not become responsible for full NumPy sugar.
Future fusion passes may optimize patterns such as `broadcast_in_dim + add`,
but that is an optimization only.

## Core API Shape

The canonical API is module free functions with matching names:

```rust
tenferro::traced_tensor::add(&x, &y)
tenferro::eager_tensor::add(&x, &y)?
tenferro::tensor::add(&x, &y, &mut backend)?
tenferro::typed_tensor::add(&x, &y, &mut backend)?
```

Return shape and error behavior are tensor-family specific:

- `TracedTensor`: graph-building functions return `TracedTensor` and may panic
  on invalid public sugar, matching the existing traced style.
- `EagerTensor`: functions return `tenferro::Result<_>` and use the tensor's
  stored runtime/backend.
- `Tensor`: functions return `tenferro_tensor::Result<_>` with an explicit
  backend argument.
- `TypedTensor<T>`: functions return `tenferro_tensor::Result<_>` with an
  explicit backend argument and bounds following existing typed/backend
  conventions.

Methods remain for receiver-centric operations: shape inspection, dtype/data
access, reshape, transpose, `broadcast_in_dim`, AD lifecycle, and runtime
access. Existing methods and operator overloads can stay where they are already
useful, but they are not the canonical surface for broad NumPy sugar.

## Broadcasting And DType Semantics

Public binary and ternary sugar uses NumPy-style broadcasting. For example:

```text
[3, 1] + [1, 4] -> [3, 4]
[] + [3, 4]     -> [3, 4]
[5] + [3, 5]    -> [3, 5]
```

Lowering is explicit:

```text
add(lhs, rhs)
  -> broadcast_shape(lhs.shape, rhs.shape)
  -> broadcast_dims_for_input(lhs.shape, output_shape)
  -> broadcast_dims_for_input(rhs.shape, output_shape)
  -> broadcast_in_dim on each input as needed
  -> primitive add
```

For `where_select(cond, x, y)`, all three inputs broadcast to a common output
shape before lowering to primitive `Select`.

DType promotion should use one public policy across traced, eager, tensor, and
typed tensor APIs. The existing promotion functions in `tenferro::shape_infer`
are the current policy source for graph lowering. This work should either move
or wrap that policy so public sugar does not duplicate it. Division-like
operations keep integer division promotion to floating output rather than
truncating.

The public target for `compare` is bool output. If changing that in the same
implementation pass is too disruptive because existing graph inference treats
`Compare` as numeric, keep the current behavior and add an explicit follow-up
issue before claiming full comparison acceptance.

## Extension Crate Cleanup

Standard operation families remain first-class crates:

- `tenferro-einsum`
- `tenferro-linalg`
- `tenferro-fft`

The cleanup pass should align their public namespaces with the core policy
without moving them into `tenferro`.

Expected direction:

- `tenferro_einsum::traced_tensor::einsum` and
  `tenferro_einsum::eager_tensor::einsum` should be the canonical tensor-family
  namespaces. Existing root-level functions may stay only if they are already
  documented as convenience entry points.
- `tenferro_linalg::traced_tensor::*` and
  `tenferro_linalg::eager_tensor::*` should be canonical for linalg extension
  APIs. The core facade still exposes no linalg module.
- `tenferro_fft` should either document its root functions as the canonical
  traced API for now or add a `traced_tensor` namespace if that makes the
  extension family consistent without duplicating implementation logic.

This pass should also update README and rustdoc examples so users see the same
import pattern everywhere: tensor/runtime types from `tenferro`, operation
families from their own crates.

## Step Plan

1. Add design/API docs and shared helper tests.
2. Implement shared pure helpers in `tenferro-internal-ops`.
3. Move traced broadcasting to the shared helpers without changing behavior.
4. Add core free functions for traced and eager tensor APIs, with tests for
   matching broadcast behavior.
5. Add concrete `tensor` and `typed_tensor` free functions with explicit
   backend parameters.
6. Add or align extension crate namespaces, keeping standard families out of
   the `tenferro` facade.
7. Update docs and examples only after the public surface exists.

Each step should follow TDD: write the narrow failing test first, verify the
expected failure, then implement the smallest change that passes it.

## Acceptance Criteria

- The four core tensor families expose matching canonical free-function names
  for the initial binary and ternary operation set.
- Eager and traced broadcasting match on representative concrete-shape tests.
- Concrete `Tensor` and `TypedTensor<T>` APIs use explicit backend parameters.
- Shared helper tests cover scalar broadcasting, rank padding, incompatible
  shapes, negative axis normalization, duplicate axes, and `keepdims` shape
  computation.
- `where_select` exists and lowers to primitive `Select` with broadcasted
  inputs.
- Backend primitive implementations are not required to implement NumPy
  broadcasting directly.
- Standard extension crates follow the same namespace policy without adding
  facade modules or dependencies to `tenferro`.
