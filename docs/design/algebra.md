# Algebra

Minimal algebra foundation for `TensorPrims<A>`. Provides the `HasAlgebra`
trait as UX sugar for automatic algebra inference and the `Standard<T>` type
for standard arithmetic, where the scalar type `T` is carried by the algebra.

---

## Core Model: `A::Scalar`-Centric Design

The fundamental design is that **the algebra `A` carries the scalar type** via
`Semiring::Scalar`. The algebra parameter in `TensorPrims<A>` is the single
source of truth for both the operation semantics and the scalar type.

`HasAlgebra` is **UX sugar only**: it lets the compiler infer `A` from `T`
automatically (e.g. `Tensor<f64>` infers `Standard<f64>` without the user
spelling it out). It does not change the core model.

```
Scalar type T
    │
    │  (via HasAlgebra<Algebra = Standard<T>>)
    ↓
Algebra A = Standard<T>
    │
    │  (via Semiring::Scalar)
    ↓
A::Scalar = T   ← single source of truth for both semantics and scalar type
```

This means all generic bounds over the algebra are expressed as
`A: Semiring<Scalar = T>` or simply as `A: Semiring`, with `A::Scalar`
used wherever the scalar type is needed.

---

## Core Types

```rust
/// Maps a scalar type T to its default algebra A.
/// This is UX sugar: it enables automatic inference so that
/// Tensor<f64> → Standard<f64>, Tensor<MaxPlus<f64>> → MaxPlus<f64>,
/// without the user spelling out the algebra explicitly.
pub trait HasAlgebra {
    type Algebra;
}

/// Standard arithmetic algebra (add = +, mul = *), parameterized by scalar type T.
/// `A::Scalar` is the canonical way to refer to the scalar type in generic code.
pub struct Standard<T>(PhantomData<T>);

impl HasAlgebra for f64     { type Algebra = Standard<f64>; }
impl HasAlgebra for f32     { type Algebra = Standard<f32>; }
impl HasAlgebra for Complex64 { type Algebra = Standard<Complex64>; }
// etc.

/// Semiring trait for algebra-generic operations.
/// Implemented for Standard<T> for each supported scalar type T.
pub trait Semiring {
    type Scalar: ScalarBase;
    fn zero() -> Self::Scalar;
    fn one() -> Self::Scalar;
    fn add(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;
    fn mul(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;
}

impl<T: Scalar> Semiring for Standard<T> {
    type Scalar = T;
    fn zero() -> T { T::zero() }
    fn one()  -> T { T::one() }
    fn add(a: T, b: T) -> T { a + b }
    fn mul(a: T, b: T) -> T { a * b }
}
```

---

## Tropical Extensibility

Tropical types (`MaxPlus`, `MinPlus`, `MaxMul`) are in the separate
`tenferro-tropical` crate, not here. This separation proves that the
algebra extension mechanism works for external crates.

```rust
// tenferro-tropical crate
pub struct MaxPlus<T>(pub T);
pub struct MaxPlusAlgebra<T>(PhantomData<T>);

impl HasAlgebra for MaxPlus<f64> {
    type Algebra = MaxPlusAlgebra<f64>;
}

impl TensorPrims<MaxPlusAlgebra<f64>> for CpuBackend {
    fn has_extension_for(ext: Extension) -> bool {
        false  // tropical uses core ops decomposition
    }
    ...
}
```

---

## User-Defined Algebras

The `TensorPrims<A>` parameterization enables external crates to implement
their own algebras (orphan rule compatible):

```rust
// User crate
struct MyScalar(f64);
struct MyAlgebra;

impl ScalarBase for MyScalar { ... }
// HasAlgebra is UX sugar — wire MyScalar to MyAlgebra for automatic inference
impl HasAlgebra for MyScalar { type Algebra = MyAlgebra; }

impl TensorPrims<MyAlgebra> for CpuBackend {
    type Plan<T: ScalarBase> = MyPlan<T>;
    type Context = CpuContext;
    ...
}

// Just works:
let a = Tensor::<MyScalar>::zeros(&[3, 4], ...);
einsum("ij,jk->ik", &[&a, &b])?;  // MyAlgebra auto-inferred via HasAlgebra
```

---

## Algebra and Autodiff

AD must remain algebra-aware:

- Standard arithmetic: direct rrule/frule formulas over `+/*`.
- Tropical algebra: formulas may need algebra-specific state (e.g., argmax
  path information for max-plus variants).
- API design keeps this extensible by relying on `HasAlgebra` and
  `TensorPrims<A>` rather than hard-coding only standard arithmetic.

See [autodiff.md](./autodiff.md) for the AD architecture design and
[einsum-dyadtensor.md](./einsum-dyadtensor.md) for einsum/dyadtensor integration details.

---

## Migration Checklist

Steps for migrating to the typed algebra model consistently across the workspace.
Work through these in order; each step unblocks the next.

1. **Each algebra type must carry its scalar via `Semiring::Scalar`.**
   - `Standard<T>` already does this. Any new algebra must define `type Scalar = …`
     in its `Semiring` impl — not a free `T` parameter on the impl itself.
   - Verify: `grep -r "impl Semiring"` — every impl must have an explicit
     `type Scalar` associated type, not a generic `T` that floats free.

2. **`HasAlgebra` impls map scalar types to their typed algebra.**
   - Add `impl HasAlgebra for MyScalar { type Algebra = MyAlgebra; }` in the
     crate that owns `MyScalar`. This is the only place `HasAlgebra` belongs;
     do not add redundant impls elsewhere.
   - `HasAlgebra` is UX sugar only — the algebra `A` is the source of truth.
     Code that needs the algebra explicitly should accept `A: Semiring`, not
     `T: HasAlgebra`.

3. **`TensorPrims<A>` impls must be parameterized by the typed algebra `A`.**
   - Signature: `impl TensorPrims<MyAlgebra> for CpuBackend { … }`
   - The scalar type inside the impl is `<MyAlgebra as Semiring>::Scalar`,
     not a free `T`. Use `A::Scalar` in method bodies instead of bare `T`.

4. **Einsum and linalg functions use `A: Semiring` bounds for algebra-generic code.**
   - Replace any `T: Scalar + HasAlgebra` bound with `A: Semiring` where the
     function needs to be algebra-generic.
   - Keep `T: HasAlgebra<Algebra = A>` on the public-facing thin wrapper so
     callers do not have to spell out `A`.

5. **Tropical algebras already carry their scalar via `Semiring::Scalar`.**
   - `MaxPlus<T>`, `MinPlus<T>`, etc. define `type Scalar = T` in their
     `Semiring` impls. No special case needed.
   - Confirm tropical `HasAlgebra` impls are in `tenferro-tropical`, not in
     `tenferro-algebra`. Tropical types must not leak into the algebra core.

6. **Future: tighten `TensorPrims<A>` method signatures to use `A::Scalar`.**
   - Currently `plan<T: ScalarBase>` and `execute<T: ScalarBase>` use a free
     `T` unconstrained relative to `A`. A future tightening would add
     `where T: ScalarBase, A: Semiring<Scalar = T>` (or remove the free `T`
     and use `A::Scalar` directly).
   - Do not make this change until all existing impls are migrated to step 3,
     to avoid a large simultaneous diff.
