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
pub struct MaxPlus<T>(PhantomData<T>);

impl HasAlgebra for MaxPlus<f64> { type Algebra = MaxPlus<f64>; }

impl TensorPrims<MaxPlus<f64>> for CpuBackend {
    fn has_extension_for<T: ScalarBase>(ext: Extension) -> bool {
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

See [autodiff.md](./autodiff.md) for the full AD architecture.
