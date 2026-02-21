# Algebra

Minimal algebra foundation for `TensorPrims<A>`. Provides the `HasAlgebra`
trait for automatic algebra inference and the `Standard` type for standard
arithmetic.

---

## Core Types

```rust
/// Maps a scalar type T to its default algebra A.
/// Enables automatic inference: Tensor<f64> → Standard, Tensor<MaxPlus<f64>> → MaxPlus.
pub trait HasAlgebra {
    type Algebra;
}

/// Standard arithmetic algebra (add = +, mul = *).
pub struct Standard;

impl HasAlgebra for f64 { type Algebra = Standard; }
impl HasAlgebra for f32 { type Algebra = Standard; }
impl HasAlgebra for Complex64 { type Algebra = Standard; }
// etc.

/// Semiring trait for algebra-generic operations.
pub trait Semiring {
    type Scalar: ScalarBase;
    fn zero() -> Self::Scalar;
    fn one() -> Self::Scalar;
    fn add(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;
    fn mul(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;
}
```

## Tropical Extensibility

Tropical types (`MaxPlus`, `MinPlus`, `MaxMul`) are in the separate
`tenferro-tropical` crate, not here. This separation proves that the
algebra extension mechanism works for external crates.

```rust
// tenferro-tropical crate
pub struct MaxPlus;

impl HasAlgebra for MaxPlus<f64> { type Algebra = MaxPlus; }

impl TensorPrims<MaxPlus> for CpuBackend {
    fn has_extension_for<T: ScalarBase>(ext: Extension) -> bool {
        false  // tropical uses core ops decomposition
    }
    ...
}
```

## User-Defined Algebras

The `TensorPrims<A>` parameterization enables external crates to implement
their own algebras (orphan rule compatible):

```rust
// User crate
struct MyScalar(f64);
struct MyAlgebra;

impl ScalarBase for MyScalar { ... }
impl HasAlgebra for MyScalar { type Algebra = MyAlgebra; }

impl TensorPrims<MyAlgebra> for CpuBackend {
    type Plan<T: ScalarBase> = MyPlan<T>;
    type Context = CpuContext;
    ...
}

// Just works:
let a = Tensor::<MyScalar>::zeros(&[3, 4], ...);
einsum("ij,jk->ik", &[&a, &b])?;  // MyAlgebra auto-inferred
```

## Algebra and Autodiff

AD must remain algebra-aware:

- Standard arithmetic: direct rrule/frule formulas over `+/*`.
- Tropical algebra: formulas may need algebra-specific state (e.g., argmax
  path information for max-plus variants).
- API design keeps this extensible by relying on `HasAlgebra` and
  `TensorPrims<A>` rather than hard-coding only standard arithmetic.

See [autodiff.md](./autodiff.md) for the full AD architecture.
