# Numerical Rust Rules

Domain-specific elaborations of `common-agent-rules.md`. The examples below
are illustrative — apply the same reasoning to any new pattern.

## Testing

- Use small deterministic inputs for correctness tests.
- Prefer hard-coded data or seeded RNGs over unseeded randomness.
- Feature-gate expensive tests.
- For local trial-and-error loops, prefer `cargo nextest run --release --workspace --no-fail-fast` for unit and integration tests.
- Run doctests separately with `cargo test --doc --release --workspace`; `nextest` does not replace them.

## Generic Test Patterns

Test logic should be written once and instantiated for each scalar type, not
copy-pasted (DRY).

- Write test functions generic over `T: Scalar` and instantiate with a macro:
  ```rust
  fn test_foo_generic<T: Scalar>() { ... }
  macro_rules! scalar_tests {
      ($name:ident, $fn:ident) => { paste::paste! {
          #[test] fn [<$name _f64>]() { $fn::<f64>(); }
          #[test] fn [<$name _c64>]() { $fn::<Complex64>(); }
      }};
  }
  scalar_tests!(test_foo, test_foo_generic);
  ```
- Centralize approximate equality helpers so tolerance policy lives in one place.
- Parameterize over algorithms when the validation logic is the same.

## Performance Habits

- Avoid allocating dense temporary buffers when the operation can work over strided or borrowed views.
- Avoid zero-filling buffers that are immediately overwritten.
- Avoid repeated index multiplication inside hot loops when incremental offsets suffice.
- Avoid allocations inside hot loops.
- Precompute plans and reusable metadata outside execution loops.

## Backend and Scalar Type Genericity

Every time you write a function, ask: "Can I replace this concrete type with a
type parameter and a trait bound?" If yes, do it.

### No TypeId / type_name dispatch in library code

Never use `std::any::TypeId` or `std::any::type_name` to branch on scalar or
backend types in production code. Use trait dispatch instead.

**Bad:**
```rust
if TypeId::of::<T>() == TypeId::of::<f64>() { ... }
if type_name::<B>() == type_name::<CpuBackend>() { ... }  // also unstable
```

**Good — trait with associated constant:**
```rust
trait HasCDtype { const C_TYPE: CDataType; }
impl HasCDtype for f64 { const C_TYPE: CDataType = C_F64; }
fn get_dtype<T: HasCDtype>() -> CDataType { T::C_TYPE }
```

If a closed set of types must be mapped to external C constants and a trait is
truly impractical, isolate the TypeId dispatch in one private function and
document why. No new TypeId checks elsewhere.

### No hardcoded concrete backend types in APIs

**Bad:**
```rust
pub(crate) type ActiveBackend = CpuBackend;
pub fn run(ctx: &mut CpuContext, ...) { ... }
```

**Good:**
```rust
pub fn run<B: TensorBackend>(ctx: &mut B::Context, ...) { ... }
```

### Scalar type conversions: use From/Into or T::from_xxx

Never embed type names in conversion function names (`from_f64`, `to_c64`,
`dense_f64_to_tensor`). Use standard traits or trait methods instead.

**Bad:**
```rust
fn dense_f64_to_tensor(s: &Storage) -> Tensor<f64> { ... }
fn dense_c64_to_tensor(s: &Storage) -> Tensor<Complex64> { ... }
```

**Good:**
```rust
fn storage_to_tensor<T: Scalar>(s: &Storage) -> Tensor<T> { ... }
// T::from_real(x)  — nalgebra ComplexField
// T::from_f64(x)   — num_traits FromPrimitive
// Complex64::from(1.0_f64)  — std From
```

### Deduplicate scalar-type impls with macros

When the same impl block, match arm, or function body repeats for multiple
scalar types with only the type name changed, use a generic helper or a
declarative macro.

**Bad — repeated match arms:**
```rust
match storage {
    Storage::DenseF64(d) => d.iter().map(|x| x.abs()).sum::<f64>(),
    Storage::DenseF32(d) => d.iter().map(|x| x.abs()).sum::<f64>(),
    ...
}
```

**Good:**
```rust
fn sum_abs<T: Scalar>(slice: &[T]) -> f64 { slice.iter().map(|x| x.abs_real()).sum() }
```

For C-API boilerplate that cannot be made generic, use `macro_rules!` with
`paste::paste!` to generate repeated struct and function names.

## Numerical API Design

- Keep examples small enough to read in rustdoc.
- Prefer APIs that make shape and layout expectations explicit.
- Expose helpers for common validation rather than repeating ad hoc checks across crates.
