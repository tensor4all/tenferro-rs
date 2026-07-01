# Tropical Extension

Use the tropical extension tutorial when you want to see a non-standard
numeric algebra implemented outside the core tenferro crates. The nested
`ext/tropical` crate is a standalone workspace that defines max-plus and
min-plus operations, exposes traced helpers, registers an extension runtime,
and optionally registers AD rules.

The main path is a fused binary tropical einsum. For max-plus matrix
multiplication, the ordinary dense product

```text
C[i, k] = sum_j A[i, j] * B[j, k]
```

is replaced by

```text
C[i, k] = max_j A[i, j] + B[j, k]
```

## Crate Layout

The tutorial crate lives outside the root workspace:

```text
ext/tropical/
  Cargo.toml
  src/newtype.rs
  src/einsum.rs
  src/traced.rs
  src/extension.rs
  tests/
```

This keeps the example close to a real downstream extension crate. The public
API is crate-owned, while the low-level graph node is an `ExtensionOp`.

## Eager And Traced Entry Points

Eager execution can call the tropical einsum implementation directly:

```rust
use tenferro_ext_tropical::{einsum::tropical_einsum_with_argmax, TropicalKind};
use tenferro_runtime::Tensor;

let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 4.0, 0.0])?;
let b = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, -1.0, 5.0])?;
let out = tropical_einsum_with_argmax(TropicalKind::MaxPlus, &[&a, &b], "ij,jk->ik")?;
```

Traced execution emits an extension op, so the executor must register the
tropical runtime:

```rust
use tenferro_cpu::CpuBackend;
use tenferro_ext_tropical::traced::tropical_dot_general_fused;
use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};

let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 4.0, 0.0])?;
let b = TracedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, -1.0, 5.0])?;
let out = tropical_dot_general_fused(&a, &b)?;

let mut compiler = GraphCompiler::new();
let program = compiler.compile(&out)?;
let mut executor = GraphExecutor::new(CpuBackend::new());
executor.register_extension(tenferro_ext_tropical::register_runtime)?;
let value = executor.run(&program)?;
```

The op payload holds the tropical kind and parsed einsum subscripts. Runtime
registration maps that stable family ID to the concrete execution hook.

## AD Rules

Tropical AD is only well-defined on paths with a unique winning contraction
index. The tutorial registers explicit extension AD rules for that case:

```rust
use tenferro_ad::AdContext;
use tenferro_ext_tropical::tropical_ad_rules;

let ad = AdContext::builder()
    .with_extension_rules(tropical_ad_rules()?)
    .build()?;
let loss = out.reduce_sum(&[0, 1])?;
let grad_a = ad.grad(&loss, &a)?;
```

The AD rule emits extension JVP/VJP ops, so the same runtime registration is
used when the compiled gradient graph executes.

## Run It

From the repository root:

```bash
cargo test --manifest-path ext/tropical/Cargo.toml --release --features autodiff
```

CI runs that command because this tutorial is short enough to execute on every
pull request.
