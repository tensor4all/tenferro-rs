# Getting Started

For multi-socket hosts, containers, or schedulers with restricted cpusets, read
[CPU Execution and Thread Pools](../guides/cpu-execution.md) before choosing the
pool you hand to `CpuEngine` and before choosing an external BLAS provider.

tenferro supports tensor computation without autodiff, immediate execution with
optional `backward()` on scalar losses, functional eager `grad`, `vjp`, and
`jvp`, traced graph execution, einsum, linear algebra, and CUDA execution
through the feature-gated CUDA backend.

## If an older example no longer compiles

Start with the [API migration guide](api-migration.md) when a removed module,
free function, constructor, or changed signature appears in an existing
example. It maps the common pre-0.3 spellings to the current trait-based API.

## Mental Model

tenferro has three independent choices. Pick the smallest tensor layer that
matches the program, decide whether work should run immediately or through a
compiled traced program, then choose a CPU or CUDA backend explicitly.

| Choice | Question | Common starting point |
| --- | --- | --- |
| Tensor layer | What kind of value do I pass around? | `TypedTensor<T>` for typed CPU values, `Tensor` for runtime dtype |
| Execution model | When does computation run? | Direct/eager for ordinary code and eager AD; traced for compiled graph reuse |
| Engine | Where does computation run, and who owns it? | `CpuEngine`; `CudaEngine::open` and explicit upload/download for CUDA |

CUDA, eager execution, and traced graphs are not competing APIs. They compose
when a workflow needs them, but the first CPU program below only needs the
runtime crate and the CPU crate.

## Setup

Start with the runtime crate, CPU backend crate, and standard linear algebra
extension from crates.io:

```toml
[dependencies]
tenferro-runtime = "0.9"
tenferro-cpu = "0.9"
tenferro-linalg = "0.9"
```

For development against a local checkout of this repository, use path
dependencies instead:

```toml
[dependencies]
tenferro-runtime = { path = "/path/to/tenferro-rs/crates/tenferro-runtime" }
tenferro-cpu = { path = "/path/to/tenferro-rs/crates/tenferro-cpu" }
tenferro-linalg = { path = "/path/to/tenferro-rs/crates/tenferro-linalg" }
```

If you create a scratch binary crate inside the `tenferro-rs` checkout, add an
empty `[workspace]` table to that scratch crate's `Cargo.toml`. Otherwise Cargo
will treat it as an unlisted member of the parent workspace and report a
workspace-membership error.

For a scratch crate directly under the checkout root, relative paths are less
error-prone:

```toml
[workspace]

[dependencies]
tenferro-runtime = { path = "../crates/tenferro-runtime" }
tenferro-cpu = { path = "../crates/tenferro-cpu" }
tenferro-linalg = { path = "../crates/tenferro-linalg" }
```

The first build still needs network access unless dependencies are already
vendored or cached. The workspace pins git dependencies and uses crates.io
packages even when the tenferro crates themselves are local path dependencies.
The default `native` provider may take several minutes to compile the first
time on a fresh machine; later incremental builds are much faster.

With default features, this compiles the `native` provider, so a `CpuEngine`
uses faer. To use the LAPACK/BLAS CPU provider, enable
`blas` and link a BLAS/LAPACK provider from the build environment:

```toml
[dependencies]
tenferro-runtime = { path = "/path/to/tenferro-rs/crates/tenferro-runtime", default-features = false, features = ["blas"] }
tenferro-cpu = { path = "/path/to/tenferro-rs/crates/tenferro-cpu", default-features = false, features = ["blas"] }
tenferro-linalg = { path = "/path/to/tenferro-rs/crates/tenferro-linalg", default-features = false, features = ["blas"] }
```

CPU provider features are additive. At least one of `native` or `blas`
must be enabled, and builds may enable both. The compiled provider is what every
`CpuEngine` in the process uses: BLAS when `blas` is compiled, otherwise faer.
External CPU providers may replace
only a provider bundle slot, such as `dot_general`; they do not replace
`native` or `blas`, so one complete CPU provider is still required for
fallback and linalg coverage. The `blas` backend needs a BLAS/LAPACK
provider. Link one from the system toolchain with `blas`, or enable exactly
one explicit provider feature to select a source provider for BLAS, LAPACK, and
strided einsum:

```toml
[dependencies]
tenferro-runtime = { path = "/path/to/tenferro-rs/crates/tenferro-runtime", default-features = false, features = ["blas-openblas"] }
tenferro-cpu = { path = "/path/to/tenferro-rs/crates/tenferro-cpu", default-features = false, features = ["blas-openblas"] }
tenferro-linalg = { path = "/path/to/tenferro-rs/crates/tenferro-linalg", default-features = false, features = ["blas-openblas"] }
```

The explicit provider features are `blas-openblas`, `blas-accelerate`, and
`blas-mkl`. Cargo features are additive, so tenferro rejects builds that enable
more than one explicit BLAS provider.

Provider build scripts may need environment variables when using system
installations. OpenBLAS setups commonly use `OPENBLAS_LIB_DIR`; MKL setups
commonly use `MKLROOT` or `MKL_LIB_DIR`; Accelerate is the macOS system
framework provider.

Add `tenferro-ad`, `tenferro-einsum`, `tenferro-linalg`, `tenferro-fft`, or
`tenferro-gpu` when a workflow needs those layers. Enable `autodiff` on
operation crates such as `tenferro-linalg` when extension AD rules are needed.
For CPU-only eager or traced AD, add:

```toml
[dependencies]
tenferro-ad = { path = "/path/to/tenferro-rs/crates/tenferro-ad" }
```

Enable concrete backend features such as `cuda` on each crate that needs GPU
support:

```toml
[dependencies]
tenferro-ad = { path = "/path/to/tenferro-rs/crates/tenferro-ad", features = ["cuda"] }
tenferro-gpu = { path = "/path/to/tenferro-rs/crates/tenferro-gpu", features = ["cuda"] }
tenferro-linalg = { path = "/path/to/tenferro-rs/crates/tenferro-linalg", features = ["autodiff", "cuda"] }
```

## Quickstart A: Direct Tensor And Linalg

<!-- snippet-source: docs/tutorial-code/src/bin/direct_linalg_quickstart.rs -->
```rust
use tenferro_cpu::CpuEngine;
use tenferro_linalg::prelude::*;
use tenferro_runtime::prelude::*;

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error < 1.0e-12,
            "value {index}: actual={actual}, expected={expected}, error={error}"
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = CpuEngine::serial();

    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![3.0, 0.0, 0.0, 1.0])?;
    let identity = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0])?;
    let rhs = TypedTensor::<f64>::from_vec_col_major(vec![2, 1], vec![6.0, 2.0])?;

    // One borrowed engine carries every operation, the plan cache and the
    // buffer pool.
    let product = a.matmul(&identity, &mut engine)?;
    let solution = product.solve(&rhs, &mut engine)?;
    let singular_values = product.svdvals(&mut engine)?;
    assert_eq!(product.shape(), &[2, 2]);
    assert_close(product.host_data()?, &[3.0, 0.0, 0.0, 1.0]);
    assert_close(solution.as_slice()?, &[2.0, 2.0]);
    assert_close(singular_values.as_slice()?, &[3.0, 1.0]);

    Ok(())
}
```
<!-- end-snippet-source -->

Expected output: the program exits silently because the shape and value
assertions pass. The same caller-held `CpuEngine` carries matmul, solve, and
values-only SVD. The `from_vec_col_major` buffer is column-major: the leftmost
axis varies fastest in memory.

## Quickstart B: Traced AD

Add `tenferro-ad` when the same tensor stack needs graph-based derivatives:

```toml
[dependencies]
tenferro-runtime = "0.9"
tenferro-cpu = "0.9"
tenferro-ad = "0.9"
```

<!-- snippet-source: docs/tutorial-code/src/bin/traced_autodiff_jax_style.rs -->
```rust
use tenferro_ad::TracedTensorAdExt;
use tenferro_cpu::CpuEngine;
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TracedTensor};

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error < 1.0e-12,
            "value {index}: actual={actual}, expected={expected}, error={error}"
        );
    }
}

fn run(tensor: &TracedTensor) -> Result<tenferro_runtime::Tensor, Box<dyn std::error::Error>> {
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(tensor)?;
    let mut engine = CpuEngine::serial();
    let mut builder = Runtime::builder();
    builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
    let runtime = builder.build()?;
    let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
    let mut outputs = runtime.run(&mut [&mut engine], &prepared, &[])?;
    assert_eq!(outputs.len(), 1);
    Ok(outputs.remove(0).into_tensor(&mut engine)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0])?;
    let y = (&x * &x)?.reduce_sum(Some(&[0]))?;

    let y_value = run(&y)?;
    assert_eq!(y_value.shape(), &[] as &[usize]);
    assert_close(y_value.as_slice::<f64>().unwrap(), &[14.0]);

    let grad = y.grad(&x)?;
    let grad_value = run(&grad)?;
    assert_eq!(grad_value.shape(), &[3]);
    assert_close(grad_value.as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0]);

    let tangent = TracedTensor::from_vec_col_major(vec![3], vec![0.1_f64, 1.0, -2.0])?;
    let directional = y.jvp(&x, &tangent)?;
    let directional_value = run(&directional)?;
    assert_eq!(directional_value.shape(), &[] as &[usize]);
    assert_close(directional_value.as_slice::<f64>().unwrap(), &[-7.8]);

    Ok(())
}
```
<!-- end-snippet-source -->

## Next Steps

After the first CPU program, read [Core Concepts](./core-concepts.md) for the
main mental model or [Choosing a Tensor API](../guides/choosing-an-api.md) to
pick between `TypedTensor`, `Tensor`, `EagerTensor`, and `TracedTensor`. The
sidebar links to the full guides for memory order, CUDA, tensor operations,
autodiff, einsum, and linear algebra.
