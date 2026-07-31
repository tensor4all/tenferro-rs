# XLA and PJRT

The `tenferro-xla` crate is an experimental peer executor for static-shaped
traced programs. It lowers tenferro programs to StableHLO and can load PJRT
plugins when the optional `pjrt` feature is enabled.

It is not a `TensorBackend` implementation and it does not replace native CPU,
CUDA, or WebGPU execution. Dynamic-shape graphs, extension runtimes, and the
full tensor backend contract still run through `Runtime::run_compiled`.

## Supported Boundary

The initial StableHLO lowering accepts exact static shapes and these dtypes:

- `F32`
- `F64`

The Phase 1 operation subset is:

- `Constant`
- `Add`
- `Multiply`
- `Negate`
- `Divide`
- `Abs`
- `Exp`
- `Log`
- `Sin`
- `Cos`
- `Tanh`
- `Sqrt`
- `Rsqrt`
- `Pow`
- `Expm1`
- `Log1p`
- `Convert`
- `Reshape`
- `BroadcastInDim`
- `Transpose`
- `ReduceSum`
- `DotGeneral`

Unsupported dtypes, dynamic or upper-bound shape extents, extension operations
without a fixed-shape standard-op lowering, and operation variants outside this
subset are rejected before PJRT is called. If an operation-family API expands
to supported standard ops, as fixed-shape N-ary einsum can, the resulting graph
can still lower through this path.

`Compare`, `Select`, `Maximum`, `Minimum`, `Clamp`, `Sign`, `Conj`, integer
dtypes, `Bool`, and complex dtypes remain outside Phase 1. They need additional
dtype plumbing or explicit NaN/edge-case parity tests before being enabled.

## Lowering Example

For a complete checked einsum path, see the
[XLA backend einsum tutorial](../tutorials/xla-einsum-backend.md). Minimal
`lower_to_stablehlo` examples also live in the `tenferro-xla` rustdoc, where
they run as doctests.

## Runtime Loading

PJRT is loaded at runtime. The `tenferro-xla` crate does not link XLA or PJRT
into `tenferro-runtime`.

Use one of these variables to point tenferro at a PJRT plugin shared library:

```bash
export TENFERRO_PJRT_PLUGIN=/path/to/pjrt_c_api_cpu_plugin.so
export TENFERRO_PJRT_GPU_PLUGIN=/path/to/pjrt_c_api_gpu_plugin.so
```

`TENFERRO_PJRT_PLUGIN` is the default loader variable. Use
`TENFERRO_PJRT_GPU_PLUGIN` when a script wants to keep CPU and GPU plugin paths
separate.

With the `pjrt` feature enabled, `XlaExecutor::from_env()` reads
`TENFERRO_PJRT_PLUGIN` and opens the plugin with `dlopen`. If the variable is
unset, empty, points to a missing file, or the library does not export
`GetPjrtApi`, tenferro returns an explicit error.

## Threading And Synchronization

PJRT owns the plugin client, device streams, and intra-op thread pool. A
`CpuBackend::with_threads(n)` setting does not change the number of workers
used inside a PJRT CPU plugin, and tenferro does not infer a PJRT affinity or
NUMA guarantee from that setting. Configure intra-op parallelism through the
selected PJRT/XLA deployment, such as the plugin's documented environment or
XLA flags, and treat those controls as plugin-owned and generally
process-scoped.

The tenferro runtime controls graph lowering, engine selection, and the
explicit call into PJRT. It does not insert a hidden CPU fallback or a second
inner pool. PJRT execution and device-to-host transfer establish the
host-visible synchronization boundary; use the executor's documented
synchronization behavior when a host-side barrier is required. Unsupported
shapes, dtypes, operations, and plugin capabilities are rejected before or by
the PJRT call with an error.

```bash
TENFERRO_PJRT_PLUGIN=/path/to/pjrt_c_api_cpu_plugin.so \
  cargo test -p tenferro-xla --features pjrt --test integration pjrt_env
```

OpenXLA/JAX CUDA PJRT wheels provide a prebuilt C API plugin. For example, on a
CUDA 12 Linux machine:

```bash
python3 -m pip download --only-binary=:all: --no-deps \
  --dest /tmp/tenferro-openxla-prebuilt \
  jax-cuda12-pjrt==0.10.2 \
  nvidia-cudnn-cu12==9.23.2.1 \
  nvidia-cuda-nvcc-cu12==12.9.86

mkdir -p /tmp/tenferro-openxla-prebuilt/unpacked
python3 - <<'PY'
import pathlib, zipfile
root = pathlib.Path("/tmp/tenferro-openxla-prebuilt")
for wheel in root.glob("*.whl"):
    out = root / "unpacked" / wheel.stem
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(out)
PY

export TENFERRO_PJRT_PLUGIN=/tmp/tenferro-openxla-prebuilt/unpacked/jax_cuda12_pjrt-0.10.2-py3-none-manylinux_2_27_x86_64/jax_plugins/xla_cuda12/xla_cuda_plugin.so
export LD_LIBRARY_PATH=/tmp/tenferro-openxla-prebuilt/unpacked/nvidia_cudnn_cu12-9.23.2.1-py3-none-manylinux_2_27_x86_64/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/tmp/tenferro-openxla-prebuilt/unpacked/nvidia_cuda_nvcc_cu12-12.9.86-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64/nvidia/cuda_nvcc

cargo test -p tenferro-xla --features pjrt --test integration pjrt_execution -- --nocapture
```

`XlaExecutor::run_with_inputs` and `XlaExecutor::run_many_with_inputs` use a
loaded PJRT plugin to compile the generated StableHLO, upload compact
column-major tenferro host tensors with explicit PJRT `byte_strides`, execute
on one addressable device, and download outputs back into tenferro tensors.
Calling these methods without the `pjrt` feature returns `PjrtFeatureDisabled`;
calling them on `XlaExecutor::default()` with the feature enabled returns
`PjrtPluginNotLoaded`.

This example compiles a fixed-shape N-ary einsum followed by elementwise ops.
It always checks StableHLO lowering. When `TENFERRO_PJRT_PLUGIN` is set, the
same binary also executes the graph through PJRT:

<!-- snippet-source: docs/tutorial-code/src/bin/xla_pjrt_execution.rs -->
```rust
use tenferro_einsum::TraceContextEinsumExt;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::program::{CoreSemanticOp, ProgramInputSpec};
use tenferro_runtime::{DType, GraphCompiler, Tensor, TraceContext};
use tenferro_xla::{XlaExecutor, TENFERRO_PJRT_PLUGIN_ENV};

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let residual = (actual - expected).abs();
        assert!(
            residual <= 1.0e-3,
            "value {index} differs: actual={actual}, expected={expected}, residual={residual}"
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut trace = TraceContext::new();
    let inputs = [[2, 3], [3, 4], [4, 2]]
        .map(|shape| {
            trace.input(ProgramInputSpec::new(
                DType::F32,
                DimExpr::from_concrete(&shape),
            ))
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let mut y = trace.einsum(&inputs, "ij,jk,kl->il")?;
    for op in [
        CoreSemanticOp::Abs,
        CoreSemanticOp::Sqrt,
        CoreSemanticOp::Log1p,
        CoreSemanticOp::Exp,
    ] {
        y = trace.add_op(op, &[y])?[0];
    }
    let graph = trace.finish(&[y])?;
    let program = GraphCompiler::new().compile_traced_graph(&graph)?;

    let module = XlaExecutor::default().lower_compiled_to_stablehlo(&program)?;
    let stablehlo = module.as_str();
    assert!(stablehlo.contains("stablehlo.dot_general"));
    assert!(stablehlo.contains("stablehlo.abs"));
    assert!(stablehlo.contains("stablehlo.sqrt"));
    assert!(stablehlo.contains("stablehlo.log_plus_one"));
    assert!(stablehlo.contains("stablehlo.exponential"));

    if std::env::var_os(TENFERRO_PJRT_PLUGIN_ENV).is_none() {
        return Ok(());
    }

    let lhs_values = vec![1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0];
    let mid_values = vec![
        1.0_f32, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
    ];
    let rhs_values = vec![1.0_f32, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0];
    let lhs_input = Tensor::from_vec_col_major(vec![2, 3], lhs_values.clone())?;
    let mid_input = Tensor::from_vec_col_major(vec![3, 4], mid_values.clone())?;
    let rhs_input = Tensor::from_vec_col_major(vec![4, 2], rhs_values.clone())?;

    let output = XlaExecutor::from_env()?
        .run_compiled_with_inputs(&program, &[&lhs_input, &mid_input, &rhs_input])?;
    assert_eq!(output.shape(), &[2, 2]);
    assert_close(
        output.as_slice::<f32>().unwrap(),
        &[29.495_613, 43.871_902, 32.622_776, 48.539_455],
    );

    Ok(())
}
```
<!-- end-snippet-source -->

## CUDA and cuTENSOR Setup

For GPU PJRT plugins and the native CUDA backend, make the CUDA toolkit and
CUDA libraries visible to the dynamic loader. Choose the installed CUDA root on
your machine:

```bash
ls -d /usr/local/cuda*
export CUDA_PATH=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```

If cuTENSOR is installed outside the CUDA toolkit directory, include its
library directory too:

```bash
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH
```

For tenferro's native CUDA backend, exact runtime-library overrides are also
available:

```bash
export TENFERRO_CUTENSOR_PATH=/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2
export TENFERRO_CUSOLVER_PATH=$CUDA_PATH/lib64/libcusolver.so.12
export TENFERRO_CUBLAS_PATH=$CUDA_PATH/lib64/libcublas.so.12
export CUBECL_DEBUG_LOG=0
```

The `TENFERRO_CUTENSOR_PATH`, `TENFERRO_CUSOLVER_PATH`, and
`TENFERRO_CUBLAS_PATH` variables are for tenferro's CubeCL/CUDA backend. PJRT
plugins may have their own dynamic-library requirements, but they are still
loaded from the plugin path supplied by `TENFERRO_PJRT_PLUGIN` or
`TENFERRO_PJRT_GPU_PLUGIN`.

## StableHLO Shape and Layout Notes

StableHLO tensor types record logical dimension order. They do not say whether
host memory is row-major or column-major.

tenferro host tensors are compact column-major. PJRT host transfer paths often
use C-contiguous host buffers. The XLA crate keeps that boundary explicit:
input upload passes column-major byte strides to PJRT, while output download
requests a column-major host layout from PJRT and constructs the returned
tenferro tensor directly from that buffer.

`dot_general` has a separate logical-order issue. StableHLO reports batched
`dot_general` results as batch dimensions first, followed by free lhs and rhs
dimensions. tenferro's `DotGeneralConfig` uses free lhs dimensions, free rhs
dimensions, then batch dimensions. The XLA lowering inserts a StableHLO
`transpose` after batched `dot_general` so the result shape matches tenferro's
logical contract.

## External StableHLO Execution Check

The repository includes an environment-gated test that executes generated
StableHLO through OpenXLA's `run_hlo_module` tool when that tool is available:

```bash
TENFERRO_XLA_RUN_HLO_MODULE=/path/to/run_hlo_module \
TENFERRO_XLA_RUN_HLO_PLATFORM=Host \
  cargo test -p tenferro-xla --test integration xla_tool_execution -- --nocapture
```

The test covers a direct static tensor graph, the Phase 1 real-floating
elementwise subset, and the fixed-shape N-ary einsum tutorial graph after the
einsum extension expands to standard `dot_general` operations.

On a configured NVIDIA machine, use `CUDA` for the platform:

```bash
CUDA_PATH=/usr/local/cuda-12.8 \
LD_LIBRARY_PATH=$CUDA_PATH/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH \
TENFERRO_XLA_RUN_HLO_MODULE=/path/to/run_hlo_module \
TENFERRO_XLA_RUN_HLO_PLATFORM=CUDA \
  cargo test -p tenferro-xla --test integration xla_tool_execution -- --nocapture
```

If `TENFERRO_XLA_RUN_HLO_MODULE` is not set, the test exits successfully after
printing a skip message. This keeps normal CPU-only CI independent of a local
OpenXLA checkout.
