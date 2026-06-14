# XLA and PJRT

The `tenferro-xla` crate is an experimental peer executor for static-shaped
traced programs. It lowers tenferro programs to StableHLO and can load PJRT
plugins when the optional `pjrt` feature is enabled.

It is not a `TensorBackend` implementation and it does not replace native CPU,
CUDA, or WebGPU execution. Dynamic-shape graphs, extension runtimes, and the
full tensor backend contract still run through `GraphExecutor<B>`.

## Supported Boundary

The initial StableHLO lowering accepts exact static shapes and these dtypes:

- `F32`
- `F64`

The initial operation subset is:

- `Constant`
- `Add`
- `Multiply`
- `Negate`
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

```bash
TENFERRO_PJRT_PLUGIN=/path/to/pjrt_c_api_cpu_plugin.so \
  cargo test -p tenferro-xla --features pjrt --test pjrt_env
```

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
use C-contiguous host buffers. The XLA crate keeps explicit conversion helpers
at that boundary so the physical host-order conversion is not hidden inside the
native runtime.

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
  cargo test -p tenferro-xla --test xla_tool_execution -- --nocapture
```

The test covers both a direct static tensor graph and the fixed-shape N-ary
einsum tutorial graph after the einsum extension expands to standard
`dot_general` operations.

On a configured NVIDIA machine, use `CUDA` for the platform:

```bash
CUDA_PATH=/usr/local/cuda-12.8 \
LD_LIBRARY_PATH=$CUDA_PATH/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH \
TENFERRO_XLA_RUN_HLO_MODULE=/path/to/run_hlo_module \
TENFERRO_XLA_RUN_HLO_PLATFORM=CUDA \
  cargo test -p tenferro-xla --test xla_tool_execution -- --nocapture
```

If `TENFERRO_XLA_RUN_HLO_MODULE` is not set, the test exits successfully after
printing a skip message. This keeps normal CPU-only CI independent of a local
OpenXLA checkout.
