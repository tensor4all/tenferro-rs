# tenferro-internal-gpubackend

CubeCL kernels and selected launch helpers for tenferro.

This crate owns GPU kernel definitions and selected launch validation. It does
not own tenferro tensor values, device placement, CPU/GPU transfers, or backend
dispatch. Higher-level crates pass validated CubeCL buffers, shapes, strides,
and operation metadata into these kernels.

## Implemented Operations

### Elementwise

Elementwise kernels operate on linear logical positions. For compatible input
tensors `x`, `z`, and output `y`, the mathematical definitions are:

| Operation | Dtypes | Definition |
| --- | --- | --- |
| `add` | float, complex | `y_i = x_i + z_i` |
| `mul` | float, complex | `y_i = x_i * z_i` |
| `div` | float, complex | `y_i = x_i / z_i` |
| `neg` | float, complex | `y_i = -x_i` |
| `conj` | complex | `y_i = conjugate(x_i)` |
| `exp`, `log`, `sin`, `cos`, `tanh`, `sqrt`, `rsqrt`, `expm1`, `log1p` | float | `y_i = f(x_i)` |
| `abs`, `sign` | float | `y_i = f(x_i)` |
| `maximum`, `minimum`, `pow` | float | `y_i = f(x_i, z_i)` |
| `compare` | float | `y_i = 1` when the selected comparison is true, otherwise `0` |
| `select` | float | `y_i = on_true_i` when `pred_i != 0`, otherwise `on_false_i` |
| `clamp` | float | `y_i = min(max(x_i, lower_i), upper_i)` |

### Structural And Indexing

These kernels follow tenferro's dense column-major logical indexing. For logical
multi-indices `i`, input tensor `x`, and output tensor `y`:

| Operation | Definition |
| --- | --- |
| `transpose` | `y_i = x_{i[perm^{-1}]}` |
| `broadcast_in_dim` | `y_i = x_{project(i, dims)}` |
| `reverse` | `y_i = x_j`, where `j_a = d_a - 1 - i_a` for reversed axes and `j_a = i_a` otherwise |
| `concatenate` | `y` copies each input into the selected axis interval |
| `convert` | `y_i = cast(x_i)` |
| `slice` | `y_i = x_{starts + i * strides}` |
| `dynamic_slice` | `y_i = x_{clamp(starts) + i}` |
| `pad` | `y_i = x_j` for in-bounds unpadded positions, otherwise `0` |
| `gather` | `y_i = operand_{mapped_start(i) + window_offset(i)}` |
| `scatter` | `y` starts as `operand`; each valid update performs `y_dst += updates_src` |
| `extract_diagonal` | `y_i = x_j` where the selected axes in `j` share the same diagonal coordinate |
| `embed_diagonal` | diagonal positions receive `x_i`; other positions receive `0` |
| `tril`, `triu` | matrix triangle positions preserve `x_i`; masked positions receive `0` |

The scatter definition matches the StableHLO-style add-scatter operation used by
tenferro. JAX `lax.scatter_add` and PyTorch `scatter_add`/`index_add` are useful
semantic references, but tenferro's full windowed scatter shape contract is the
backend contract here. GPU scatter initializes the output with a parallel copy
from `operand`, then applies valid updates in parallel with atomic add for
overlapping destinations. Complex scatter atomically accumulates the real and
imaginary scalar parts separately.

### Reductions

All reductions use single-axis keepdims semantics inside this crate. For an input
tensor `x` with shape `(d_0, ..., d_{r-1})` and reduction axis `a`, the output
tensor `y` has shape:

```text
(d_0, ..., d_{a-1}, 1, d_{a+1}, ..., d_{r-1})
```

For every output index `i = (i_0, ..., i_{r-1})` with `i_a = 0`, define:

```text
i[k <- a] = (i_0, ..., i_{a-1}, k, i_{a+1}, ..., i_{r-1})
```

| Operation | Dtypes | Definition |
| --- | --- | --- |
| `reduce_sum` | `f32`, `f64`, `i64`, `Complex32`, `Complex64` | `y_i = sum_{k=0}^{d_a-1} x_{i[k <- a]}` |
| `reduce_prod` | `f32`, `f64`, `i64`, `Complex32`, `Complex64` | `y_i = prod_{k=0}^{d_a-1} x_{i[k <- a]}` |
| `reduce_max` | `f32`, `f64` | `y_i = max_{0 <= k < d_a} x_{i[k <- a]}` |
| `reduce_min` | `f32`, `f64` | `y_i = min_{0 <= k < d_a} x_{i[k <- a]}` |

Complex `max` and `min` are unsupported because complex numbers have no
canonical ordering. `i64 max` and `i64 min` are unsupported in this first split
because the CPU backend does not currently expose them either.

## Layout

Shape and strides are runtime metadata supplied by the caller through CubeCL
`TensorBinding`. tenferro passes dense column-major strides such as
`[1, d_0, d_0 * d_1, ...]`. The full backend metadata contract is documented
in [`docs/design/gpu-backend-design.md`](../docs/design/gpu-backend-design.md#kernel-metadata-contract).

## Third-Party Notices

The reduction launch strategy includes code adapted from CubeK Reduce. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for upstream copyright,
source commit, source paths, license, and change notices. The adapted portions
preserve CubeK Reduce's `MIT OR Apache-2.0` dual-license grant.
