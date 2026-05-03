# tenferro-cubecl

CubeCL kernels and launch helpers for tenferro.

This crate owns GPU kernel definitions and launch validation only. It does not
own tenferro tensor values, device placement, CPU/GPU transfers, or backend
dispatch.

## Implemented Operations

All reductions use single-axis keepdims semantics inside this crate. For an
input tensor `x` with shape `(d_0, ..., d_{r-1})` and reduction axis `a`, the
output tensor `y` has shape:

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
`[1, d_0, d_0 * d_1, ...]`.
