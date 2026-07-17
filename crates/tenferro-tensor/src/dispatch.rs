//! Dtype-stripping dispatch macros for erased tensor values.

/// Dispatch a dtype-erased [`Tensor`](crate::Tensor) to a typed tensor body.
///
/// The dtype-set guard keeps unsupported dtype rejection at the boundary where
/// the backend and operation name are still visible.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{BackendId, Tensor};
///
/// let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
/// let shape = tenferro_tensor::with_scalar!(
///     &tensor,
///     float_only,
///     backend = BackendId::Cpu,
///     op = "shape_probe",
///     |typed| -> tenferro_tensor::Result<Vec<usize>> { Ok(typed.shape().to_vec()) }
/// )?;
/// assert_eq!(shape, vec![2]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[macro_export]
macro_rules! with_scalar {
    ($tensor:expr, all, backend = $backend:expr, op = $op:expr, |$typed:ident| $(-> $ret:ty)? $body:block) => {{
        let _ = &$backend;
        let _ = &$op;
        match $tensor {
            $crate::Tensor::F32($typed) => $body,
            $crate::Tensor::F64($typed) => $body,
            $crate::Tensor::I32($typed) => $body,
            $crate::Tensor::I64($typed) => $body,
            $crate::Tensor::Bool($typed) => $body,
            $crate::Tensor::C32($typed) => $body,
            $crate::Tensor::C64($typed) => $body,
        }
    }};
    ($tensor:expr, numeric, backend = $backend:expr, op = $op:expr, |$typed:ident| $(-> $ret:ty)? $body:block) => {{
        match $tensor {
            $crate::Tensor::F32($typed) => $body,
            $crate::Tensor::F64($typed) => $body,
            $crate::Tensor::I32($typed) => $body,
            $crate::Tensor::I64($typed) => $body,
            $crate::Tensor::C32($typed) => $body,
            $crate::Tensor::C64($typed) => $body,
            other => Err($crate::Error::unsupported_dtype_conversion(
                $op,
                other.dtype(),
                other.dtype(),
                format!("backend {} does not support this operation/dtype", $backend),
            )),
        }
    }};
    ($tensor:expr, float_complex, backend = $backend:expr, op = $op:expr, |$typed:ident| $(-> $ret:ty)? $body:block) => {{
        match $tensor {
            $crate::Tensor::F32($typed) => $body,
            $crate::Tensor::F64($typed) => $body,
            $crate::Tensor::C32($typed) => $body,
            $crate::Tensor::C64($typed) => $body,
            other => Err($crate::Error::unsupported_dtype_conversion(
                $op,
                other.dtype(),
                other.dtype(),
                format!("backend {} does not support this operation/dtype", $backend),
            )),
        }
    }};
    ($tensor:expr, float_only, backend = $backend:expr, op = $op:expr, |$typed:ident| $(-> $ret:ty)? $body:block) => {{
        match $tensor {
            $crate::Tensor::F32($typed) => $body,
            $crate::Tensor::F64($typed) => $body,
            other => Err($crate::Error::unsupported_dtype_conversion(
                $op,
                other.dtype(),
                other.dtype(),
                format!("backend {} does not support this operation/dtype", $backend),
            )),
        }
    }};
}

/// Dispatch a [`TensorRead`](crate::TensorRead) to a typed tensor view body.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{BackendId, Tensor, TensorRead};
///
/// let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0])?;
/// let read = TensorRead::from_tensor(&tensor);
/// let shape = tenferro_tensor::with_scalar_read!(
///     read,
///     float_only,
///     backend = BackendId::Cpu,
///     op = "shape_probe",
///     |view| -> tenferro_tensor::Result<Vec<usize>> { Ok(view.shape().to_vec()) }
/// )?;
/// assert_eq!(shape, vec![2]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[macro_export]
macro_rules! with_scalar_read {
    ($read:expr, all, backend = $backend:expr, op = $op:expr, |$view:ident| $(-> $ret:ty)? $body:block) => {{
        let _ = &$backend;
        let _ = &$op;
        match $read {
            $crate::TensorRead::Tensor(tensor) => match tensor {
                $crate::Tensor::F32(tensor) => {
                    let $view = tensor.as_view();
                    $body
                }
                $crate::Tensor::F64(tensor) => {
                    let $view = tensor.as_view();
                    $body
                }
                $crate::Tensor::I32(tensor) => {
                    let $view = tensor.as_view();
                    $body
                }
                $crate::Tensor::I64(tensor) => {
                    let $view = tensor.as_view();
                    $body
                }
                $crate::Tensor::Bool(tensor) => {
                    let $view = tensor.as_view();
                    $body
                }
                $crate::Tensor::C32(tensor) => {
                    let $view = tensor.as_view();
                    $body
                }
                $crate::Tensor::C64(tensor) => {
                    let $view = tensor.as_view();
                    $body
                }
            },
            $crate::TensorRead::View(view) => match view {
                $crate::TensorView::F32($view) => $body,
                $crate::TensorView::F64($view) => $body,
                $crate::TensorView::I32($view) => $body,
                $crate::TensorView::I64($view) => $body,
                $crate::TensorView::Bool($view) => $body,
                $crate::TensorView::C32($view) => $body,
                $crate::TensorView::C64($view) => $body,
            },
        }
    }};
    ($read:expr, numeric, backend = $backend:expr, op = $op:expr, |$view:ident| $(-> $ret:ty)? $body:block) => {{
        let read = $read;
        let dtype = read.dtype();
        match read {
            $crate::TensorRead::Tensor(tensor) => match tensor {
                $crate::Tensor::F32(tensor) => {
                    let $view = tensor.as_view();
                    $body
                }
                $crate::Tensor::F64(tensor) => {
                    let $view = tensor.as_view();
                    $body
                }
                $crate::Tensor::I32(tensor) => {
                    let $view = tensor.as_view();
                    $body
                }
                $crate::Tensor::I64(tensor) => {
                    let $view = tensor.as_view();
                    $body
                }
                $crate::Tensor::C32(tensor) => {
                    let $view = tensor.as_view();
                    $body
                }
                $crate::Tensor::C64(tensor) => {
                    let $view = tensor.as_view();
                    $body
                }
                $crate::Tensor::Bool(_) => Err($crate::Error::unsupported_dtype_conversion(
                    $op,
                    dtype,
                    dtype,
                    format!("backend {} does not support this operation/dtype", $backend),
                )),
            },
            $crate::TensorRead::View(view) => match view {
                $crate::TensorView::F32($view) => $body,
                $crate::TensorView::F64($view) => $body,
                $crate::TensorView::I32($view) => $body,
                $crate::TensorView::I64($view) => $body,
                $crate::TensorView::C32($view) => $body,
                $crate::TensorView::C64($view) => $body,
                $crate::TensorView::Bool(_) => Err($crate::Error::unsupported_dtype_conversion(
                    $op,
                    dtype,
                    dtype,
                    format!("backend {} does not support this operation/dtype", $backend),
                )),
            },
        }
    }};
    ($read:expr, float_complex, backend = $backend:expr, op = $op:expr, |$view:ident| $(-> $ret:ty)? $body:block) => {{
        let read = $read;
        let dtype = read.dtype();
        match read {
            $crate::TensorRead::Tensor(tensor) => match tensor {
                $crate::Tensor::F32(tensor) => {
                    let $view = tensor.as_view();
                    $body
                }
                $crate::Tensor::F64(tensor) => {
                    let $view = tensor.as_view();
                    $body
                }
                $crate::Tensor::C32(tensor) => {
                    let $view = tensor.as_view();
                    $body
                }
                $crate::Tensor::C64(tensor) => {
                    let $view = tensor.as_view();
                    $body
                }
                $crate::Tensor::I32(_) | $crate::Tensor::I64(_) | $crate::Tensor::Bool(_) => {
                    Err($crate::Error::unsupported_dtype_conversion(
                        $op,
                        dtype,
                        dtype,
                        format!("backend {} does not support this operation/dtype", $backend),
                    ))
                }
            },
            $crate::TensorRead::View(view) => match view {
                $crate::TensorView::F32($view) => $body,
                $crate::TensorView::F64($view) => $body,
                $crate::TensorView::C32($view) => $body,
                $crate::TensorView::C64($view) => $body,
                $crate::TensorView::I32(_)
                | $crate::TensorView::I64(_)
                | $crate::TensorView::Bool(_) => Err($crate::Error::unsupported_dtype_conversion(
                    $op,
                    dtype,
                    dtype,
                    format!("backend {} does not support this operation/dtype", $backend),
                )),
            },
        }
    }};
    ($read:expr, float_only, backend = $backend:expr, op = $op:expr, |$view:ident| $(-> $ret:ty)? $body:block) => {{
        let read = $read;
        let dtype = read.dtype();
        match read {
            $crate::TensorRead::Tensor(tensor) => match tensor {
                $crate::Tensor::F32(tensor) => {
                    let $view = tensor.as_view();
                    $body
                }
                $crate::Tensor::F64(tensor) => {
                    let $view = tensor.as_view();
                    $body
                }
                $crate::Tensor::I32(_)
                | $crate::Tensor::I64(_)
                | $crate::Tensor::Bool(_)
                | $crate::Tensor::C32(_)
                | $crate::Tensor::C64(_) => Err($crate::Error::unsupported_dtype_conversion(
                    $op,
                    dtype,
                    dtype,
                    format!("backend {} does not support this operation/dtype", $backend),
                )),
            },
            $crate::TensorRead::View(view) => match view {
                $crate::TensorView::F32($view) => $body,
                $crate::TensorView::F64($view) => $body,
                $crate::TensorView::I32(_)
                | $crate::TensorView::I64(_)
                | $crate::TensorView::Bool(_)
                | $crate::TensorView::C32(_)
                | $crate::TensorView::C64(_) => Err($crate::Error::unsupported_dtype_conversion(
                    $op,
                    dtype,
                    dtype,
                    format!("backend {} does not support this operation/dtype", $backend),
                )),
            },
        }
    }};
}
