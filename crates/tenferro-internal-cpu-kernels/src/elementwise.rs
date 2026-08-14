use std::mem::{size_of_val, MaybeUninit};
use std::ops::{Add, Div, Mul, Neg, Rem as StdRem, Sub};
use std::ptr::NonNull;

use num_complex::Complex;
use num_traits::{One, Zero};
use strided_kernel::{
    batched_outer_product_into_uninit, broadcast_mul_into_uninit, compare_into_uninit, map_into,
    mul_into_uninit, reduce, zip_map2_into, zip_map3_into, CompareOp, ErasedFusedPlan,
    ErasedRawStridedPtr, ErasedRawStridedRef, ErasedRawStridedUninitMut, ExecContext, FusedInst,
    FusedOp, FusedPlan, KernelDType, StridedView,
};

use crate::buffer_pool::{BufferPool, PoolScalar};
use crate::ConjElem;
use crate::PooledUninitOutput;
use tenferro_tensor::backend::{
    ElementwiseFusionInputView, ElementwiseFusionOp, ElementwiseFusionPlan,
};
use tenferro_tensor::{
    col_major_strides, CompareDir, DType, Tensor, TensorRank, TensorRead, TensorScalar,
    TensorValue, TensorView, TypedTensor, TypedTensorView,
};

use super::{typed_host_data, typed_view, typed_view_from_view};

// This hidden public boundary is required by the separate tenferro-cpu crate;
// that crate re-exports it only at crate visibility and no user-facing safe
// wrapper exposes the raw constructor.
/// Construct an erased read-only strided view over initialized typed storage.
///
/// # Safety
///
/// The caller must ensure that `data` retains the alignment and byte layout
/// required by `dtype`, and that `dtype` agrees with the elements in the
/// backing storage. Every element reachable through `dims`, `strides`, and
/// `offset` must be within `data`; the metadata must remain alive for `'a`, and
/// the borrow must not be invalidated or mutated incompatibly while the view
/// exists. All reachable elements must be initialized and readable for the
/// lifetime of the returned view.
#[doc(hidden)]
pub unsafe fn erased_raw_strided_ref<'a>(
    dtype: KernelDType,
    data: &'a [u8],
    dims: &'a [usize],
    strides: &'a [isize],
    offset: isize,
) -> strided_kernel::Result<ErasedRawStridedRef<'a>> {
    let data_ptr = NonNull::new(data.as_ptr().cast_mut()).unwrap_or_else(NonNull::dangling);
    // SAFETY: all documented preconditions are met: `dtype` matches the
    // aligned, initialized storage; the metadata keeps every reachable element
    // in bounds; and the storage and metadata remain valid for the returned view.
    unsafe {
        ErasedRawStridedRef::from_raw_parts(dtype, data_ptr, data.len(), dims, strides, offset)
    }
}

// The same narrow cross-crate boundary is used for the uninitialized output
// constructor; it is not a user-facing API and has no safe compatibility shim.
/// Construct an erased writable strided view over exclusively owned storage.
///
/// # Safety
///
/// The caller must ensure that `data` retains the alignment and byte layout
/// required by `dtype`, and that `dtype` agrees with the eventual typed output.
/// Every element reachable through `dims`, `strides`, and `offset` must be
/// within the allocation, with all metadata and the exclusive borrow valid for
/// `'a`. No other reference may access the allocation while the view exists.
/// The caller must keep the storage uninitialized only as `MaybeUninit` until
/// every reachable element has been fully written, and must not expose it as
/// typed storage before that full initialization is proven.
#[doc(hidden)]
pub unsafe fn erased_raw_strided_uninit_mut<'a>(
    dtype: KernelDType,
    data: &'a mut [MaybeUninit<u8>],
    dims: &'a [usize],
    strides: &'a [isize],
    offset: isize,
) -> strided_kernel::Result<ErasedRawStridedUninitMut<'a>> {
    let data_ptr = NonNull::new(data.as_mut_ptr().cast::<u8>()).unwrap_or_else(NonNull::dangling);
    // SAFETY: all documented preconditions are met: `dtype` matches the
    // aligned storage; the metadata keeps every reachable element in bounds;
    // and the exclusive borrow remains valid while the storage is kept as
    // `MaybeUninit` until every reachable element is written before exposure.
    unsafe {
        ErasedRawStridedUninitMut::from_raw_parts(
            dtype,
            data_ptr,
            data.len(),
            dims,
            strides,
            offset,
        )
    }
}

macro_rules! dispatch_ternary_result_with_pool {
    ($op:literal, $a:expr, $b:expr, $c:expr, |$x:ident, $y:ident, $z:ident| $body:expr) => {
        match ($a, $b, $c) {
            (Tensor::F32($x), Tensor::F32($y), Tensor::F32($z)) => Ok(Tensor::F32($body?)),
            (Tensor::F64($x), Tensor::F64($y), Tensor::F64($z)) => Ok(Tensor::F64($body?)),
            _ => Err(ternary_dtype_error(
                $op,
                [$a.dtype(), $b.dtype(), $c.dtype()],
            )),
        }
    };
}

fn ternary_dtype_error(op: &'static str, dtypes: [DType; 3]) -> crate::Error {
    if dtypes[0] != dtypes[1] {
        dtype_pair_error(op, dtypes[0], dtypes[1])
    } else if dtypes[0] != dtypes[2] {
        dtype_pair_error(op, dtypes[0], dtypes[2])
    } else {
        dtype_pair_error(op, dtypes[0], dtypes[0])
    }
}

fn dtype_pair_error(op: &'static str, lhs: DType, rhs: DType) -> crate::Error {
    if lhs == rhs {
        let supported = match op {
            "clamp" => "F32/F64",
            "maximum" | "minimum" | "rem" => "F32/F64/I32/I64",
            "add" | "mul" | "sub" => "F32/F64/I32/I64/C32/C64",
            _ => unreachable!("dtype_pair_error has no supported-dtype contract for {op}"),
        };
        crate::Error::unsupported(
            op,
            format!("unsupported dtype {lhs:?}; supported dtypes: {supported}"),
        )
    } else {
        crate::Error::dtype_mismatch(op, lhs, rhs)
    }
}

fn unary_dtype_error(
    op: &'static str,
    dtype: DType,
    supported: &'static str,
    recommend_f64: bool,
) -> crate::Error {
    let remedy = (recommend_f64 && matches!(dtype, DType::I32 | DType::I64))
        .then_some("; convert to F64 before this operation");
    crate::Error::unsupported(
        op,
        format!(
            "unsupported dtype {dtype:?}; supported dtypes: {supported}{}",
            remedy.unwrap_or("")
        ),
    )
}

fn tensor_pair_error(op: &'static str, lhs: &Tensor, rhs: &Tensor) -> crate::Error {
    dtype_pair_error(op, lhs.dtype(), rhs.dtype())
}

fn read_pair_error(op: &'static str, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Error {
    dtype_pair_error(op, lhs.dtype(), rhs.dtype())
}

fn is_complex_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::C32 | DType::C64)
}

fn ordered_complex_error(op: &'static str) -> crate::Error {
    crate::Error::unsupported(
        op,
        "complex tensors do not have a total order; compute abs/norm explicitly before ordered operations",
    )
}

fn reject_complex_ordered_dtypes(op: &'static str, dtypes: &[DType]) -> crate::Result<()> {
    if dtypes.iter().copied().any(is_complex_dtype) {
        return Err(ordered_complex_error(op));
    }
    Ok(())
}

fn reject_complex_unsupported_compare_dtypes(
    dir: &CompareDir,
    dtypes: &[DType],
) -> crate::Result<()> {
    if *dir != CompareDir::Eq {
        reject_complex_ordered_dtypes("compare", dtypes)?;
    }
    Ok(())
}

const ELEMENTWISE_FUSION_OP: &str = "execute_elementwise_fusion";
const ELEMENTWISE_FUSION_MIN_ELEMENTS: usize = 16 * 1024;

fn validate_elementwise_fusion_inputs(
    inputs: &[&Tensor],
    plan: &ElementwiseFusionPlan,
) -> crate::Result<bool> {
    if inputs.len() != plan.input_count() {
        return Err(crate::Error::invalid_argument(
            ELEMENTWISE_FUSION_OP,
            "inputs",
            format!(
                "plan expects {} inputs but backend received {}",
                plan.input_count(),
                inputs.len()
            ),
        ));
    }
    if plan.input_views().len() != plan.input_count() {
        return Err(crate::Error::invalid_argument(
            ELEMENTWISE_FUSION_OP,
            "input_views",
            format!(
                "plan has {} input views for {} inputs",
                plan.input_views().len(),
                plan.input_count()
            ),
        ));
    }
    if plan.outputs().is_empty() {
        return Ok(false);
    }
    for input in inputs {
        if input.dtype() != plan.dtype() {
            return Err(crate::Error::dtype_mismatch(
                ELEMENTWISE_FUSION_OP,
                input.dtype(),
                plan.dtype(),
            ));
        }
    }
    Ok(true)
}

fn strided_fused_op(op: ElementwiseFusionOp) -> FusedOp {
    match op {
        ElementwiseFusionOp::Add => FusedOp::Add,
        ElementwiseFusionOp::Multiply => FusedOp::Multiply,
        ElementwiseFusionOp::Negate => FusedOp::Negate,
        ElementwiseFusionOp::Conj => FusedOp::Conj,
        ElementwiseFusionOp::Divide => FusedOp::Divide,
        ElementwiseFusionOp::Abs => FusedOp::Abs,
        ElementwiseFusionOp::Maximum => FusedOp::Maximum,
        ElementwiseFusionOp::Minimum => FusedOp::Minimum,
        ElementwiseFusionOp::Clamp => FusedOp::Clamp,
        ElementwiseFusionOp::Exp => FusedOp::Exp,
        ElementwiseFusionOp::Log => FusedOp::Log,
        ElementwiseFusionOp::Sin => FusedOp::Sin,
        ElementwiseFusionOp::Cos => FusedOp::Cos,
        ElementwiseFusionOp::Tanh => FusedOp::Tanh,
        ElementwiseFusionOp::Sqrt => FusedOp::Sqrt,
        ElementwiseFusionOp::Rsqrt => FusedOp::Rsqrt,
        ElementwiseFusionOp::Pow => FusedOp::Pow,
        ElementwiseFusionOp::Expm1 => FusedOp::Expm1,
        ElementwiseFusionOp::Log1p => FusedOp::Log1p,
        ElementwiseFusionOp::Remainder => {
            unreachable!("remainder must be filtered before CPU elementwise fusion")
        }
    }
}

fn plan_uses_unfused_op(plan: &ElementwiseFusionPlan) -> bool {
    plan.ops()
        .iter()
        .any(|inst| inst.op() == ElementwiseFusionOp::Remainder)
}

fn plan_uses_ordered_op(plan: &ElementwiseFusionPlan) -> bool {
    plan.ops().iter().any(|inst| {
        matches!(
            inst.op(),
            ElementwiseFusionOp::Maximum
                | ElementwiseFusionOp::Minimum
                | ElementwiseFusionOp::Clamp
        )
    })
}

fn should_defer_to_broadcast_multiply_special_case(plan: &ElementwiseFusionPlan) -> bool {
    !plan.input_views().iter().all(|view| view.is_identity())
        && plan.ops().len() == 1
        && plan.outputs() == [plan.input_count()]
        && plan.ops()[0].op() == ElementwiseFusionOp::Multiply
}

fn single_output_strided_fused_plan(plan: &ElementwiseFusionPlan, output: usize) -> FusedPlan {
    FusedPlan {
        input_count: plan.input_count(),
        outputs: vec![output],
        ops: plan
            .ops()
            .iter()
            .map(|inst| FusedInst {
                op: strided_fused_op(inst.op()),
                inputs: inst.inputs().to_vec(),
            })
            .collect(),
    }
}

fn kernel_dtype(dtype: DType) -> KernelDType {
    match dtype {
        DType::F32 => KernelDType::F32,
        DType::F64 => KernelDType::F64,
        DType::I32 => KernelDType::I32,
        DType::I64 => KernelDType::I64,
        DType::Bool => KernelDType::Bool,
        DType::C32 => KernelDType::C32,
        DType::C64 => KernelDType::C64,
    }
}

fn typed_bytes<T>(data: &[T]) -> &[u8] {
    // SAFETY: `data` is an aligned typed slice. The returned byte slice has
    // the same lifetime and exact byte length, and is read-only.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), size_of_val(data)) }
}

struct ErasedFusionInput<'a> {
    data: &'a [u8],
    dims: Vec<usize>,
    strides: Vec<isize>,
}

fn tensor_host_bytes<'a>(op: &'static str, input: &'a Tensor) -> crate::Result<&'a [u8]> {
    macro_rules! bytes {
        ($tensor:expr) => {
            typed_host_data(op, $tensor).map(typed_bytes)
        };
    }

    match input {
        Tensor::F32(tensor) => bytes!(tensor),
        Tensor::F64(tensor) => bytes!(tensor),
        Tensor::I32(tensor) => bytes!(tensor),
        Tensor::I64(tensor) => bytes!(tensor),
        Tensor::Bool(tensor) => bytes!(tensor),
        Tensor::C32(tensor) => bytes!(tensor),
        Tensor::C64(tensor) => bytes!(tensor),
    }
}

fn erased_fusion_input<'a>(
    input: &'a Tensor,
    view: &ElementwiseFusionInputView,
) -> crate::Result<ErasedFusionInput<'a>> {
    let data = tensor_host_bytes(ELEMENTWISE_FUSION_OP, input)?;
    let base_shape = input.shape();
    let base_strides = col_major_strides(base_shape)?;
    let ElementwiseFusionInputView::BroadcastInDim { shape, dims } = view else {
        return Ok(ErasedFusionInput {
            data,
            dims: base_shape.to_vec(),
            strides: base_strides,
        });
    };

    if dims.len() != base_shape.len() {
        return Err(crate::Error::invalid_argument(
            ELEMENTWISE_FUSION_OP,
            "configuration",
            format!(
                "broadcast dims length {} does not match input rank {}",
                dims.len(),
                base_shape.len()
            ),
        ));
    }

    let mut strides = vec![0; shape.len()];
    let mut seen = vec![false; shape.len()];
    for (source_axis, &target_axis) in dims.iter().enumerate() {
        if target_axis >= shape.len() {
            return Err(crate::Error::axis_out_of_bounds(
                ELEMENTWISE_FUSION_OP,
                target_axis,
                shape.len(),
            ));
        }
        if seen[target_axis] {
            return Err(crate::Error::duplicate_axis(
                ELEMENTWISE_FUSION_OP,
                target_axis,
                "broadcast dims",
            ));
        }
        seen[target_axis] = true;
        let source_dim = base_shape[source_axis];
        let target_dim = shape[target_axis];
        if source_dim != target_dim && source_dim != 1 {
            return Err(crate::Error::shape_mismatch(
                ELEMENTWISE_FUSION_OP,
                shape.to_vec(),
                base_shape.to_vec(),
            ));
        }
        if source_dim == target_dim {
            strides[target_axis] = base_strides[source_axis];
        }
    }

    Ok(ErasedFusionInput {
        data,
        dims: shape.to_vec(),
        strides,
    })
}

#[doc(hidden)]
pub trait Tier2Elem: Copy + Clone + One + Zero + Send + Sync {
    fn abs_elem(self) -> Self;
    fn sign_elem(self) -> Self;
}

// Keep ordering separate from abs/sign so complex tensors cannot silently pick
// a magnitude ordering. Callers should compute abs/norm explicitly first.
#[doc(hidden)]
pub trait OrderedElem: Copy + Clone + Send + Sync {
    fn max_elem(self, other: Self) -> Self;
    fn min_elem(self, other: Self) -> Self;
}

#[doc(hidden)]
pub trait CompareElem: Copy + Send + Sync {
    fn compare_elem(self, other: Self, dir: &CompareDir) -> bool;
}

trait WrappingIntegerElem:
    Copy
    + PoolScalar
    + TensorScalar
    + Zero
    + PartialEq
    + Eq
    + Send
    + Sync
    + Mul<Output = Self>
    + 'static
{
    fn wrapping_add_elem(self, other: Self) -> Self;
    fn wrapping_sub_elem(self, other: Self) -> Self;
    fn wrapping_mul_elem(self, other: Self) -> Self;
    fn wrapping_div_elem(self, other: Self) -> Self;
    fn wrapping_rem_elem(self, other: Self) -> Self;
    fn wrapping_neg_elem(self) -> Self;
    fn wrapping_abs_elem(self) -> Self;
    fn signum_elem(self) -> Self;
}

macro_rules! impl_tier2_elem_real {
    ($ty:ty) => {
        impl Tier2Elem for $ty {
            fn abs_elem(self) -> Self {
                self.abs()
            }

            fn sign_elem(self) -> Self {
                if self == Self::zero() {
                    Self::zero()
                } else {
                    self.signum()
                }
            }
        }

        impl OrderedElem for $ty {
            fn max_elem(self, other: Self) -> Self {
                if self.is_nan() || other.is_nan() {
                    <$ty>::NAN
                } else if self >= other {
                    self
                } else {
                    other
                }
            }

            fn min_elem(self, other: Self) -> Self {
                if self.is_nan() || other.is_nan() {
                    <$ty>::NAN
                } else if self <= other {
                    self
                } else {
                    other
                }
            }
        }

        impl CompareElem for $ty {
            fn compare_elem(self, other: Self, dir: &CompareDir) -> bool {
                match dir {
                    CompareDir::Eq => self == other,
                    CompareDir::Lt => self < other,
                    CompareDir::Le => self <= other,
                    CompareDir::Gt => self > other,
                    CompareDir::Ge => self >= other,
                }
            }
        }
    };
}

macro_rules! impl_tier2_elem_complex {
    ($real:ty) => {
        impl Tier2Elem for Complex<$real> {
            fn abs_elem(self) -> Self {
                Self::new(self.norm(), <$real>::zero())
            }

            fn sign_elem(self) -> Self {
                if self.is_zero() {
                    Self::zero()
                } else {
                    self / self.abs_elem()
                }
            }
        }

        impl CompareElem for Complex<$real> {
            fn compare_elem(self, other: Self, dir: &CompareDir) -> bool {
                match dir {
                    CompareDir::Eq => self == other,
                    CompareDir::Lt | CompareDir::Le | CompareDir::Gt | CompareDir::Ge => false,
                }
            }
        }
    };
}

impl_tier2_elem_real!(f32);
impl_tier2_elem_real!(f64);
impl_tier2_elem_complex!(f32);
impl_tier2_elem_complex!(f64);

macro_rules! impl_compare_elem_ord {
    ($ty:ty) => {
        impl CompareElem for $ty {
            fn compare_elem(self, other: Self, dir: &CompareDir) -> bool {
                match dir {
                    CompareDir::Eq => self == other,
                    CompareDir::Lt => self < other,
                    CompareDir::Le => self <= other,
                    CompareDir::Gt => self > other,
                    CompareDir::Ge => self >= other,
                }
            }
        }
    };
}

impl_compare_elem_ord!(i32);
impl_compare_elem_ord!(i64);
impl_compare_elem_ord!(bool);

macro_rules! impl_ordered_elem_ord {
    ($ty:ty) => {
        impl OrderedElem for $ty {
            fn max_elem(self, other: Self) -> Self {
                self.max(other)
            }

            fn min_elem(self, other: Self) -> Self {
                self.min(other)
            }
        }
    };
}

impl_ordered_elem_ord!(i32);
impl_ordered_elem_ord!(i64);

macro_rules! impl_wrapping_integer_elem {
    ($ty:ty) => {
        impl WrappingIntegerElem for $ty {
            fn wrapping_add_elem(self, other: Self) -> Self {
                self.wrapping_add(other)
            }

            fn wrapping_sub_elem(self, other: Self) -> Self {
                self.wrapping_sub(other)
            }

            fn wrapping_mul_elem(self, other: Self) -> Self {
                self.wrapping_mul(other)
            }

            fn wrapping_div_elem(self, other: Self) -> Self {
                self.wrapping_div(other)
            }

            fn wrapping_rem_elem(self, other: Self) -> Self {
                self.wrapping_rem(other)
            }

            fn wrapping_neg_elem(self) -> Self {
                self.wrapping_neg()
            }

            fn wrapping_abs_elem(self) -> Self {
                self.wrapping_abs()
            }

            fn signum_elem(self) -> Self {
                self.signum()
            }
        }
    };
}

impl_wrapping_integer_elem!(i32);
impl_wrapping_integer_elem!(i64);

fn strided_view_contains<T>(
    op: &'static str,
    view: &StridedView<'_, T>,
    pred: impl Fn(T) -> bool + Copy + Sync,
) -> crate::Result<bool>
where
    T: Copy + Send + Sync,
{
    reduce(view, pred, |lhs, rhs| lhs || rhs, false)
        .map_err(|err| crate::Error::backend_source(op, err))
}

fn ensure_no_zero_divisor<T>(op: &'static str, rhs: &StridedView<'_, T>) -> crate::Result<()>
where
    T: WrappingIntegerElem,
{
    if strided_view_contains(op, rhs, |value| value == T::zero())? {
        return Err(crate::cpu_division_by_zero(op, T::dtype()));
    }
    Ok(())
}

fn complex_scalar_tensor<T>(scalar: T) -> crate::Result<TypedTensor<Complex<T>>>
where
    T: Copy + Clone + Zero + tenferro_tensor::TensorScalar,
    Complex<T>: tenferro_tensor::TensorScalar,
{
    TypedTensor::from_vec_col_major(vec![], vec![Complex::new(scalar, T::zero())])
}

fn complex_scalar_tensor_from_tensor<T>(
    input: &TypedTensor<T>,
) -> crate::Result<TypedTensor<Complex<T>>>
where
    T: Copy + Clone + Zero + tenferro_tensor::TensorScalar,
    Complex<T>: tenferro_tensor::TensorScalar,
{
    complex_scalar_tensor(typed_host_data("add", input)?[0])
}

fn complex_scalar_tensor_from_view<T, R>(
    input: &TypedTensorView<'_, T, R>,
) -> crate::Result<TypedTensor<Complex<T>>>
where
    T: Copy + Clone + Zero + 'static + tenferro_tensor::TensorScalar,
    Complex<T>: tenferro_tensor::TensorScalar,
    R: TensorRank,
{
    complex_scalar_tensor(typed_view_from_view("add", input)?.get(&[]))
}

#[cfg(test)]
fn with_test_pool<T>(f: impl FnOnce(&mut BufferPool) -> T) -> T {
    let mut buffers = BufferPool::new();
    f(&mut buffers)
}

/// Add two CPU tensors elementwise.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::add;
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
/// let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0])?;
/// let out = add(&a, &b)?;
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[cfg(test)]
#[doc(hidden)]
pub fn add(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
    with_test_pool(|buffers| add_with_pool(buffers, lhs, rhs))
}

#[doc(hidden)]
pub fn add_with_pool(
    buffers: &mut BufferPool,
    lhs: &Tensor,
    rhs: &Tensor,
) -> crate::Result<Tensor> {
    match (lhs, rhs) {
        (Tensor::F32(a), Tensor::F32(b)) => Ok(Tensor::F32(typed_add_with_pool(buffers, a, b)?)),
        (Tensor::F64(a), Tensor::F64(b)) => Ok(Tensor::F64(typed_add_with_pool(buffers, a, b)?)),
        (Tensor::I32(a), Tensor::I32(b)) => {
            Ok(Tensor::I32(typed_wrapping_add_with_pool(buffers, a, b)?))
        }
        (Tensor::I64(a), Tensor::I64(b)) => {
            Ok(Tensor::I64(typed_wrapping_add_with_pool(buffers, a, b)?))
        }
        (Tensor::C32(a), Tensor::C32(b)) => Ok(Tensor::C32(typed_add_with_pool(buffers, a, b)?)),
        (Tensor::C64(a), Tensor::C64(b)) => Ok(Tensor::C64(typed_add_with_pool(buffers, a, b)?)),
        (Tensor::F32(a), Tensor::C32(b)) if a.shape().is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("add", a)?[0])?;
            Ok(Tensor::C32(typed_add_with_pool(buffers, &scalar, b)?))
        }
        (Tensor::C32(a), Tensor::F32(b)) if b.shape().is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("add", b)?[0])?;
            Ok(Tensor::C32(typed_add_with_pool(buffers, a, &scalar)?))
        }
        (Tensor::F64(a), Tensor::C64(b)) if a.shape().is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("add", a)?[0])?;
            Ok(Tensor::C64(typed_add_with_pool(buffers, &scalar, b)?))
        }
        (Tensor::C64(a), Tensor::F64(b)) if b.shape().is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("add", b)?[0])?;
            Ok(Tensor::C64(typed_add_with_pool(buffers, a, &scalar)?))
        }
        _ => Err(tensor_pair_error("add", lhs, rhs)),
    }
}

#[doc(hidden)]
pub fn add_read_with_pool(
    buffers: &mut BufferPool,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    if let (TensorRead::Tensor(lhs), TensorRead::Tensor(rhs)) = (&lhs, &rhs) {
        return add_with_pool(buffers, lhs, rhs);
    }

    macro_rules! dispatch {
        ($variant:ident, $func:ident) => {
            match (&lhs, &rhs) {
                (
                    TensorRead::Tensor(Tensor::$variant(a)),
                    TensorRead::View(TensorView::$variant(b)),
                ) => {
                    let a = a.as_view();
                    return Ok(Tensor::$variant($func(buffers, &a, b)?));
                }
                (
                    TensorRead::View(TensorView::$variant(a)),
                    TensorRead::Tensor(Tensor::$variant(b)),
                ) => {
                    let b = b.as_view();
                    return Ok(Tensor::$variant($func(buffers, a, &b)?));
                }
                (
                    TensorRead::View(TensorView::$variant(a)),
                    TensorRead::View(TensorView::$variant(b)),
                ) => {
                    return Ok(Tensor::$variant($func(buffers, a, b)?));
                }
                _ => {}
            }
        };
    }

    macro_rules! dispatch_real_complex_scalar {
        ($real_variant:ident, $complex_variant:ident) => {
            match (&lhs, &rhs) {
                (
                    TensorRead::Tensor(Tensor::$real_variant(real)),
                    TensorRead::View(TensorView::$complex_variant(complex)),
                ) if real.shape().is_empty() => {
                    let scalar = complex_scalar_tensor_from_tensor(real)?;
                    let scalar = scalar.as_view();
                    return Ok(Tensor::$complex_variant(typed_add_view_with_pool(
                        buffers, &scalar, complex,
                    )?));
                }
                (
                    TensorRead::View(TensorView::$real_variant(real)),
                    TensorRead::Tensor(Tensor::$complex_variant(complex)),
                ) if real.shape().is_empty() => {
                    let scalar = complex_scalar_tensor_from_view(real)?;
                    let scalar = scalar.as_view();
                    let complex = complex.as_view();
                    return Ok(Tensor::$complex_variant(typed_add_view_with_pool(
                        buffers, &scalar, &complex,
                    )?));
                }
                (
                    TensorRead::View(TensorView::$real_variant(real)),
                    TensorRead::View(TensorView::$complex_variant(complex)),
                ) if real.shape().is_empty() => {
                    let scalar = complex_scalar_tensor_from_view(real)?;
                    let scalar = scalar.as_view();
                    return Ok(Tensor::$complex_variant(typed_add_view_with_pool(
                        buffers, &scalar, complex,
                    )?));
                }
                (
                    TensorRead::Tensor(Tensor::$complex_variant(complex)),
                    TensorRead::View(TensorView::$real_variant(real)),
                ) if real.shape().is_empty() => {
                    let complex = complex.as_view();
                    let scalar = complex_scalar_tensor_from_view(real)?;
                    let scalar = scalar.as_view();
                    return Ok(Tensor::$complex_variant(typed_add_view_with_pool(
                        buffers, &complex, &scalar,
                    )?));
                }
                (
                    TensorRead::View(TensorView::$complex_variant(complex)),
                    TensorRead::Tensor(Tensor::$real_variant(real)),
                ) if real.shape().is_empty() => {
                    let scalar = complex_scalar_tensor_from_tensor(real)?;
                    let scalar = scalar.as_view();
                    return Ok(Tensor::$complex_variant(typed_add_view_with_pool(
                        buffers, complex, &scalar,
                    )?));
                }
                (
                    TensorRead::View(TensorView::$complex_variant(complex)),
                    TensorRead::View(TensorView::$real_variant(real)),
                ) if real.shape().is_empty() => {
                    let scalar = complex_scalar_tensor_from_view(real)?;
                    let scalar = scalar.as_view();
                    return Ok(Tensor::$complex_variant(typed_add_view_with_pool(
                        buffers, complex, &scalar,
                    )?));
                }
                _ => {}
            }
        };
    }

    dispatch_real_complex_scalar!(F32, C32);
    dispatch_real_complex_scalar!(F64, C64);

    dispatch!(F32, typed_add_view_with_pool);
    dispatch!(F64, typed_add_view_with_pool);
    dispatch!(I32, typed_wrapping_add_view_with_pool);
    dispatch!(I64, typed_wrapping_add_view_with_pool);
    dispatch!(C32, typed_add_view_with_pool);
    dispatch!(C64, typed_add_view_with_pool);

    Err(read_pair_error("add", lhs, rhs))
}

/// Subtract two CPU tensors elementwise.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::sub;
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![2], vec![5.0_f64, 2.0])?;
/// let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0])?;
/// let out = sub(&a, &b)?;
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, -2.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[cfg(test)]
#[doc(hidden)]
pub fn sub(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
    with_test_pool(|buffers| sub_with_pool(buffers, lhs, rhs))
}

#[doc(hidden)]
pub fn sub_with_pool(
    buffers: &mut BufferPool,
    lhs: &Tensor,
    rhs: &Tensor,
) -> crate::Result<Tensor> {
    match (lhs, rhs) {
        (Tensor::F32(a), Tensor::F32(b)) => Ok(Tensor::F32(typed_sub_with_pool(buffers, a, b)?)),
        (Tensor::F64(a), Tensor::F64(b)) => Ok(Tensor::F64(typed_sub_with_pool(buffers, a, b)?)),
        (Tensor::I32(a), Tensor::I32(b)) => {
            Ok(Tensor::I32(typed_wrapping_sub_with_pool(buffers, a, b)?))
        }
        (Tensor::I64(a), Tensor::I64(b)) => {
            Ok(Tensor::I64(typed_wrapping_sub_with_pool(buffers, a, b)?))
        }
        (Tensor::C32(a), Tensor::C32(b)) => Ok(Tensor::C32(typed_sub_with_pool(buffers, a, b)?)),
        (Tensor::C64(a), Tensor::C64(b)) => Ok(Tensor::C64(typed_sub_with_pool(buffers, a, b)?)),
        (Tensor::F32(a), Tensor::C32(b)) if a.shape().is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("sub", a)?[0])?;
            Ok(Tensor::C32(typed_sub_with_pool(buffers, &scalar, b)?))
        }
        (Tensor::C32(a), Tensor::F32(b)) if b.shape().is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("sub", b)?[0])?;
            Ok(Tensor::C32(typed_sub_with_pool(buffers, a, &scalar)?))
        }
        (Tensor::F64(a), Tensor::C64(b)) if a.shape().is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("sub", a)?[0])?;
            Ok(Tensor::C64(typed_sub_with_pool(buffers, &scalar, b)?))
        }
        (Tensor::C64(a), Tensor::F64(b)) if b.shape().is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("sub", b)?[0])?;
            Ok(Tensor::C64(typed_sub_with_pool(buffers, a, &scalar)?))
        }
        _ => Err(tensor_pair_error("sub", lhs, rhs)),
    }
}

#[doc(hidden)]
pub fn sub_read_with_pool(
    buffers: &mut BufferPool,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    if let (TensorRead::Tensor(lhs), TensorRead::Tensor(rhs)) = (&lhs, &rhs) {
        return sub_with_pool(buffers, lhs, rhs);
    }

    macro_rules! dispatch {
        ($variant:ident, $func:ident) => {
            match (&lhs, &rhs) {
                (
                    TensorRead::Tensor(Tensor::$variant(a)),
                    TensorRead::View(TensorView::$variant(b)),
                ) => {
                    let a = a.as_view();
                    return Ok(Tensor::$variant($func(buffers, &a, b)?));
                }
                (
                    TensorRead::View(TensorView::$variant(a)),
                    TensorRead::Tensor(Tensor::$variant(b)),
                ) => {
                    let b = b.as_view();
                    return Ok(Tensor::$variant($func(buffers, a, &b)?));
                }
                (
                    TensorRead::View(TensorView::$variant(a)),
                    TensorRead::View(TensorView::$variant(b)),
                ) => {
                    return Ok(Tensor::$variant($func(buffers, a, b)?));
                }
                _ => {}
            }
        };
    }

    macro_rules! dispatch_real_complex_scalar {
        ($real_variant:ident, $complex_variant:ident) => {
            match (&lhs, &rhs) {
                (
                    TensorRead::Tensor(Tensor::$real_variant(real)),
                    TensorRead::View(TensorView::$complex_variant(complex)),
                ) if real.shape().is_empty() => {
                    let scalar = complex_scalar_tensor_from_tensor(real)?;
                    let scalar = scalar.as_view();
                    return Ok(Tensor::$complex_variant(typed_sub_view_with_pool(
                        buffers, &scalar, complex,
                    )?));
                }
                (
                    TensorRead::View(TensorView::$real_variant(real)),
                    TensorRead::Tensor(Tensor::$complex_variant(complex)),
                ) if real.shape().is_empty() => {
                    let scalar = complex_scalar_tensor_from_view(real)?;
                    let scalar = scalar.as_view();
                    let complex = complex.as_view();
                    return Ok(Tensor::$complex_variant(typed_sub_view_with_pool(
                        buffers, &scalar, &complex,
                    )?));
                }
                (
                    TensorRead::View(TensorView::$real_variant(real)),
                    TensorRead::View(TensorView::$complex_variant(complex)),
                ) if real.shape().is_empty() => {
                    let scalar = complex_scalar_tensor_from_view(real)?;
                    let scalar = scalar.as_view();
                    return Ok(Tensor::$complex_variant(typed_sub_view_with_pool(
                        buffers, &scalar, complex,
                    )?));
                }
                (
                    TensorRead::Tensor(Tensor::$complex_variant(complex)),
                    TensorRead::View(TensorView::$real_variant(real)),
                ) if real.shape().is_empty() => {
                    let complex = complex.as_view();
                    let scalar = complex_scalar_tensor_from_view(real)?;
                    let scalar = scalar.as_view();
                    return Ok(Tensor::$complex_variant(typed_sub_view_with_pool(
                        buffers, &complex, &scalar,
                    )?));
                }
                (
                    TensorRead::View(TensorView::$complex_variant(complex)),
                    TensorRead::Tensor(Tensor::$real_variant(real)),
                ) if real.shape().is_empty() => {
                    let scalar = complex_scalar_tensor_from_tensor(real)?;
                    let scalar = scalar.as_view();
                    return Ok(Tensor::$complex_variant(typed_sub_view_with_pool(
                        buffers, complex, &scalar,
                    )?));
                }
                (
                    TensorRead::View(TensorView::$complex_variant(complex)),
                    TensorRead::View(TensorView::$real_variant(real)),
                ) if real.shape().is_empty() => {
                    let scalar = complex_scalar_tensor_from_view(real)?;
                    let scalar = scalar.as_view();
                    return Ok(Tensor::$complex_variant(typed_sub_view_with_pool(
                        buffers, complex, &scalar,
                    )?));
                }
                _ => {}
            }
        };
    }

    dispatch_real_complex_scalar!(F32, C32);
    dispatch_real_complex_scalar!(F64, C64);

    dispatch!(F32, typed_sub_view_with_pool);
    dispatch!(F64, typed_sub_view_with_pool);
    dispatch!(I32, typed_wrapping_sub_view_with_pool);
    dispatch!(I64, typed_wrapping_sub_view_with_pool);
    dispatch!(C32, typed_sub_view_with_pool);
    dispatch!(C64, typed_sub_view_with_pool);

    Err(read_pair_error("sub", lhs, rhs))
}

/// Multiply two CPU tensors elementwise.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::mul;
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0])?;
/// let b = Tensor::from_vec_col_major(vec![2], vec![4.0_f64, 5.0])?;
/// let out = mul(&a, &b)?;
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[8.0, 15.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[cfg(test)]
#[doc(hidden)]
pub fn mul(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
    with_test_pool(|buffers| mul_with_pool(buffers, lhs, rhs))
}

fn binary_read_with_pool(
    op: &'static str,
    buffers: &mut BufferPool,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
    f: impl FnOnce(&mut BufferPool, &Tensor, &Tensor) -> crate::Result<Tensor>,
) -> crate::Result<Tensor> {
    if let (Some(lhs), Some(rhs)) = (lhs.as_tensor(), rhs.as_tensor()) {
        return f(buffers, lhs, rhs);
    }

    Err(read_pair_error(op, lhs, rhs))
}

#[doc(hidden)]
pub fn mul_with_pool(
    buffers: &mut BufferPool,
    lhs: &Tensor,
    rhs: &Tensor,
) -> crate::Result<Tensor> {
    match (lhs, rhs) {
        (Tensor::F32(a), Tensor::F32(b)) => Ok(Tensor::F32(typed_mul_with_pool(buffers, a, b)?)),
        (Tensor::F64(a), Tensor::F64(b)) => Ok(Tensor::F64(typed_mul_with_pool(buffers, a, b)?)),
        (Tensor::I32(a), Tensor::I32(b)) => {
            Ok(Tensor::I32(typed_wrapping_mul_with_pool(buffers, a, b)?))
        }
        (Tensor::I64(a), Tensor::I64(b)) => {
            Ok(Tensor::I64(typed_wrapping_mul_with_pool(buffers, a, b)?))
        }
        (Tensor::C32(a), Tensor::C32(b)) => Ok(Tensor::C32(typed_mul_with_pool(buffers, a, b)?)),
        (Tensor::C64(a), Tensor::C64(b)) => Ok(Tensor::C64(typed_mul_with_pool(buffers, a, b)?)),
        (Tensor::F32(a), Tensor::C32(b)) if a.shape().is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("mul", a)?[0])?;
            Ok(Tensor::C32(typed_mul_with_pool(buffers, &scalar, b)?))
        }
        (Tensor::C32(a), Tensor::F32(b)) if b.shape().is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("mul", b)?[0])?;
            Ok(Tensor::C32(typed_mul_with_pool(buffers, a, &scalar)?))
        }
        (Tensor::F64(a), Tensor::C64(b)) if a.shape().is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("mul", a)?[0])?;
            Ok(Tensor::C64(typed_mul_with_pool(buffers, &scalar, b)?))
        }
        (Tensor::C64(a), Tensor::F64(b)) if b.shape().is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("mul", b)?[0])?;
            Ok(Tensor::C64(typed_mul_with_pool(buffers, a, &scalar)?))
        }
        _ => Err(tensor_pair_error("mul", lhs, rhs)),
    }
}

#[doc(hidden)]
pub fn mul_read_with_pool(
    buffers: &mut BufferPool,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    if let (TensorRead::Tensor(lhs), TensorRead::Tensor(rhs)) = (&lhs, &rhs) {
        return mul_with_pool(buffers, lhs, rhs);
    }

    macro_rules! dispatch {
        ($variant:ident, $func:ident) => {
            match (&lhs, &rhs) {
                (
                    TensorRead::Tensor(Tensor::$variant(a)),
                    TensorRead::View(TensorView::$variant(b)),
                ) => {
                    let a = a.as_view();
                    return Ok(Tensor::$variant($func(buffers, &a, b)?));
                }
                (
                    TensorRead::View(TensorView::$variant(a)),
                    TensorRead::Tensor(Tensor::$variant(b)),
                ) => {
                    let b = b.as_view();
                    return Ok(Tensor::$variant($func(buffers, a, &b)?));
                }
                (
                    TensorRead::View(TensorView::$variant(a)),
                    TensorRead::View(TensorView::$variant(b)),
                ) => {
                    return Ok(Tensor::$variant($func(buffers, a, b)?));
                }
                _ => {}
            }
        };
    }

    macro_rules! dispatch_real_complex_scalar {
        ($real_variant:ident, $complex_variant:ident) => {
            match (&lhs, &rhs) {
                (
                    TensorRead::Tensor(Tensor::$real_variant(real)),
                    TensorRead::View(TensorView::$complex_variant(complex)),
                ) if real.shape().is_empty() => {
                    let scalar = complex_scalar_tensor_from_tensor(real)?;
                    let scalar = scalar.as_view();
                    return Ok(Tensor::$complex_variant(typed_mul_view_with_pool(
                        buffers, &scalar, complex,
                    )?));
                }
                (
                    TensorRead::View(TensorView::$real_variant(real)),
                    TensorRead::Tensor(Tensor::$complex_variant(complex)),
                ) if real.shape().is_empty() => {
                    let scalar = complex_scalar_tensor_from_view(real)?;
                    let scalar = scalar.as_view();
                    let complex = complex.as_view();
                    return Ok(Tensor::$complex_variant(typed_mul_view_with_pool(
                        buffers, &scalar, &complex,
                    )?));
                }
                (
                    TensorRead::View(TensorView::$real_variant(real)),
                    TensorRead::View(TensorView::$complex_variant(complex)),
                ) if real.shape().is_empty() => {
                    let scalar = complex_scalar_tensor_from_view(real)?;
                    let scalar = scalar.as_view();
                    return Ok(Tensor::$complex_variant(typed_mul_view_with_pool(
                        buffers, &scalar, complex,
                    )?));
                }
                (
                    TensorRead::Tensor(Tensor::$complex_variant(complex)),
                    TensorRead::View(TensorView::$real_variant(real)),
                ) if real.shape().is_empty() => {
                    let complex = complex.as_view();
                    let scalar = complex_scalar_tensor_from_view(real)?;
                    let scalar = scalar.as_view();
                    return Ok(Tensor::$complex_variant(typed_mul_view_with_pool(
                        buffers, &complex, &scalar,
                    )?));
                }
                (
                    TensorRead::View(TensorView::$complex_variant(complex)),
                    TensorRead::Tensor(Tensor::$real_variant(real)),
                ) if real.shape().is_empty() => {
                    let scalar = complex_scalar_tensor_from_tensor(real)?;
                    let scalar = scalar.as_view();
                    return Ok(Tensor::$complex_variant(typed_mul_view_with_pool(
                        buffers, complex, &scalar,
                    )?));
                }
                (
                    TensorRead::View(TensorView::$complex_variant(complex)),
                    TensorRead::View(TensorView::$real_variant(real)),
                ) if real.shape().is_empty() => {
                    let scalar = complex_scalar_tensor_from_view(real)?;
                    let scalar = scalar.as_view();
                    return Ok(Tensor::$complex_variant(typed_mul_view_with_pool(
                        buffers, complex, &scalar,
                    )?));
                }
                _ => {}
            }
        };
    }

    dispatch_real_complex_scalar!(F32, C32);
    dispatch_real_complex_scalar!(F64, C64);

    dispatch!(F32, typed_mul_view_with_pool);
    dispatch!(F64, typed_mul_view_with_pool);
    dispatch!(I32, typed_wrapping_mul_view_with_pool);
    dispatch!(I64, typed_wrapping_mul_view_with_pool);
    dispatch!(C32, typed_mul_view_with_pool);
    dispatch!(C64, typed_mul_view_with_pool);

    binary_read_with_pool("mul", buffers, lhs, rhs, mul_with_pool)
}

enum CpuReadView<'a> {
    F32(TypedTensorView<'a, f32>),
    F64(TypedTensorView<'a, f64>),
    I32(TypedTensorView<'a, i32>),
    I64(TypedTensorView<'a, i64>),
    Bool(TypedTensorView<'a, bool>),
    C32(TypedTensorView<'a, Complex<f32>>),
    C64(TypedTensorView<'a, Complex<f64>>),
}

fn read_as_cpu_view(input: TensorRead<'_>) -> CpuReadView<'_> {
    match input {
        TensorRead::Tensor(Tensor::F32(tensor)) => CpuReadView::F32(tensor.as_view()),
        TensorRead::Tensor(Tensor::F64(tensor)) => CpuReadView::F64(tensor.as_view()),
        TensorRead::Tensor(Tensor::I32(tensor)) => CpuReadView::I32(tensor.as_view()),
        TensorRead::Tensor(Tensor::I64(tensor)) => CpuReadView::I64(tensor.as_view()),
        TensorRead::Tensor(Tensor::Bool(tensor)) => CpuReadView::Bool(tensor.as_view()),
        TensorRead::Tensor(Tensor::C32(tensor)) => CpuReadView::C32(tensor.as_view()),
        TensorRead::Tensor(Tensor::C64(tensor)) => CpuReadView::C64(tensor.as_view()),
        TensorRead::View(TensorView::F32(view)) => CpuReadView::F32(view),
        TensorRead::View(TensorView::F64(view)) => CpuReadView::F64(view),
        TensorRead::View(TensorView::I32(view)) => CpuReadView::I32(view),
        TensorRead::View(TensorView::I64(view)) => CpuReadView::I64(view),
        TensorRead::View(TensorView::Bool(view)) => CpuReadView::Bool(view),
        TensorRead::View(TensorView::C32(view)) => CpuReadView::C32(view),
        TensorRead::View(TensorView::C64(view)) => CpuReadView::C64(view),
    }
}

#[doc(hidden)]
pub fn elementwise_fusion_with_pool(
    buffers: &mut BufferPool,
    exec_context: &ExecContext,
    inputs: &[&Tensor],
    plan: &ElementwiseFusionPlan,
) -> crate::Result<Option<Vec<Tensor>>> {
    if !validate_elementwise_fusion_inputs(inputs, plan)? {
        return Ok(None);
    }
    if inputs.is_empty() {
        return Ok(None);
    }
    if plan_uses_unfused_op(plan) {
        return Ok(None);
    }
    if should_defer_to_broadcast_multiply_special_case(plan) {
        return Ok(None);
    }
    if !dtype_supports_erased_fusion(plan.dtype(), plan) {
        return Ok(None);
    }

    let input_layouts = inputs
        .iter()
        .zip(plan.input_views())
        .map(|(input, view)| erased_fusion_input(input, view))
        .collect::<crate::Result<Vec<_>>>()?;
    let shape = input_layouts[0].dims.clone();
    if input_layouts
        .iter()
        .skip(1)
        .any(|input| input.dims != shape)
    {
        return Ok(None);
    }
    let element_count =
        tenferro_tensor::validate::checked_shape_product(ELEMENTWISE_FUSION_OP, "shape", &shape)?;
    if element_count < ELEMENTWISE_FUSION_MIN_ELEMENTS {
        return Ok(None);
    }

    let dtype = kernel_dtype(plan.dtype());
    let input_refs = input_layouts
        .iter()
        .map(|input| {
            // SAFETY: fusion inputs are initialized typed storage with matching
            // dtype and alignment; validated layouts bound every reachable read
            // for the retained input borrow.
            unsafe { erased_raw_strided_ref(dtype, input.data, &input.dims, &input.strides, 0) }
                .map_err(|err| crate::Error::backend_source(ELEMENTWISE_FUSION_OP, err))
        })
        .collect::<crate::Result<Vec<_>>>()?;

    execute_erased_fused_outputs(buffers, exec_context, dtype, &input_refs, &shape, plan).map(Some)
}

fn dtype_supports_erased_fusion(dtype: DType, plan: &ElementwiseFusionPlan) -> bool {
    match dtype {
        DType::F32 | DType::F64 => true,
        DType::C32 | DType::C64 => !plan_uses_ordered_op(plan),
        DType::I32 | DType::I64 => plan.ops().iter().all(|inst| {
            matches!(
                inst.op(),
                ElementwiseFusionOp::Add
                    | ElementwiseFusionOp::Multiply
                    | ElementwiseFusionOp::Negate
                    | ElementwiseFusionOp::Conj
                    | ElementwiseFusionOp::Abs
                    | ElementwiseFusionOp::Maximum
                    | ElementwiseFusionOp::Minimum
                    | ElementwiseFusionOp::Clamp
            )
        }),
        DType::Bool => plan
            .ops()
            .iter()
            .all(|inst| inst.op() == ElementwiseFusionOp::Conj),
    }
}

fn execute_erased_fused_outputs(
    buffers: &mut BufferPool,
    exec_context: &ExecContext,
    dtype: KernelDType,
    input_refs: &[ErasedRawStridedRef<'_>],
    shape: &[usize],
    plan: &ElementwiseFusionPlan,
) -> crate::Result<Vec<Tensor>> {
    let input_ptrs: Vec<_> = input_refs
        .iter()
        .map(ErasedRawStridedPtr::from_ref)
        .collect();
    match dtype {
        KernelDType::F32 => plan
            .outputs()
            .iter()
            .map(|&output| {
                execute_erased_fused_output::<f32>(
                    buffers,
                    exec_context,
                    dtype,
                    &input_ptrs,
                    shape,
                    plan,
                    output,
                    Tensor::F32,
                )
            })
            .collect(),
        KernelDType::F64 => plan
            .outputs()
            .iter()
            .map(|&output| {
                execute_erased_fused_output::<f64>(
                    buffers,
                    exec_context,
                    dtype,
                    &input_ptrs,
                    shape,
                    plan,
                    output,
                    Tensor::F64,
                )
            })
            .collect(),
        KernelDType::I32 => plan
            .outputs()
            .iter()
            .map(|&output| {
                execute_erased_fused_output::<i32>(
                    buffers,
                    exec_context,
                    dtype,
                    &input_ptrs,
                    shape,
                    plan,
                    output,
                    Tensor::I32,
                )
            })
            .collect(),
        KernelDType::I64 => plan
            .outputs()
            .iter()
            .map(|&output| {
                execute_erased_fused_output::<i64>(
                    buffers,
                    exec_context,
                    dtype,
                    &input_ptrs,
                    shape,
                    plan,
                    output,
                    Tensor::I64,
                )
            })
            .collect(),
        KernelDType::Bool => plan
            .outputs()
            .iter()
            .map(|&output| {
                execute_erased_fused_output::<bool>(
                    buffers,
                    exec_context,
                    dtype,
                    &input_ptrs,
                    shape,
                    plan,
                    output,
                    Tensor::Bool,
                )
            })
            .collect(),
        KernelDType::C32 => plan
            .outputs()
            .iter()
            .map(|&output| {
                execute_erased_fused_output::<num_complex::Complex32>(
                    buffers,
                    exec_context,
                    dtype,
                    &input_ptrs,
                    shape,
                    plan,
                    output,
                    Tensor::C32,
                )
            })
            .collect(),
        KernelDType::C64 => plan
            .outputs()
            .iter()
            .map(|&output| {
                execute_erased_fused_output::<num_complex::Complex64>(
                    buffers,
                    exec_context,
                    dtype,
                    &input_ptrs,
                    shape,
                    plan,
                    output,
                    Tensor::C64,
                )
            })
            .collect(),
        _ => Err(crate::Error::unsupported(
            ELEMENTWISE_FUSION_OP,
            format!(
                "unsupported dtype {}; supported dtypes: F32/F64/I32/I64/Bool/C32/C64",
                dtype.label()
            ),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_erased_fused_output<T>(
    buffers: &mut BufferPool,
    exec_context: &ExecContext,
    dtype: KernelDType,
    input_ptrs: &[ErasedRawStridedPtr<'_>],
    shape: &[usize],
    plan: &ElementwiseFusionPlan,
    output: usize,
    wrap: fn(TypedTensor<T>) -> Tensor,
) -> crate::Result<Tensor>
where
    T: Clone + PoolScalar,
{
    let fused_plan = single_output_strided_fused_plan(plan, output);
    let erased_plan = ErasedFusedPlan::compile(dtype, fused_plan)
        .map_err(|err| crate::Error::backend_source(ELEMENTWISE_FUSION_OP, err))?;
    let mut out = PooledUninitOutput::<T>::new(buffers, shape.to_vec())?;
    let output_strides = col_major_strides(shape)?;
    // SAFETY: `out` exclusively owns the output allocation with matching
    // dtype/alignment and the fused plan overwrites every reachable element
    // before `assume_init` exposes typed storage.
    let mut dest = unsafe {
        erased_raw_strided_uninit_mut(dtype, out.as_uninit_bytes_mut(), shape, &output_strides, 0)
    }
    .map_err(|err| crate::Error::backend_source(ELEMENTWISE_FUSION_OP, err))?;
    erased_plan
        .execute_uninit(exec_context, &mut dest, input_ptrs)
        .map_err(|err| crate::Error::backend_source(ELEMENTWISE_FUSION_OP, err))?;
    // SAFETY: the fused replay writes every logical destination element and retains no destination view.
    Ok(wrap(unsafe { out.assume_init()? }))
}

fn typed_binary_view_with_pool<T, L, R>(
    op: &'static str,
    buffers: &mut BufferPool,
    lhs: &TypedTensorView<'_, T, L>,
    rhs: &TypedTensorView<'_, T, R>,
    f: impl Fn(T, T) -> T + Copy + Sync,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + PoolScalar + 'static,
    L: TensorRank,
    R: TensorRank,
{
    if lhs.shape() == rhs.shape() {
        let mut out = PooledUninitOutput::<T>::new(buffers, lhs.shape().to_vec())?;
        zip_map2_into(
            &mut out.as_uninit_view_mut()?,
            &typed_view_from_view(op, lhs)?,
            &typed_view_from_view(op, rhs)?,
            |a, b| MaybeUninit::new(f(a, b)),
        )
        .map_err(|err| crate::Error::backend_source(op, err))?;
        // SAFETY: the successful runtime-selected zip/map replay writes every logical destination element and retains no destination view.
        Ok(unsafe { out.assume_init()? })
    } else if lhs.shape().is_empty() {
        let scalar = typed_view_from_view(op, lhs)?.get(&[]);
        let mut out = PooledUninitOutput::<T>::new(buffers, rhs.shape().to_vec())?;
        map_into(
            &mut out.as_uninit_view_mut()?,
            &typed_view_from_view(op, rhs)?,
            |x| MaybeUninit::new(f(scalar, x)),
        )
        .map_err(|err| crate::Error::backend_source(op, err))?;
        // SAFETY: the successful runtime-selected scalar-map replay writes every logical destination element and retains no destination view.
        Ok(unsafe { out.assume_init()? })
    } else if rhs.shape().is_empty() {
        let scalar = typed_view_from_view(op, rhs)?.get(&[]);
        let mut out = PooledUninitOutput::<T>::new(buffers, lhs.shape().to_vec())?;
        map_into(
            &mut out.as_uninit_view_mut()?,
            &typed_view_from_view(op, lhs)?,
            |x| MaybeUninit::new(f(x, scalar)),
        )
        .map_err(|err| crate::Error::backend_source(op, err))?;
        // SAFETY: the successful scalar map replay writes every logical destination element and retains no destination view.
        Ok(unsafe { out.assume_init()? })
    } else {
        Err(crate::Error::shape_mismatch(
            op,
            lhs.shape().to_vec(),
            rhs.shape().to_vec(),
        ))
    }
}

fn typed_unary_view_with_pool<T, R>(
    op: &'static str,
    buffers: &mut BufferPool,
    input: &TypedTensorView<'_, T, R>,
    f: impl Fn(T) -> T + Copy + Sync,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + PoolScalar + 'static,
    R: TensorRank,
{
    let mut out = PooledUninitOutput::<T>::new(buffers, input.shape().to_vec())?;
    map_into(
        &mut out.as_uninit_view_mut()?,
        &typed_view_from_view(op, input)?,
        |x| MaybeUninit::new(f(x)),
    )
    .map_err(|err| crate::Error::backend_source(op, err))?;
    // SAFETY: the successful map replay writes every logical destination element and retains no destination view.
    Ok(unsafe { out.assume_init()? })
}

fn typed_same_shape_binary_view_with_pool<T, O, L, R>(
    op: &'static str,
    buffers: &mut BufferPool,
    lhs: &TypedTensorView<'_, T, L>,
    rhs: &TypedTensorView<'_, T, R>,
    f: impl Fn(T, T) -> O + Copy + Sync,
) -> crate::Result<TypedTensor<O>>
where
    T: Copy + Send + Sync + TensorScalar + 'static,
    O: Copy + PoolScalar,
    L: TensorRank,
    R: TensorRank,
{
    if lhs.shape() != rhs.shape() {
        return Err(crate::Error::shape_mismatch(
            op,
            lhs.shape().to_vec(),
            rhs.shape().to_vec(),
        ));
    }
    let mut out = PooledUninitOutput::<O>::new(buffers, lhs.shape().to_vec())?;
    zip_map2_into(
        &mut out.as_uninit_view_mut()?,
        &typed_view_from_view(op, lhs)?,
        &typed_view_from_view(op, rhs)?,
        |a, b| MaybeUninit::new(f(a, b)),
    )
    .map_err(|err| crate::Error::backend_source(op, err))?;
    // SAFETY: the successful zip replay writes every logical destination element and retains no destination view.
    Ok(unsafe { out.assume_init()? })
}

fn typed_ordered_compare_view_with_pool<T, L, R>(
    buffers: &mut BufferPool,
    lhs: &TypedTensorView<'_, T, L>,
    rhs: &TypedTensorView<'_, T, R>,
    dir: &CompareDir,
) -> crate::Result<TypedTensor<bool>>
where
    T: Copy + Send + Sync + PartialOrd + 'static,
    L: TensorRank,
    R: TensorRank,
{
    if lhs.shape() != rhs.shape() {
        return Err(crate::Error::shape_mismatch(
            "compare",
            lhs.shape().to_vec(),
            rhs.shape().to_vec(),
        ));
    }
    let op = match dir {
        CompareDir::Eq => CompareOp::Eq,
        CompareDir::Lt => CompareOp::Lt,
        CompareDir::Le => CompareOp::Le,
        CompareDir::Gt => CompareOp::Gt,
        CompareDir::Ge => CompareOp::Ge,
    };
    let mut out = PooledUninitOutput::<bool>::new(buffers, lhs.shape().to_vec())?;
    compare_into_uninit(
        &mut out.as_uninit_view_mut()?,
        &typed_view_from_view("compare", lhs)?,
        &typed_view_from_view("compare", rhs)?,
        op,
    )
    .map_err(|err| crate::Error::backend_source("compare", err))?;
    // SAFETY: the successful compare replay writes every logical destination element and retains no destination view.
    Ok(unsafe { out.assume_init()? })
}

fn typed_select_view_with_pool<T, P, A, B>(
    buffers: &mut BufferPool,
    pred: &TypedTensorView<'_, bool, P>,
    on_true: &TypedTensorView<'_, T, A>,
    on_false: &TypedTensorView<'_, T, B>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + PoolScalar + 'static,
    P: TensorRank,
    A: TensorRank,
    B: TensorRank,
{
    if pred.shape() != on_true.shape() {
        return Err(crate::Error::shape_mismatch(
            "select",
            pred.shape().to_vec(),
            on_true.shape().to_vec(),
        ));
    }
    if pred.shape() != on_false.shape() {
        return Err(crate::Error::shape_mismatch(
            "select",
            pred.shape().to_vec(),
            on_false.shape().to_vec(),
        ));
    }
    let mut out = PooledUninitOutput::<T>::new(buffers, pred.shape().to_vec())?;
    zip_map3_into(
        &mut out.as_uninit_view_mut()?,
        &typed_view_from_view("select", pred)?,
        &typed_view_from_view("select", on_true)?,
        &typed_view_from_view("select", on_false)?,
        |p, t, f| MaybeUninit::new(if p { t } else { f }),
    )
    .map_err(|err| crate::Error::backend_source("select", err))?;
    // SAFETY: the successful select replay writes every logical destination element and retains no destination view.
    Ok(unsafe { out.assume_init()? })
}

fn typed_clamp_view_with_pool<T, I, L, U>(
    buffers: &mut BufferPool,
    input: &TypedTensorView<'_, T, I>,
    lower: &TypedTensorView<'_, T, L>,
    upper: &TypedTensorView<'_, T, U>,
) -> crate::Result<TypedTensor<T>>
where
    T: OrderedElem + PoolScalar + 'static,
    I: TensorRank,
    L: TensorRank,
    U: TensorRank,
{
    if input.shape() != lower.shape() {
        return Err(crate::Error::shape_mismatch(
            "clamp",
            input.shape().to_vec(),
            lower.shape().to_vec(),
        ));
    }
    if input.shape() != upper.shape() {
        return Err(crate::Error::shape_mismatch(
            "clamp",
            input.shape().to_vec(),
            upper.shape().to_vec(),
        ));
    }
    let mut out = PooledUninitOutput::<T>::new(buffers, input.shape().to_vec())?;
    zip_map3_into(
        &mut out.as_uninit_view_mut()?,
        &typed_view_from_view("clamp", input)?,
        &typed_view_from_view("clamp", lower)?,
        &typed_view_from_view("clamp", upper)?,
        |x, lo, hi| MaybeUninit::new(hi.min_elem(lo.max_elem(x))),
    )
    .map_err(|err| crate::Error::backend_source("clamp", err))?;
    // SAFETY: the successful clamp replay writes every logical destination element and retains no destination view.
    Ok(unsafe { out.assume_init()? })
}

#[derive(Clone, Copy)]
enum SplitOuterProductLayout {
    LhsPrefix,
    RhsPrefix,
}

struct SplitOuterProductPlan {
    #[allow(dead_code)]
    rows: usize,
    #[allow(dead_code)]
    cols: usize,
    #[allow(dead_code)]
    batches: usize,
    layout: SplitOuterProductLayout,
    lhs_free_axes: Vec<usize>,
    rhs_free_axes: Vec<usize>,
    lhs_batch_axes: Vec<usize>,
    rhs_batch_axes: Vec<usize>,
}

struct OuterProductAxisPartition {
    lhs_free_output_axes: Vec<usize>,
    rhs_free_output_axes: Vec<usize>,
    batch_output_axes: Vec<usize>,
    lhs_free_axes: Vec<usize>,
    rhs_free_axes: Vec<usize>,
    lhs_batch_axes: Vec<usize>,
    rhs_batch_axes: Vec<usize>,
}

fn shape_matches_dims(source_shape: &[usize], output_shape: &[usize], dims: &[usize]) -> bool {
    source_shape.len() == dims.len()
        && source_shape
            .iter()
            .zip(dims.iter())
            .all(|(&dim, &axis)| output_shape.get(axis).copied() == Some(dim))
}

fn axes_by_output(dims: &[usize], output_rank: usize) -> Option<Vec<Option<usize>>> {
    let mut axes = vec![None; output_rank];
    for (src_axis, &dst_axis) in dims.iter().enumerate() {
        let slot = axes.get_mut(dst_axis)?;
        if slot.replace(src_axis).is_some() {
            return None;
        }
    }
    Some(axes)
}

fn axes_shape_product<T>(
    op: &'static str,
    view: &TypedTensorView<'_, T>,
    axes: &[usize],
) -> crate::Result<usize>
where
    T: 'static,
{
    axes.iter().try_fold(1usize, |acc, &axis| {
        acc.checked_mul(view.shape()[axis]).ok_or_else(|| {
            crate::Error::invalid_argument(op, "shape", "shape size overflows usize")
        })
    })
}

fn classify_outer_product_axes(
    lhs_dims: &[usize],
    rhs_dims: &[usize],
    output_rank: usize,
) -> Option<OuterProductAxisPartition> {
    let lhs_axes_by_output = axes_by_output(lhs_dims, output_rank)?;
    let rhs_axes_by_output = axes_by_output(rhs_dims, output_rank)?;

    let mut lhs_free_output_axes = Vec::new();
    let mut rhs_free_output_axes = Vec::new();
    let mut batch_output_axes = Vec::new();
    let mut lhs_free_axes = Vec::new();
    let mut rhs_free_axes = Vec::new();
    let mut lhs_batch_axes = Vec::new();
    let mut rhs_batch_axes = Vec::new();

    for output_axis in 0..output_rank {
        match (
            lhs_axes_by_output[output_axis],
            rhs_axes_by_output[output_axis],
        ) {
            (Some(lhs_axis), Some(rhs_axis)) => {
                batch_output_axes.push(output_axis);
                lhs_batch_axes.push(lhs_axis);
                rhs_batch_axes.push(rhs_axis);
            }
            (Some(lhs_axis), None) => {
                lhs_free_output_axes.push(output_axis);
                lhs_free_axes.push(lhs_axis);
            }
            (None, Some(rhs_axis)) => {
                rhs_free_output_axes.push(output_axis);
                rhs_free_axes.push(rhs_axis);
            }
            (None, None) => return None,
        }
    }

    Some(OuterProductAxisPartition {
        lhs_free_output_axes,
        rhs_free_output_axes,
        batch_output_axes,
        lhs_free_axes,
        rhs_free_axes,
        lhs_batch_axes,
        rhs_batch_axes,
    })
}

fn output_axes_match_partition(output_rank: usize, groups: &[&[usize]]) -> bool {
    groups
        .iter()
        .flat_map(|group| group.iter().copied())
        .eq(0..output_rank)
}

fn split_outer_product_plan<T>(
    lhs: &TypedTensorView<'_, T>,
    lhs_shape: &[usize],
    lhs_dims: &[usize],
    rhs: &TypedTensorView<'_, T>,
    rhs_shape: &[usize],
    rhs_dims: &[usize],
) -> crate::Result<Option<SplitOuterProductPlan>>
where
    T: 'static,
{
    let output_rank = lhs_shape.len();
    if lhs_shape != rhs_shape
        || !shape_matches_dims(lhs.shape(), lhs_shape, lhs_dims)
        || !shape_matches_dims(rhs.shape(), rhs_shape, rhs_dims)
        || lhs.backend_buffer().is_some()
        || rhs.backend_buffer().is_some()
        || lhs.offset() < 0
        || rhs.offset() < 0
        || lhs.strides().iter().any(|&stride| stride < 0)
        || rhs.strides().iter().any(|&stride| stride < 0)
    {
        return Ok(None);
    }

    let Some(partition) = classify_outer_product_axes(lhs_dims, rhs_dims, output_rank) else {
        return Ok(None);
    };

    let lhs_free_size = axes_shape_product("broadcast_multiply", lhs, &partition.lhs_free_axes)?;
    let rhs_free_size = axes_shape_product("broadcast_multiply", rhs, &partition.rhs_free_axes)?;
    if lhs_free_size <= 1 || rhs_free_size <= 1 {
        return Ok(None);
    }
    let batches = axes_shape_product("broadcast_multiply", lhs, &partition.lhs_batch_axes)?;

    let lhs_prefix = output_axes_match_partition(
        output_rank,
        &[
            &partition.lhs_free_output_axes,
            &partition.rhs_free_output_axes,
            &partition.batch_output_axes,
        ],
    );
    if lhs_prefix {
        return Ok(Some(SplitOuterProductPlan {
            rows: lhs_free_size,
            cols: rhs_free_size,
            batches,
            layout: SplitOuterProductLayout::LhsPrefix,
            lhs_free_axes: partition.lhs_free_axes,
            rhs_free_axes: partition.rhs_free_axes,
            lhs_batch_axes: partition.lhs_batch_axes,
            rhs_batch_axes: partition.rhs_batch_axes,
        }));
    }

    let rhs_prefix = output_axes_match_partition(
        output_rank,
        &[
            &partition.rhs_free_output_axes,
            &partition.lhs_free_output_axes,
            &partition.batch_output_axes,
        ],
    );
    if rhs_prefix {
        return Ok(Some(SplitOuterProductPlan {
            rows: rhs_free_size,
            cols: lhs_free_size,
            batches,
            layout: SplitOuterProductLayout::RhsPrefix,
            lhs_free_axes: partition.lhs_free_axes,
            rhs_free_axes: partition.rhs_free_axes,
            lhs_batch_axes: partition.lhs_batch_axes,
            rhs_batch_axes: partition.rhs_batch_axes,
        }));
    }

    Ok(None)
}

fn try_outer_product_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensorView<'_, T>,
    lhs_shape: &[usize],
    lhs_dims: &[usize],
    rhs: &TypedTensorView<'_, T>,
    rhs_shape: &[usize],
    rhs_dims: &[usize],
) -> crate::Result<Option<TypedTensor<T>>>
where
    T: Copy + Clone + Mul<Output = T> + PoolScalar + 'static,
{
    let Some(plan) = split_outer_product_plan(lhs, lhs_shape, lhs_dims, rhs, rhs_shape, rhs_dims)?
    else {
        return Ok(None);
    };

    let mut out = PooledUninitOutput::<T>::new(buffers, lhs_shape.to_vec())?;
    let lhs_view = typed_view_from_view("broadcast_multiply", lhs)?;
    let rhs_view = typed_view_from_view("broadcast_multiply", rhs)?;
    match plan.layout {
        SplitOuterProductLayout::LhsPrefix => {
            let lhs_perm: Vec<_> = plan
                .lhs_free_axes
                .iter()
                .chain(plan.lhs_batch_axes.iter())
                .copied()
                .collect();
            let rhs_perm: Vec<_> = plan
                .rhs_free_axes
                .iter()
                .chain(plan.rhs_batch_axes.iter())
                .copied()
                .collect();
            let lhs_outer = lhs_view
                .permute(&lhs_perm)
                .map_err(|err| crate::Error::backend_source("broadcast_multiply", err))?;
            let rhs_outer = rhs_view
                .permute(&rhs_perm)
                .map_err(|err| crate::Error::backend_source("broadcast_multiply", err))?;
            batched_outer_product_into_uninit(
                &mut out.as_uninit_view_mut()?,
                &lhs_outer,
                &rhs_outer,
                plan.lhs_free_axes.len(),
                plan.rhs_free_axes.len(),
            )
            .map_err(|err| crate::Error::backend_source("broadcast_multiply", err))?;
        }
        SplitOuterProductLayout::RhsPrefix => {
            let lhs_perm: Vec<_> = plan
                .lhs_free_axes
                .iter()
                .chain(plan.lhs_batch_axes.iter())
                .copied()
                .collect();
            let rhs_perm: Vec<_> = plan
                .rhs_free_axes
                .iter()
                .chain(plan.rhs_batch_axes.iter())
                .copied()
                .collect();
            let lhs_outer = lhs_view
                .permute(&lhs_perm)
                .map_err(|err| crate::Error::backend_source("broadcast_multiply", err))?;
            let rhs_outer = rhs_view
                .permute(&rhs_perm)
                .map_err(|err| crate::Error::backend_source("broadcast_multiply", err))?;
            batched_outer_product_into_uninit(
                &mut out.as_uninit_view_mut()?,
                &rhs_outer,
                &lhs_outer,
                plan.rhs_free_axes.len(),
                plan.lhs_free_axes.len(),
            )
            .map_err(|err| crate::Error::backend_source("broadcast_multiply", err))?;
        }
    }
    // SAFETY: the successful outer-product replay writes every logical destination element and retains no destination view.
    Ok(Some(unsafe { out.assume_init()? }))
}

struct LazyOuterProduct<T> {
    base: TypedTensor<T>,
    shape: Vec<usize>,
    strides: Vec<isize>,
}

fn axes_by_physical_stride<T>(view: &TypedTensorView<'_, T>, axes: &[usize]) -> Vec<usize>
where
    T: 'static,
{
    let mut sorted = axes.to_vec();
    sorted.sort_by(|&lhs_axis, &rhs_axis| {
        view.strides()[lhs_axis]
            .cmp(&view.strides()[rhs_axis])
            .then_with(|| lhs_axis.cmp(&rhs_axis))
    });
    sorted
}

fn append_axis_shapes<T>(shape: &mut Vec<usize>, view: &TypedTensorView<'_, T>, axes: &[usize])
where
    T: 'static,
{
    shape.extend(axes.iter().map(|&axis| view.shape()[axis]));
}

fn set_lazy_stride(
    logical_strides: &mut [Option<isize>],
    output_axis: usize,
    stride: isize,
) -> crate::Result<()> {
    let rank = logical_strides.len();
    let slot = logical_strides
        .get_mut(output_axis)
        .ok_or(crate::Error::axis_out_of_bounds(
            "broadcast_multiply",
            output_axis,
            rank,
        ))?;
    if slot.replace(stride).is_some() {
        return Err(crate::Error::duplicate_axis(
            "broadcast_multiply",
            output_axis,
            "lazy output layout",
        ));
    }
    Ok(())
}

struct LazyOuterProductStrideSpec<'a> {
    output_shape: &'a [usize],
    base_shape: &'a [usize],
    leading_axes: &'a [usize],
    leading_dims: &'a [usize],
    trailing_axes: &'a [usize],
    trailing_dims: &'a [usize],
    lhs_batch_axes: &'a [usize],
    rhs_batch_axes: &'a [usize],
    lhs_dims: &'a [usize],
    rhs_dims: &'a [usize],
}

fn lazy_outer_product_strides(spec: LazyOuterProductStrideSpec<'_>) -> crate::Result<Vec<isize>> {
    let base_strides = col_major_strides(spec.base_shape)?;
    let mut logical_strides = vec![None; spec.output_shape.len()];
    let mut base_axis = 0usize;

    for &axis in spec.leading_axes {
        set_lazy_stride(
            &mut logical_strides,
            spec.leading_dims[axis],
            base_strides[base_axis],
        )?;
        base_axis += 1;
    }
    for &axis in spec.trailing_axes {
        set_lazy_stride(
            &mut logical_strides,
            spec.trailing_dims[axis],
            base_strides[base_axis],
        )?;
        base_axis += 1;
    }
    for (&lhs_axis, &rhs_axis) in spec.lhs_batch_axes.iter().zip(spec.rhs_batch_axes.iter()) {
        let output_axis = spec.lhs_dims[lhs_axis];
        if spec.rhs_dims[rhs_axis] != output_axis {
            return Err(crate::Error::invalid_argument(
                "broadcast_multiply",
                "dimensions",
                "batch axes disagree while building lazy outer-product layout",
            ));
        }
        set_lazy_stride(&mut logical_strides, output_axis, base_strides[base_axis])?;
        base_axis += 1;
    }

    logical_strides
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            crate::Error::invalid_argument(
                "broadcast_multiply",
                "dimensions",
                "lazy outer-product layout did not cover every output axis",
            )
        })
}

fn lazy_outer_product_value(
    tensor: Tensor,
    shape: Vec<usize>,
    strides: Vec<isize>,
) -> crate::Result<TensorValue> {
    TensorValue::from_parts(tensor, shape, strides, 0)
}

fn try_lazy_outer_product_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensorView<'_, T>,
    lhs_shape: &[usize],
    lhs_dims: &[usize],
    rhs: &TypedTensorView<'_, T>,
    rhs_shape: &[usize],
    rhs_dims: &[usize],
) -> crate::Result<Option<LazyOuterProduct<T>>>
where
    T: Copy + Clone + Mul<Output = T> + PoolScalar + 'static,
{
    let Some(plan) = split_outer_product_plan(lhs, lhs_shape, lhs_dims, rhs, rhs_shape, rhs_dims)?
    else {
        return Ok(None);
    };

    let lhs_free_axes = axes_by_physical_stride(lhs, &plan.lhs_free_axes);
    let rhs_free_axes = axes_by_physical_stride(rhs, &plan.rhs_free_axes);
    if lhs_free_axes == plan.lhs_free_axes && rhs_free_axes == plan.rhs_free_axes {
        return Ok(None);
    }

    let lhs_view = typed_view_from_view("broadcast_multiply", lhs)?;
    let rhs_view = typed_view_from_view("broadcast_multiply", rhs)?;

    match plan.layout {
        SplitOuterProductLayout::LhsPrefix => {
            let lhs_perm: Vec<_> = lhs_free_axes
                .iter()
                .chain(plan.lhs_batch_axes.iter())
                .copied()
                .collect();
            let rhs_perm: Vec<_> = rhs_free_axes
                .iter()
                .chain(plan.rhs_batch_axes.iter())
                .copied()
                .collect();
            let lhs_outer = lhs_view
                .permute(&lhs_perm)
                .map_err(|err| crate::Error::backend_source("broadcast_multiply", err))?;
            let rhs_outer = rhs_view
                .permute(&rhs_perm)
                .map_err(|err| crate::Error::backend_source("broadcast_multiply", err))?;

            let mut base_shape = Vec::with_capacity(lhs_shape.len());
            append_axis_shapes(&mut base_shape, lhs, &lhs_free_axes);
            append_axis_shapes(&mut base_shape, rhs, &rhs_free_axes);
            append_axis_shapes(&mut base_shape, lhs, &plan.lhs_batch_axes);
            let strides = lazy_outer_product_strides(LazyOuterProductStrideSpec {
                output_shape: lhs_shape,
                base_shape: &base_shape,
                leading_axes: &lhs_free_axes,
                leading_dims: lhs_dims,
                trailing_axes: &rhs_free_axes,
                trailing_dims: rhs_dims,
                lhs_batch_axes: &plan.lhs_batch_axes,
                rhs_batch_axes: &plan.rhs_batch_axes,
                lhs_dims,
                rhs_dims,
            })?;

            let mut base = PooledUninitOutput::<T>::new(buffers, base_shape.clone())?;
            batched_outer_product_into_uninit(
                &mut base.as_uninit_view_mut()?,
                &lhs_outer,
                &rhs_outer,
                lhs_free_axes.len(),
                rhs_free_axes.len(),
            )
            .map_err(|err| crate::Error::backend_source("broadcast_multiply", err))?;
            Ok(Some(LazyOuterProduct {
                // SAFETY: the successful outer-product replay writes every logical base element and retains no destination view.
                base: unsafe { base.assume_init()? },
                shape: lhs_shape.to_vec(),
                strides,
            }))
        }
        SplitOuterProductLayout::RhsPrefix => {
            let lhs_perm: Vec<_> = lhs_free_axes
                .iter()
                .chain(plan.lhs_batch_axes.iter())
                .copied()
                .collect();
            let rhs_perm: Vec<_> = rhs_free_axes
                .iter()
                .chain(plan.rhs_batch_axes.iter())
                .copied()
                .collect();
            let lhs_outer = lhs_view
                .permute(&lhs_perm)
                .map_err(|err| crate::Error::backend_source("broadcast_multiply", err))?;
            let rhs_outer = rhs_view
                .permute(&rhs_perm)
                .map_err(|err| crate::Error::backend_source("broadcast_multiply", err))?;

            let mut base_shape = Vec::with_capacity(lhs_shape.len());
            append_axis_shapes(&mut base_shape, rhs, &rhs_free_axes);
            append_axis_shapes(&mut base_shape, lhs, &lhs_free_axes);
            append_axis_shapes(&mut base_shape, lhs, &plan.lhs_batch_axes);
            let strides = lazy_outer_product_strides(LazyOuterProductStrideSpec {
                output_shape: lhs_shape,
                base_shape: &base_shape,
                leading_axes: &rhs_free_axes,
                leading_dims: rhs_dims,
                trailing_axes: &lhs_free_axes,
                trailing_dims: lhs_dims,
                lhs_batch_axes: &plan.lhs_batch_axes,
                rhs_batch_axes: &plan.rhs_batch_axes,
                lhs_dims,
                rhs_dims,
            })?;

            let mut base = PooledUninitOutput::<T>::new(buffers, base_shape.clone())?;
            batched_outer_product_into_uninit(
                &mut base.as_uninit_view_mut()?,
                &rhs_outer,
                &lhs_outer,
                rhs_free_axes.len(),
                lhs_free_axes.len(),
            )
            .map_err(|err| crate::Error::backend_source("broadcast_multiply", err))?;
            Ok(Some(LazyOuterProduct {
                // SAFETY: the successful outer-product replay writes every logical base element and retains no destination view.
                base: unsafe { base.assume_init()? },
                shape: lhs_shape.to_vec(),
                strides,
            }))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn typed_broadcast_mul_view_with_pool<T, L, R>(
    buffers: &mut BufferPool,
    lhs: &TypedTensorView<'_, T, L>,
    lhs_shape: &[usize],
    lhs_dims: &[usize],
    rhs: &TypedTensorView<'_, T, R>,
    rhs_shape: &[usize],
    rhs_dims: &[usize],
    mul: impl Fn(T, T) -> T + Copy + Sync,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + Mul<Output = T> + PoolScalar + 'static,
    L: TensorRank,
    R: TensorRank,
{
    if lhs_shape != rhs_shape {
        return Err(crate::Error::shape_mismatch(
            "broadcast_multiply",
            lhs_shape.to_vec(),
            rhs_shape.to_vec(),
        ));
    }
    let output_rank = lhs_shape.len();
    let lhs_is_scalar = lhs.shape().is_empty() && lhs_dims.is_empty();
    let rhs_is_scalar = rhs.shape().is_empty() && rhs_dims.is_empty();
    let lhs_is_full_output =
        lhs.shape() == lhs_shape && lhs_dims.iter().copied().eq(0..output_rank);
    let rhs_is_full_output =
        rhs.shape() == rhs_shape && rhs_dims.iter().copied().eq(0..output_rank);
    if lhs_is_scalar && rhs_is_scalar {
        let lhs_scalar = typed_view_from_view("broadcast_multiply", lhs)?.get(&[]);
        let rhs_scalar = typed_view_from_view("broadcast_multiply", rhs)?.get(&[]);
        return filled_broadcast_multiply_tensor(buffers, lhs_shape, mul(lhs_scalar, rhs_scalar));
    }
    if lhs_is_scalar && rhs_is_full_output {
        let scalar = typed_view_from_view("broadcast_multiply", lhs)?.get(&[]);
        let mut out = PooledUninitOutput::<T>::new(buffers, lhs_shape.to_vec())?;
        map_into(
            &mut out.as_uninit_view_mut()?,
            &typed_view_from_view("broadcast_multiply", rhs)?,
            |x| MaybeUninit::new(mul(scalar, x)),
        )
        .map_err(|err| crate::Error::backend_source("broadcast_multiply", err))?;
        // SAFETY: the successful broadcast map replay writes every logical destination element and retains no destination view.
        return Ok(unsafe { out.assume_init()? });
    }
    if rhs_is_scalar && lhs_is_full_output {
        let scalar = typed_view_from_view("broadcast_multiply", rhs)?.get(&[]);
        let mut out = PooledUninitOutput::<T>::new(buffers, lhs_shape.to_vec())?;
        map_into(
            &mut out.as_uninit_view_mut()?,
            &typed_view_from_view("broadcast_multiply", lhs)?,
            |x| MaybeUninit::new(mul(x, scalar)),
        )
        .map_err(|err| crate::Error::backend_source("broadcast_multiply", err))?;
        // SAFETY: the successful broadcast map replay writes every logical destination element and retains no destination view.
        return Ok(unsafe { out.assume_init()? });
    }

    let mut out = PooledUninitOutput::<T>::new(buffers, lhs_shape.to_vec())?;
    let lhs_view = typed_view_from_view("broadcast_multiply", lhs)?;
    let rhs_view = typed_view_from_view("broadcast_multiply", rhs)?;
    broadcast_mul_into_uninit(
        &mut out.as_uninit_view_mut()?,
        &lhs_view,
        lhs_dims,
        &rhs_view,
        rhs_dims,
    )
    .map_err(|err| crate::Error::backend_source("broadcast_multiply", err))?;
    // SAFETY: the successful broadcast replay writes every logical destination element and retains no destination view.
    Ok(unsafe { out.assume_init()? })
}

fn filled_broadcast_multiply_tensor<T>(
    buffers: &mut BufferPool,
    shape: &[usize],
    fill: T,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + PoolScalar + 'static,
{
    let mut out = PooledUninitOutput::<T>::new(buffers, shape.to_vec())?;
    for item in out.as_uninit_slice_mut().iter_mut() {
        item.write(fill);
    }
    // SAFETY: the fill replay writes every logical destination element and retains no destination view.
    unsafe { out.assume_init() }
}

#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub fn broadcast_multiply_read_with_pool(
    buffers: &mut BufferPool,
    lhs: TensorRead<'_>,
    lhs_shape: &[usize],
    lhs_dims: &[usize],
    rhs: TensorRead<'_>,
    rhs_shape: &[usize],
    rhs_dims: &[usize],
) -> crate::Result<Option<Tensor>> {
    let lhs = read_as_cpu_view(lhs);
    let rhs = read_as_cpu_view(rhs);

    macro_rules! dispatch {
        ($variant:ident, $lhs:expr, $rhs:expr, $mul:expr) => {{
            if let Some(out) = try_outer_product_with_pool(
                buffers, &$lhs, lhs_shape, lhs_dims, &$rhs, rhs_shape, rhs_dims,
            )? {
                return Ok(Some(Tensor::$variant(out)));
            }
            Ok(Some(Tensor::$variant(typed_broadcast_mul_view_with_pool(
                buffers, &$lhs, lhs_shape, lhs_dims, &$rhs, rhs_shape, rhs_dims, $mul,
            )?)))
        }};
    }

    match (lhs, rhs) {
        (CpuReadView::F32(lhs), CpuReadView::F32(rhs)) => dispatch!(F32, lhs, rhs, |x, y| x * y),
        (CpuReadView::F64(lhs), CpuReadView::F64(rhs)) => dispatch!(F64, lhs, rhs, |x, y| x * y),
        (CpuReadView::I32(lhs), CpuReadView::I32(rhs)) => {
            dispatch!(I32, lhs, rhs, |x, y| x.wrapping_mul(y))
        }
        (CpuReadView::I64(lhs), CpuReadView::I64(rhs)) => {
            dispatch!(I64, lhs, rhs, |x, y| x.wrapping_mul(y))
        }
        (CpuReadView::C32(lhs), CpuReadView::C32(rhs)) => dispatch!(C32, lhs, rhs, |x, y| x * y),
        (CpuReadView::C64(lhs), CpuReadView::C64(rhs)) => dispatch!(C64, lhs, rhs, |x, y| x * y),
        _ => Ok(None),
    }
}

#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub fn broadcast_multiply_value_with_pool(
    buffers: &mut BufferPool,
    lhs: TensorRead<'_>,
    lhs_shape: &[usize],
    lhs_dims: &[usize],
    rhs: TensorRead<'_>,
    rhs_shape: &[usize],
    rhs_dims: &[usize],
) -> crate::Result<Option<TensorValue>> {
    broadcast_multiply_value_with_pool_and_tag(
        buffers,
        lhs,
        lhs_shape,
        lhs_dims,
        rhs,
        rhs_shape,
        rhs_dims,
        |_| {},
    )
}

#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub fn broadcast_multiply_value_with_pool_and_tag(
    buffers: &mut BufferPool,
    lhs: TensorRead<'_>,
    lhs_shape: &[usize],
    lhs_dims: &[usize],
    rhs: TensorRead<'_>,
    rhs_shape: &[usize],
    rhs_dims: &[usize],
    mut tag_output: impl FnMut(&mut Tensor),
) -> crate::Result<Option<TensorValue>> {
    let lhs_view = read_as_cpu_view(lhs.clone());
    let rhs_view = read_as_cpu_view(rhs.clone());

    macro_rules! dispatch_lazy {
        ($variant:ident, $lhs:expr, $rhs:expr) => {{
            if let Some(out) = try_lazy_outer_product_with_pool(
                buffers, &$lhs, lhs_shape, lhs_dims, &$rhs, rhs_shape, rhs_dims,
            )? {
                let mut base = Tensor::$variant(out.base);
                tag_output(&mut base);
                return Ok(Some(lazy_outer_product_value(
                    base,
                    out.shape,
                    out.strides,
                )?));
            }
        }};
    }

    match (lhs_view, rhs_view) {
        (CpuReadView::F32(lhs_view), CpuReadView::F32(rhs_view)) => {
            dispatch_lazy!(F32, lhs_view, rhs_view);
        }
        (CpuReadView::F64(lhs_view), CpuReadView::F64(rhs_view)) => {
            dispatch_lazy!(F64, lhs_view, rhs_view);
        }
        (CpuReadView::I32(lhs_view), CpuReadView::I32(rhs_view)) => {
            dispatch_lazy!(I32, lhs_view, rhs_view);
        }
        (CpuReadView::I64(lhs_view), CpuReadView::I64(rhs_view)) => {
            dispatch_lazy!(I64, lhs_view, rhs_view);
        }
        (CpuReadView::C32(lhs_view), CpuReadView::C32(rhs_view)) => {
            dispatch_lazy!(C32, lhs_view, rhs_view);
        }
        (CpuReadView::C64(lhs_view), CpuReadView::C64(rhs_view)) => {
            dispatch_lazy!(C64, lhs_view, rhs_view);
        }
        _ => {}
    }

    let mut tensor = broadcast_multiply_read_with_pool(
        buffers, lhs, lhs_shape, lhs_dims, rhs, rhs_shape, rhs_dims,
    )?;
    if let Some(tensor) = &mut tensor {
        tag_output(tensor);
    }
    Ok(tensor.map(TensorValue::from_tensor))
}

/// Divide two CPU tensors elementwise.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::div;
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![2], vec![8.0_f64, 15.0])?;
/// let b = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 5.0])?;
/// let out = div(&a, &b)?;
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[4.0, 3.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[cfg(test)]
#[doc(hidden)]
pub fn div(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
    with_test_pool(|buffers| div_with_pool(buffers, lhs, rhs))
}

#[doc(hidden)]
pub fn div_with_pool(
    buffers: &mut BufferPool,
    lhs: &Tensor,
    rhs: &Tensor,
) -> crate::Result<Tensor> {
    match (lhs, rhs) {
        (Tensor::F32(a), Tensor::F32(b)) => Ok(Tensor::F32(typed_div_with_pool(buffers, a, b)?)),
        (Tensor::F64(a), Tensor::F64(b)) => Ok(Tensor::F64(typed_div_with_pool(buffers, a, b)?)),
        (Tensor::I32(a), Tensor::I32(b)) => {
            Ok(Tensor::I32(typed_integer_div_with_pool(buffers, a, b)?))
        }
        (Tensor::I64(a), Tensor::I64(b)) => {
            Ok(Tensor::I64(typed_integer_div_with_pool(buffers, a, b)?))
        }
        (Tensor::C32(a), Tensor::C32(b)) => Ok(Tensor::C32(typed_div_with_pool(buffers, a, b)?)),
        (Tensor::C64(a), Tensor::C64(b)) => Ok(Tensor::C64(typed_div_with_pool(buffers, a, b)?)),
        (Tensor::F32(a), Tensor::C32(b)) if a.shape().is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("div", a)?[0])?;
            Ok(Tensor::C32(typed_div_with_pool(buffers, &scalar, b)?))
        }
        (Tensor::C32(a), Tensor::F32(b)) if b.shape().is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("div", b)?[0])?;
            Ok(Tensor::C32(typed_div_with_pool(buffers, a, &scalar)?))
        }
        (Tensor::F64(a), Tensor::C64(b)) if a.shape().is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("div", a)?[0])?;
            Ok(Tensor::C64(typed_div_with_pool(buffers, &scalar, b)?))
        }
        (Tensor::C64(a), Tensor::F64(b)) if b.shape().is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("div", b)?[0])?;
            Ok(Tensor::C64(typed_div_with_pool(buffers, a, &scalar)?))
        }
        _ => Err(crate::Error::dtype_mismatch(
            "div",
            lhs.dtype(),
            rhs.dtype(),
        )),
    }
}

#[doc(hidden)]
pub fn div_read_with_pool(
    buffers: &mut BufferPool,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let lhs_dtype = lhs.dtype();
    let rhs_dtype = rhs.dtype();
    match (read_as_cpu_view(lhs), read_as_cpu_view(rhs)) {
        (CpuReadView::F32(a), CpuReadView::F32(b)) => Ok(Tensor::F32(typed_binary_view_with_pool(
            "div",
            buffers,
            &a,
            &b,
            |x, y| x / y,
        )?)),
        (CpuReadView::F64(a), CpuReadView::F64(b)) => Ok(Tensor::F64(typed_binary_view_with_pool(
            "div",
            buffers,
            &a,
            &b,
            |x, y| x / y,
        )?)),
        (CpuReadView::I32(a), CpuReadView::I32(b)) => Ok(Tensor::I32(
            typed_integer_div_view_with_pool(buffers, &a, &b)?,
        )),
        (CpuReadView::I64(a), CpuReadView::I64(b)) => Ok(Tensor::I64(
            typed_integer_div_view_with_pool(buffers, &a, &b)?,
        )),
        (CpuReadView::C32(a), CpuReadView::C32(b)) => Ok(Tensor::C32(typed_binary_view_with_pool(
            "div",
            buffers,
            &a,
            &b,
            |x, y| x / y,
        )?)),
        (CpuReadView::C64(a), CpuReadView::C64(b)) => Ok(Tensor::C64(typed_binary_view_with_pool(
            "div",
            buffers,
            &a,
            &b,
            |x, y| x / y,
        )?)),
        (CpuReadView::F32(real), CpuReadView::C32(complex)) if real.shape().is_empty() => {
            let scalar = complex_scalar_tensor_from_view(&real)?;
            let scalar = scalar.as_view();
            Ok(Tensor::C32(typed_binary_view_with_pool(
                "div",
                buffers,
                &scalar,
                &complex,
                |x, y| x / y,
            )?))
        }
        (CpuReadView::C32(complex), CpuReadView::F32(real)) if real.shape().is_empty() => {
            let scalar = complex_scalar_tensor_from_view(&real)?;
            let scalar = scalar.as_view();
            Ok(Tensor::C32(typed_binary_view_with_pool(
                "div",
                buffers,
                &complex,
                &scalar,
                |x, y| x / y,
            )?))
        }
        (CpuReadView::F64(real), CpuReadView::C64(complex)) if real.shape().is_empty() => {
            let scalar = complex_scalar_tensor_from_view(&real)?;
            let scalar = scalar.as_view();
            Ok(Tensor::C64(typed_binary_view_with_pool(
                "div",
                buffers,
                &scalar,
                &complex,
                |x, y| x / y,
            )?))
        }
        (CpuReadView::C64(complex), CpuReadView::F64(real)) if real.shape().is_empty() => {
            let scalar = complex_scalar_tensor_from_view(&real)?;
            let scalar = scalar.as_view();
            Ok(Tensor::C64(typed_binary_view_with_pool(
                "div",
                buffers,
                &complex,
                &scalar,
                |x, y| x / y,
            )?))
        }
        _ => Err(crate::Error::dtype_mismatch("div", lhs_dtype, rhs_dtype)),
    }
}

/// Compute elementwise remainders on CPU tensors.
///
/// Integer remainders use wrapping two's-complement arithmetic for the
/// `MIN % -1` edge and return a structured error on zero divisors.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::rem;
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![2], vec![7_i32, -7])?;
/// let b = Tensor::from_vec_col_major(vec![2], vec![3_i32, 3])?;
/// let out = rem(&a, &b)?;
/// assert_eq!(out.as_slice::<i32>().unwrap(), &[1, -1]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[cfg(test)]
#[doc(hidden)]
pub fn rem(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
    with_test_pool(|buffers| rem_with_pool(buffers, lhs, rhs))
}

#[doc(hidden)]
pub fn rem_with_pool(
    buffers: &mut BufferPool,
    lhs: &Tensor,
    rhs: &Tensor,
) -> crate::Result<Tensor> {
    match (lhs, rhs) {
        (Tensor::F32(a), Tensor::F32(b)) => Ok(Tensor::F32(typed_rem_with_pool(buffers, a, b)?)),
        (Tensor::F64(a), Tensor::F64(b)) => Ok(Tensor::F64(typed_rem_with_pool(buffers, a, b)?)),
        (Tensor::I32(a), Tensor::I32(b)) => {
            Ok(Tensor::I32(typed_integer_rem_with_pool(buffers, a, b)?))
        }
        (Tensor::I64(a), Tensor::I64(b)) => {
            Ok(Tensor::I64(typed_integer_rem_with_pool(buffers, a, b)?))
        }
        _ => Err(tensor_pair_error("rem", lhs, rhs)),
    }
}

#[doc(hidden)]
pub fn rem_read_with_pool(
    buffers: &mut BufferPool,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let lhs_dtype = lhs.dtype();
    let rhs_dtype = rhs.dtype();
    match (read_as_cpu_view(lhs), read_as_cpu_view(rhs)) {
        (CpuReadView::F32(a), CpuReadView::F32(b)) => Ok(Tensor::F32(typed_binary_view_with_pool(
            "rem",
            buffers,
            &a,
            &b,
            |x, y| x % y,
        )?)),
        (CpuReadView::F64(a), CpuReadView::F64(b)) => Ok(Tensor::F64(typed_binary_view_with_pool(
            "rem",
            buffers,
            &a,
            &b,
            |x, y| x % y,
        )?)),
        (CpuReadView::I32(a), CpuReadView::I32(b)) => Ok(Tensor::I32(
            typed_integer_rem_view_with_pool(buffers, &a, &b)?,
        )),
        (CpuReadView::I64(a), CpuReadView::I64(b)) => Ok(Tensor::I64(
            typed_integer_rem_view_with_pool(buffers, &a, &b)?,
        )),
        _ => Err(dtype_pair_error("rem", lhs_dtype, rhs_dtype)),
    }
}

/// Negate a CPU tensor elementwise.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::neg;
/// use tenferro_tensor::Tensor;
///
/// let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0])?;
/// let out = neg(&input)?;
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[-1.0, 2.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[cfg(test)]
#[doc(hidden)]
pub fn neg(input: &Tensor) -> crate::Result<Tensor> {
    with_test_pool(|buffers| neg_with_pool(buffers, input))
}

#[doc(hidden)]
pub fn neg_with_pool(buffers: &mut BufferPool, input: &Tensor) -> crate::Result<Tensor> {
    match input {
        Tensor::F32(t) => Ok(Tensor::F32(typed_neg_with_pool(buffers, t)?)),
        Tensor::F64(t) => Ok(Tensor::F64(typed_neg_with_pool(buffers, t)?)),
        Tensor::I32(t) => Ok(Tensor::I32(typed_wrapping_neg_with_pool(buffers, t)?)),
        Tensor::I64(t) => Ok(Tensor::I64(typed_wrapping_neg_with_pool(buffers, t)?)),
        Tensor::Bool(_) => Err(unary_dtype_error(
            "neg",
            input.dtype(),
            "F32/F64/I32/I64/C32/C64",
            false,
        )),
        Tensor::C32(t) => Ok(Tensor::C32(typed_neg_with_pool(buffers, t)?)),
        Tensor::C64(t) => Ok(Tensor::C64(typed_neg_with_pool(buffers, t)?)),
    }
}

#[doc(hidden)]
pub fn neg_read_with_pool(
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let dtype = input.dtype();
    match read_as_cpu_view(input) {
        CpuReadView::F32(t) => Ok(Tensor::F32(typed_unary_view_with_pool(
            "neg",
            buffers,
            &t,
            |x| -x,
        )?)),
        CpuReadView::F64(t) => Ok(Tensor::F64(typed_unary_view_with_pool(
            "neg",
            buffers,
            &t,
            |x| -x,
        )?)),
        CpuReadView::I32(t) => Ok(Tensor::I32(typed_unary_view_with_pool(
            "neg",
            buffers,
            &t,
            |x| x.wrapping_neg_elem(),
        )?)),
        CpuReadView::I64(t) => Ok(Tensor::I64(typed_unary_view_with_pool(
            "neg",
            buffers,
            &t,
            |x| x.wrapping_neg_elem(),
        )?)),
        CpuReadView::C32(t) => Ok(Tensor::C32(typed_unary_view_with_pool(
            "neg",
            buffers,
            &t,
            |x| -x,
        )?)),
        CpuReadView::C64(t) => Ok(Tensor::C64(typed_unary_view_with_pool(
            "neg",
            buffers,
            &t,
            |x| -x,
        )?)),
        _ => Err(unary_dtype_error(
            "neg",
            dtype,
            "F32/F64/I32/I64/C32/C64",
            false,
        )),
    }
}

/// Conjugate a real or complex CPU tensor elementwise.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use tenferro_cpu::conj;
/// use tenferro_tensor::Tensor;
///
/// let input = Tensor::from_vec_col_major(vec![1], vec![Complex64::new(1.0, 2.0)])?;
/// let out = conj(&input)?;
/// assert_eq!(out.as_slice::<Complex64>().unwrap(), &[Complex64::new(1.0, -2.0)]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[cfg(test)]
#[doc(hidden)]
pub fn conj(input: &Tensor) -> crate::Result<Tensor> {
    with_test_pool(|buffers| conj_with_pool(buffers, input))
}

#[doc(hidden)]
pub fn conj_with_pool(buffers: &mut BufferPool, input: &Tensor) -> crate::Result<Tensor> {
    match input {
        Tensor::F32(t) => Ok(Tensor::F32(typed_conj_with_pool(buffers, t)?)),
        Tensor::F64(t) => Ok(Tensor::F64(typed_conj_with_pool(buffers, t)?)),
        Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) => Err(unary_dtype_error(
            "conj",
            input.dtype(),
            "F32/F64/C32/C64",
            true,
        )),
        Tensor::C32(t) => Ok(Tensor::C32(typed_conj_with_pool(buffers, t)?)),
        Tensor::C64(t) => Ok(Tensor::C64(typed_conj_with_pool(buffers, t)?)),
    }
}

#[doc(hidden)]
pub fn conj_read_with_pool(
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let dtype = input.dtype();
    match read_as_cpu_view(input) {
        CpuReadView::F32(t) => Ok(Tensor::F32(typed_unary_view_with_pool(
            "conj",
            buffers,
            &t,
            |x| x.conj_elem(),
        )?)),
        CpuReadView::F64(t) => Ok(Tensor::F64(typed_unary_view_with_pool(
            "conj",
            buffers,
            &t,
            |x| x.conj_elem(),
        )?)),
        CpuReadView::C32(t) => Ok(Tensor::C32(typed_unary_view_with_pool(
            "conj",
            buffers,
            &t,
            |x| x.conj_elem(),
        )?)),
        CpuReadView::C64(t) => Ok(Tensor::C64(typed_unary_view_with_pool(
            "conj",
            buffers,
            &t,
            |x| x.conj_elem(),
        )?)),
        _ => Err(unary_dtype_error("conj", dtype, "F32/F64/C32/C64", true)),
    }
}

/// Compute elementwise absolute values.
///
/// Complex inputs return real magnitudes (`C32 -> F32`, `C64 -> F64`).
///
/// # Examples
///
/// ```
/// use tenferro_cpu::abs;
/// use tenferro_tensor::Tensor;
///
/// let input = Tensor::from_vec_col_major(vec![2], vec![-3.0_f64, 4.0])?;
/// let out = abs(&input)?;
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[3.0, 4.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[cfg(test)]
#[doc(hidden)]
pub fn abs(input: &Tensor) -> crate::Result<Tensor> {
    with_test_pool(|buffers| abs_with_pool(buffers, input))
}

#[doc(hidden)]
pub fn abs_with_pool(buffers: &mut BufferPool, input: &Tensor) -> crate::Result<Tensor> {
    match input {
        Tensor::F32(t) => Ok(Tensor::F32(typed_abs_with_pool(buffers, t)?)),
        Tensor::F64(t) => Ok(Tensor::F64(typed_abs_with_pool(buffers, t)?)),
        Tensor::I32(t) => Ok(Tensor::I32(typed_wrapping_abs_with_pool(buffers, t)?)),
        Tensor::I64(t) => Ok(Tensor::I64(typed_wrapping_abs_with_pool(buffers, t)?)),
        Tensor::Bool(_) => Err(unary_dtype_error(
            "abs",
            input.dtype(),
            "F32/F64/I32/I64/C32/C64",
            false,
        )),
        Tensor::C32(t) => Ok(Tensor::F32(typed_complex_abs_with_pool(buffers, t)?)),
        Tensor::C64(t) => Ok(Tensor::F64(typed_complex_abs_with_pool(buffers, t)?)),
    }
}

#[doc(hidden)]
pub fn abs_read_with_pool(
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let dtype = input.dtype();
    match read_as_cpu_view(input) {
        CpuReadView::F32(t) => Ok(Tensor::F32(typed_unary_view_with_pool(
            "abs",
            buffers,
            &t,
            |x| x.abs_elem(),
        )?)),
        CpuReadView::F64(t) => Ok(Tensor::F64(typed_unary_view_with_pool(
            "abs",
            buffers,
            &t,
            |x| x.abs_elem(),
        )?)),
        CpuReadView::I32(t) => Ok(Tensor::I32(typed_unary_view_with_pool(
            "abs",
            buffers,
            &t,
            |x| x.wrapping_abs_elem(),
        )?)),
        CpuReadView::I64(t) => Ok(Tensor::I64(typed_unary_view_with_pool(
            "abs",
            buffers,
            &t,
            |x| x.wrapping_abs_elem(),
        )?)),
        CpuReadView::C32(t) => Ok(Tensor::F32(typed_complex_abs_view_with_pool(buffers, &t)?)),
        CpuReadView::C64(t) => Ok(Tensor::F64(typed_complex_abs_view_with_pool(buffers, &t)?)),
        _ => Err(unary_dtype_error(
            "abs",
            dtype,
            "F32/F64/I32/I64/C32/C64",
            false,
        )),
    }
}

/// Compute elementwise signs.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::sign;
/// use tenferro_tensor::Tensor;
///
/// let input = Tensor::from_vec_col_major(vec![3], vec![-2.0_f64, 0.0, 3.0])?;
/// let out = sign(&input)?;
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[-1.0, 0.0, 1.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[cfg(test)]
#[doc(hidden)]
pub fn sign(input: &Tensor) -> crate::Result<Tensor> {
    with_test_pool(|buffers| sign_with_pool(buffers, input))
}

#[doc(hidden)]
pub fn sign_with_pool(buffers: &mut BufferPool, input: &Tensor) -> crate::Result<Tensor> {
    match input {
        Tensor::F32(t) => Ok(Tensor::F32(typed_sign_with_pool(buffers, t)?)),
        Tensor::F64(t) => Ok(Tensor::F64(typed_sign_with_pool(buffers, t)?)),
        Tensor::I32(t) => Ok(Tensor::I32(typed_integer_sign_with_pool(buffers, t)?)),
        Tensor::I64(t) => Ok(Tensor::I64(typed_integer_sign_with_pool(buffers, t)?)),
        Tensor::Bool(_) => Err(unary_dtype_error(
            "sign",
            input.dtype(),
            "F32/F64/I32/I64/C32/C64",
            false,
        )),
        Tensor::C32(t) => Ok(Tensor::C32(typed_sign_with_pool(buffers, t)?)),
        Tensor::C64(t) => Ok(Tensor::C64(typed_sign_with_pool(buffers, t)?)),
    }
}

#[doc(hidden)]
pub fn sign_read_with_pool(
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let dtype = input.dtype();
    match read_as_cpu_view(input) {
        CpuReadView::F32(t) => Ok(Tensor::F32(typed_unary_view_with_pool(
            "sign",
            buffers,
            &t,
            |x| x.sign_elem(),
        )?)),
        CpuReadView::F64(t) => Ok(Tensor::F64(typed_unary_view_with_pool(
            "sign",
            buffers,
            &t,
            |x| x.sign_elem(),
        )?)),
        CpuReadView::I32(t) => Ok(Tensor::I32(typed_unary_view_with_pool(
            "sign",
            buffers,
            &t,
            |x| x.signum_elem(),
        )?)),
        CpuReadView::I64(t) => Ok(Tensor::I64(typed_unary_view_with_pool(
            "sign",
            buffers,
            &t,
            |x| x.signum_elem(),
        )?)),
        CpuReadView::C32(t) => Ok(Tensor::C32(typed_unary_view_with_pool(
            "sign",
            buffers,
            &t,
            |x| x.sign_elem(),
        )?)),
        CpuReadView::C64(t) => Ok(Tensor::C64(typed_unary_view_with_pool(
            "sign",
            buffers,
            &t,
            |x| x.sign_elem(),
        )?)),
        _ => Err(unary_dtype_error(
            "sign",
            dtype,
            "F32/F64/I32/I64/C32/C64",
            false,
        )),
    }
}

/// Compute elementwise maximum values.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::maximum;
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 5.0])?;
/// let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0])?;
/// let out = maximum(&a, &b)?;
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[3.0, 5.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[cfg(test)]
#[doc(hidden)]
pub fn maximum(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
    with_test_pool(|buffers| maximum_with_pool(buffers, lhs, rhs))
}

#[doc(hidden)]
pub fn maximum_with_pool(
    buffers: &mut BufferPool,
    lhs: &Tensor,
    rhs: &Tensor,
) -> crate::Result<Tensor> {
    reject_complex_ordered_dtypes("maximum", &[lhs.dtype(), rhs.dtype()])?;

    match (lhs, rhs) {
        (Tensor::F32(a), Tensor::F32(b)) => {
            Ok(Tensor::F32(typed_maximum_with_pool(buffers, a, b)?))
        }
        (Tensor::F64(a), Tensor::F64(b)) => {
            Ok(Tensor::F64(typed_maximum_with_pool(buffers, a, b)?))
        }
        (Tensor::I32(a), Tensor::I32(b)) => {
            Ok(Tensor::I32(typed_maximum_with_pool(buffers, a, b)?))
        }
        (Tensor::I64(a), Tensor::I64(b)) => {
            Ok(Tensor::I64(typed_maximum_with_pool(buffers, a, b)?))
        }
        _ => Err(tensor_pair_error("maximum", lhs, rhs)),
    }
}

#[doc(hidden)]
pub fn maximum_read_with_pool(
    buffers: &mut BufferPool,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let lhs_dtype = lhs.dtype();
    let rhs_dtype = rhs.dtype();
    reject_complex_ordered_dtypes("maximum", &[lhs_dtype, rhs_dtype])?;

    match (read_as_cpu_view(lhs), read_as_cpu_view(rhs)) {
        (CpuReadView::F32(a), CpuReadView::F32(b)) => Ok(Tensor::F32(
            typed_same_shape_binary_view_with_pool("maximum", buffers, &a, &b, |x, y| {
                x.max_elem(y)
            })?,
        )),
        (CpuReadView::F64(a), CpuReadView::F64(b)) => Ok(Tensor::F64(
            typed_same_shape_binary_view_with_pool("maximum", buffers, &a, &b, |x, y| {
                x.max_elem(y)
            })?,
        )),
        (CpuReadView::I32(a), CpuReadView::I32(b)) => Ok(Tensor::I32(
            typed_same_shape_binary_view_with_pool("maximum", buffers, &a, &b, |x, y| {
                x.max_elem(y)
            })?,
        )),
        (CpuReadView::I64(a), CpuReadView::I64(b)) => Ok(Tensor::I64(
            typed_same_shape_binary_view_with_pool("maximum", buffers, &a, &b, |x, y| {
                x.max_elem(y)
            })?,
        )),
        _ => Err(dtype_pair_error("maximum", lhs_dtype, rhs_dtype)),
    }
}

/// Compute elementwise minimum values.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::minimum;
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 5.0])?;
/// let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0])?;
/// let out = minimum(&a, &b)?;
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[1.0, 4.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[cfg(test)]
#[doc(hidden)]
pub fn minimum(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
    with_test_pool(|buffers| minimum_with_pool(buffers, lhs, rhs))
}

#[doc(hidden)]
pub fn minimum_with_pool(
    buffers: &mut BufferPool,
    lhs: &Tensor,
    rhs: &Tensor,
) -> crate::Result<Tensor> {
    reject_complex_ordered_dtypes("minimum", &[lhs.dtype(), rhs.dtype()])?;

    match (lhs, rhs) {
        (Tensor::F32(a), Tensor::F32(b)) => {
            Ok(Tensor::F32(typed_minimum_with_pool(buffers, a, b)?))
        }
        (Tensor::F64(a), Tensor::F64(b)) => {
            Ok(Tensor::F64(typed_minimum_with_pool(buffers, a, b)?))
        }
        (Tensor::I32(a), Tensor::I32(b)) => {
            Ok(Tensor::I32(typed_minimum_with_pool(buffers, a, b)?))
        }
        (Tensor::I64(a), Tensor::I64(b)) => {
            Ok(Tensor::I64(typed_minimum_with_pool(buffers, a, b)?))
        }
        _ => Err(tensor_pair_error("minimum", lhs, rhs)),
    }
}

#[doc(hidden)]
pub fn minimum_read_with_pool(
    buffers: &mut BufferPool,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let lhs_dtype = lhs.dtype();
    let rhs_dtype = rhs.dtype();
    reject_complex_ordered_dtypes("minimum", &[lhs_dtype, rhs_dtype])?;

    match (read_as_cpu_view(lhs), read_as_cpu_view(rhs)) {
        (CpuReadView::F32(a), CpuReadView::F32(b)) => Ok(Tensor::F32(
            typed_same_shape_binary_view_with_pool("minimum", buffers, &a, &b, |x, y| {
                x.min_elem(y)
            })?,
        )),
        (CpuReadView::F64(a), CpuReadView::F64(b)) => Ok(Tensor::F64(
            typed_same_shape_binary_view_with_pool("minimum", buffers, &a, &b, |x, y| {
                x.min_elem(y)
            })?,
        )),
        (CpuReadView::I32(a), CpuReadView::I32(b)) => Ok(Tensor::I32(
            typed_same_shape_binary_view_with_pool("minimum", buffers, &a, &b, |x, y| {
                x.min_elem(y)
            })?,
        )),
        (CpuReadView::I64(a), CpuReadView::I64(b)) => Ok(Tensor::I64(
            typed_same_shape_binary_view_with_pool("minimum", buffers, &a, &b, |x, y| {
                x.min_elem(y)
            })?,
        )),
        _ => Err(dtype_pair_error("minimum", lhs_dtype, rhs_dtype)),
    }
}

/// Compare two CPU tensors elementwise.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::compare;
/// use tenferro_tensor::{CompareDir, Tensor};
///
/// let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 5.0])?;
/// let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0])?;
/// let out = compare(&a, &b, &CompareDir::Gt)?;
/// assert_eq!(out.as_slice::<bool>().unwrap(), &[false, true]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[cfg(test)]
#[doc(hidden)]
pub fn compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> crate::Result<Tensor> {
    with_test_pool(|buffers| compare_with_pool(buffers, lhs, rhs, dir))
}

#[doc(hidden)]
pub fn compare_with_pool(
    buffers: &mut BufferPool,
    lhs: &Tensor,
    rhs: &Tensor,
    dir: &CompareDir,
) -> crate::Result<Tensor> {
    reject_complex_unsupported_compare_dtypes(dir, &[lhs.dtype(), rhs.dtype()])?;

    match (lhs, rhs) {
        (Tensor::F32(a), Tensor::F32(b)) => Ok(Tensor::Bool(typed_ordered_compare_view_with_pool(
            buffers,
            &a.as_view(),
            &b.as_view(),
            dir,
        )?)),
        (Tensor::F64(a), Tensor::F64(b)) => Ok(Tensor::Bool(typed_ordered_compare_view_with_pool(
            buffers,
            &a.as_view(),
            &b.as_view(),
            dir,
        )?)),
        (Tensor::I32(a), Tensor::I32(b)) => Ok(Tensor::Bool(typed_ordered_compare_view_with_pool(
            buffers,
            &a.as_view(),
            &b.as_view(),
            dir,
        )?)),
        (Tensor::I64(a), Tensor::I64(b)) => Ok(Tensor::Bool(typed_ordered_compare_view_with_pool(
            buffers,
            &a.as_view(),
            &b.as_view(),
            dir,
        )?)),
        (Tensor::Bool(a), Tensor::Bool(b)) => Ok(Tensor::Bool(
            typed_ordered_compare_view_with_pool(buffers, &a.as_view(), &b.as_view(), dir)?,
        )),
        (Tensor::C32(a), Tensor::C32(b)) => {
            Ok(Tensor::Bool(typed_compare_with_pool(buffers, a, b, dir)?))
        }
        (Tensor::C64(a), Tensor::C64(b)) => {
            Ok(Tensor::Bool(typed_compare_with_pool(buffers, a, b, dir)?))
        }
        _ => Err(crate::Error::dtype_mismatch(
            "compare",
            lhs.dtype(),
            rhs.dtype(),
        )),
    }
}

#[doc(hidden)]
pub fn compare_read_with_pool(
    buffers: &mut BufferPool,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
    dir: &CompareDir,
) -> crate::Result<Tensor> {
    let lhs_dtype = lhs.dtype();
    let rhs_dtype = rhs.dtype();
    reject_complex_unsupported_compare_dtypes(dir, &[lhs_dtype, rhs_dtype])?;

    match (read_as_cpu_view(lhs), read_as_cpu_view(rhs)) {
        (CpuReadView::F32(a), CpuReadView::F32(b)) => Ok(Tensor::Bool(
            typed_ordered_compare_view_with_pool(buffers, &a, &b, dir)?,
        )),
        (CpuReadView::F64(a), CpuReadView::F64(b)) => Ok(Tensor::Bool(
            typed_ordered_compare_view_with_pool(buffers, &a, &b, dir)?,
        )),
        (CpuReadView::I32(a), CpuReadView::I32(b)) => Ok(Tensor::Bool(
            typed_ordered_compare_view_with_pool(buffers, &a, &b, dir)?,
        )),
        (CpuReadView::I64(a), CpuReadView::I64(b)) => Ok(Tensor::Bool(
            typed_ordered_compare_view_with_pool(buffers, &a, &b, dir)?,
        )),
        (CpuReadView::Bool(a), CpuReadView::Bool(b)) => Ok(Tensor::Bool(
            typed_ordered_compare_view_with_pool(buffers, &a, &b, dir)?,
        )),
        (CpuReadView::C32(a), CpuReadView::C32(b)) => Ok(Tensor::Bool(
            typed_same_shape_binary_view_with_pool("compare", buffers, &a, &b, |x, y| {
                x.compare_elem(y, dir)
            })?,
        )),
        (CpuReadView::C64(a), CpuReadView::C64(b)) => Ok(Tensor::Bool(
            typed_same_shape_binary_view_with_pool("compare", buffers, &a, &b, |x, y| {
                x.compare_elem(y, dir)
            })?,
        )),
        _ => Err(crate::Error::dtype_mismatch(
            "compare", lhs_dtype, rhs_dtype,
        )),
    }
}

/// Select values from two tensors using a boolean predicate tensor.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::select;
/// use tenferro_tensor::Tensor;
///
/// let pred = Tensor::from_vec_col_major(vec![2], vec![true, false])?;
/// let on_true = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
/// let on_false = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0])?;
/// let out = select(&pred, &on_true, &on_false)?;
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[1.0, 4.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[cfg(test)]
#[doc(hidden)]
pub fn select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> crate::Result<Tensor> {
    with_test_pool(|buffers| select_with_pool(buffers, pred, on_true, on_false))
}

#[doc(hidden)]
pub fn select_with_pool(
    buffers: &mut BufferPool,
    pred: &Tensor,
    on_true: &Tensor,
    on_false: &Tensor,
) -> crate::Result<Tensor> {
    match (pred, on_true, on_false) {
        (Tensor::Bool(p), Tensor::F32(t), Tensor::F32(f)) => {
            Ok(Tensor::F32(typed_select_with_pool(buffers, p, t, f)?))
        }
        (Tensor::Bool(p), Tensor::F64(t), Tensor::F64(f)) => {
            Ok(Tensor::F64(typed_select_with_pool(buffers, p, t, f)?))
        }
        (Tensor::Bool(p), Tensor::I32(t), Tensor::I32(f)) => {
            Ok(Tensor::I32(typed_select_with_pool(buffers, p, t, f)?))
        }
        (Tensor::Bool(p), Tensor::I64(t), Tensor::I64(f)) => {
            Ok(Tensor::I64(typed_select_with_pool(buffers, p, t, f)?))
        }
        (Tensor::Bool(p), Tensor::Bool(t), Tensor::Bool(f)) => {
            Ok(Tensor::Bool(typed_select_with_pool(buffers, p, t, f)?))
        }
        (Tensor::Bool(p), Tensor::C32(t), Tensor::C32(f)) => {
            Ok(Tensor::C32(typed_select_with_pool(buffers, p, t, f)?))
        }
        (Tensor::Bool(p), Tensor::C64(t), Tensor::C64(f)) => {
            Ok(Tensor::C64(typed_select_with_pool(buffers, p, t, f)?))
        }
        (Tensor::Bool(_), _, _) => Err(crate::Error::dtype_mismatch(
            "select",
            on_true.dtype(),
            on_false.dtype(),
        )),
        _ => Err(crate::Error::dtype_mismatch(
            "select",
            pred.dtype(),
            crate::DType::Bool,
        )),
    }
}

#[doc(hidden)]
pub fn select_read_with_pool(
    buffers: &mut BufferPool,
    pred: TensorRead<'_>,
    on_true: TensorRead<'_>,
    on_false: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let pred_dtype = pred.dtype();
    let true_dtype = on_true.dtype();
    let false_dtype = on_false.dtype();
    match (
        read_as_cpu_view(pred),
        read_as_cpu_view(on_true),
        read_as_cpu_view(on_false),
    ) {
        (CpuReadView::Bool(p), CpuReadView::F32(t), CpuReadView::F32(f)) => Ok(Tensor::F32(
            typed_select_view_with_pool(buffers, &p, &t, &f)?,
        )),
        (CpuReadView::Bool(p), CpuReadView::F64(t), CpuReadView::F64(f)) => Ok(Tensor::F64(
            typed_select_view_with_pool(buffers, &p, &t, &f)?,
        )),
        (CpuReadView::Bool(p), CpuReadView::I32(t), CpuReadView::I32(f)) => Ok(Tensor::I32(
            typed_select_view_with_pool(buffers, &p, &t, &f)?,
        )),
        (CpuReadView::Bool(p), CpuReadView::I64(t), CpuReadView::I64(f)) => Ok(Tensor::I64(
            typed_select_view_with_pool(buffers, &p, &t, &f)?,
        )),
        (CpuReadView::Bool(p), CpuReadView::Bool(t), CpuReadView::Bool(f)) => Ok(Tensor::Bool(
            typed_select_view_with_pool(buffers, &p, &t, &f)?,
        )),
        (CpuReadView::Bool(p), CpuReadView::C32(t), CpuReadView::C32(f)) => Ok(Tensor::C32(
            typed_select_view_with_pool(buffers, &p, &t, &f)?,
        )),
        (CpuReadView::Bool(p), CpuReadView::C64(t), CpuReadView::C64(f)) => Ok(Tensor::C64(
            typed_select_view_with_pool(buffers, &p, &t, &f)?,
        )),
        (CpuReadView::Bool(_), _, _) => Err(crate::Error::dtype_mismatch(
            "select",
            true_dtype,
            false_dtype,
        )),
        _ => Err(crate::Error::dtype_mismatch(
            "select",
            pred_dtype,
            crate::DType::Bool,
        )),
    }
}

/// Clamp CPU tensor values elementwise between lower and upper bounds.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::clamp;
/// use tenferro_tensor::Tensor;
///
/// let input = Tensor::from_vec_col_major(vec![3], vec![-1.0_f64, 2.0, 8.0])?;
/// let lower = Tensor::from_vec_col_major(vec![3], vec![0.0_f64, 0.0, 0.0])?;
/// let upper = Tensor::from_vec_col_major(vec![3], vec![5.0_f64, 5.0, 5.0])?;
/// let out = clamp(&input, &lower, &upper)?;
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[0.0, 2.0, 5.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[cfg(test)]
#[doc(hidden)]
pub fn clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) -> crate::Result<Tensor> {
    with_test_pool(|buffers| clamp_with_pool(buffers, input, lower, upper))
}

#[doc(hidden)]
pub fn clamp_with_pool(
    buffers: &mut BufferPool,
    input: &Tensor,
    lower: &Tensor,
    upper: &Tensor,
) -> crate::Result<Tensor> {
    reject_complex_ordered_dtypes("clamp", &[input.dtype(), lower.dtype(), upper.dtype()])?;

    dispatch_ternary_result_with_pool!("clamp", input, lower, upper, |x, lo, hi| {
        typed_clamp_with_pool(buffers, x, lo, hi)
    })
}

#[doc(hidden)]
pub fn clamp_read_with_pool(
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
    lower: TensorRead<'_>,
    upper: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let input_dtype = input.dtype();
    let lower_dtype = lower.dtype();
    let upper_dtype = upper.dtype();
    reject_complex_ordered_dtypes("clamp", &[input_dtype, lower_dtype, upper_dtype])?;

    match (
        read_as_cpu_view(input),
        read_as_cpu_view(lower),
        read_as_cpu_view(upper),
    ) {
        (CpuReadView::F32(input), CpuReadView::F32(lower), CpuReadView::F32(upper)) => Ok(
            Tensor::F32(typed_clamp_view_with_pool(buffers, &input, &lower, &upper)?),
        ),
        (CpuReadView::F64(input), CpuReadView::F64(lower), CpuReadView::F64(upper)) => Ok(
            Tensor::F64(typed_clamp_view_with_pool(buffers, &input, &lower, &upper)?),
        ),
        _ => Err(crate::Error::dtype_mismatch(
            "clamp",
            input_dtype,
            lower_dtype,
        )),
    }
}

#[doc(hidden)]
pub fn typed_add_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + Add<Output = T> + PoolScalar,
{
    if lhs.shape() == rhs.shape() {
        let mut out = PooledUninitOutput::<T>::new(buffers, lhs.shape().to_vec())?;
        zip_map2_into(
            &mut out.as_uninit_view_mut()?,
            &typed_view("add", lhs)?,
            &typed_view("add", rhs)?,
            |x, y| MaybeUninit::new(x + y),
        )
        .map_err(|err| crate::Error::backend_source("add", err))?;
        // SAFETY: the successful add zip/map replay writes every logical destination element and retains no destination view.
        Ok(unsafe { out.assume_init()? })
    } else if lhs.shape().is_empty() {
        let scalar = typed_host_data("add", lhs)?[0];
        let mut out = PooledUninitOutput::<T>::new(buffers, rhs.shape().to_vec())?;
        map_into(
            &mut out.as_uninit_view_mut()?,
            &typed_view("add", rhs)?,
            |x| MaybeUninit::new(scalar + x),
        )
        .map_err(|err| crate::Error::backend_source("add", err))?;
        // SAFETY: the successful add scalar map replay writes every logical destination element and retains no destination view.
        Ok(unsafe { out.assume_init()? })
    } else if rhs.shape().is_empty() {
        let scalar = typed_host_data("add", rhs)?[0];
        let mut out = PooledUninitOutput::<T>::new(buffers, lhs.shape().to_vec())?;
        map_into(
            &mut out.as_uninit_view_mut()?,
            &typed_view("add", lhs)?,
            |x| MaybeUninit::new(x + scalar),
        )
        .map_err(|err| crate::Error::backend_source("add", err))?;
        // SAFETY: the successful add scalar map replay writes every logical destination element and retains no destination view.
        Ok(unsafe { out.assume_init()? })
    } else {
        Err(crate::Error::shape_mismatch(
            "add",
            lhs.shape().to_vec(),
            rhs.shape().to_vec(),
        ))
    }
}

fn typed_binary_with_pool<T>(
    op: &'static str,
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    f: impl Fn(T, T) -> T + Copy + Sync,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + PoolScalar + 'static,
{
    if lhs.shape() == rhs.shape() {
        let mut out = PooledUninitOutput::<T>::new(buffers, lhs.shape().to_vec())?;
        zip_map2_into(
            &mut out.as_uninit_view_mut()?,
            &typed_view(op, lhs)?,
            &typed_view(op, rhs)?,
            |x, y| MaybeUninit::new(f(x, y)),
        )
        .map_err(|err| crate::Error::backend_source(op, err))?;
        // SAFETY: the successful binary zip/map replay writes every logical destination element and retains no destination view.
        Ok(unsafe { out.assume_init()? })
    } else if lhs.shape().is_empty() {
        let scalar = typed_host_data(op, lhs)?[0];
        let mut out = PooledUninitOutput::<T>::new(buffers, rhs.shape().to_vec())?;
        map_into(&mut out.as_uninit_view_mut()?, &typed_view(op, rhs)?, |x| {
            MaybeUninit::new(f(scalar, x))
        })
        .map_err(|err| crate::Error::backend_source(op, err))?;
        // SAFETY: the successful binary scalar map replay writes every logical destination element and retains no destination view.
        Ok(unsafe { out.assume_init()? })
    } else if rhs.shape().is_empty() {
        let scalar = typed_host_data(op, rhs)?[0];
        let mut out = PooledUninitOutput::<T>::new(buffers, lhs.shape().to_vec())?;
        map_into(&mut out.as_uninit_view_mut()?, &typed_view(op, lhs)?, |x| {
            MaybeUninit::new(f(x, scalar))
        })
        .map_err(|err| crate::Error::backend_source(op, err))?;
        // SAFETY: the successful binary scalar map replay writes every logical destination element and retains no destination view.
        Ok(unsafe { out.assume_init()? })
    } else {
        Err(crate::Error::shape_mismatch(
            op,
            lhs.shape().to_vec(),
            rhs.shape().to_vec(),
        ))
    }
}

fn typed_wrapping_add_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: WrappingIntegerElem + Mul<Output = T>,
{
    typed_binary_with_pool("add", buffers, lhs, rhs, |x, y| x.wrapping_add_elem(y))
}

fn typed_wrapping_add_view_with_pool<T, L, R>(
    buffers: &mut BufferPool,
    lhs: &TypedTensorView<'_, T, L>,
    rhs: &TypedTensorView<'_, T, R>,
) -> crate::Result<TypedTensor<T>>
where
    T: WrappingIntegerElem + Mul<Output = T>,
    L: TensorRank,
    R: TensorRank,
{
    typed_binary_view_with_pool("add", buffers, lhs, rhs, |x, y| x.wrapping_add_elem(y))
}

#[doc(hidden)]
pub fn typed_add_view_with_pool<T, L, R>(
    buffers: &mut BufferPool,
    lhs: &TypedTensorView<'_, T, L>,
    rhs: &TypedTensorView<'_, T, R>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + Add<Output = T> + PoolScalar + 'static,
    L: TensorRank,
    R: TensorRank,
{
    if lhs.shape() == rhs.shape() {
        let mut out = PooledUninitOutput::<T>::new(buffers, lhs.shape().to_vec())?;
        zip_map2_into(
            &mut out.as_uninit_view_mut()?,
            &typed_view_from_view("add", lhs)?,
            &typed_view_from_view("add", rhs)?,
            |x, y| MaybeUninit::new(x + y),
        )
        .map_err(|err| crate::Error::backend_source("add", err))?;
        // SAFETY: the successful add zip/map replay writes every logical destination element and retains no destination view.
        Ok(unsafe { out.assume_init()? })
    } else if lhs.shape().is_empty() {
        let scalar = typed_view_from_view("add", lhs)?.get(&[]);
        let mut out = PooledUninitOutput::<T>::new(buffers, rhs.shape().to_vec())?;
        map_into(
            &mut out.as_uninit_view_mut()?,
            &typed_view_from_view("add", rhs)?,
            |x| MaybeUninit::new(scalar + x),
        )
        .map_err(|err| crate::Error::backend_source("add", err))?;
        // SAFETY: the successful add scalar-map replay writes every logical destination element and retains no destination view.
        Ok(unsafe { out.assume_init()? })
    } else if rhs.shape().is_empty() {
        let scalar = typed_view_from_view("add", rhs)?.get(&[]);
        let mut out = PooledUninitOutput::<T>::new(buffers, lhs.shape().to_vec())?;
        map_into(
            &mut out.as_uninit_view_mut()?,
            &typed_view_from_view("add", lhs)?,
            |x| MaybeUninit::new(x + scalar),
        )
        .map_err(|err| crate::Error::backend_source("add", err))?;
        // SAFETY: the successful add zip/map replay writes every logical destination element and retains no destination view.
        Ok(unsafe { out.assume_init()? })
    } else {
        Err(crate::Error::shape_mismatch(
            "add",
            lhs.shape().to_vec(),
            rhs.shape().to_vec(),
        ))
    }
}

#[doc(hidden)]
pub fn typed_sub_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + PoolScalar + Sub<Output = T> + 'static,
{
    typed_binary_with_pool("sub", buffers, lhs, rhs, |x, y| x - y)
}

fn typed_wrapping_sub_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: WrappingIntegerElem,
{
    typed_binary_with_pool("sub", buffers, lhs, rhs, |x, y| x.wrapping_sub_elem(y))
}

fn typed_wrapping_sub_view_with_pool<T, L, R>(
    buffers: &mut BufferPool,
    lhs: &TypedTensorView<'_, T, L>,
    rhs: &TypedTensorView<'_, T, R>,
) -> crate::Result<TypedTensor<T>>
where
    T: WrappingIntegerElem,
    L: TensorRank,
    R: TensorRank,
{
    typed_binary_view_with_pool("sub", buffers, lhs, rhs, |x, y| x.wrapping_sub_elem(y))
}

#[doc(hidden)]
pub fn typed_sub_view_with_pool<T, L, R>(
    buffers: &mut BufferPool,
    lhs: &TypedTensorView<'_, T, L>,
    rhs: &TypedTensorView<'_, T, R>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + PoolScalar + Sub<Output = T> + 'static,
    L: TensorRank,
    R: TensorRank,
{
    typed_binary_view_with_pool("sub", buffers, lhs, rhs, |x, y| x - y)
}

#[doc(hidden)]
pub fn typed_mul_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + Mul<Output = T> + PoolScalar + 'static,
{
    if lhs.shape() == rhs.shape() {
        let mut out = PooledUninitOutput::<T>::new(buffers, lhs.shape().to_vec())?;
        mul_into_uninit(
            &mut out.as_uninit_view_mut()?,
            &typed_view("mul", lhs)?,
            &typed_view("mul", rhs)?,
        )
        .map_err(|err| crate::Error::backend_source("mul", err))?;
        // SAFETY: the successful multiplication kernel writes every logical destination element and retains no destination view.
        Ok(unsafe { out.assume_init()? })
    } else {
        typed_binary_with_pool("mul", buffers, lhs, rhs, |x, y| x * y)
    }
}

fn typed_wrapping_mul_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: WrappingIntegerElem,
{
    if lhs.shape() == rhs.shape() {
        let mut out = PooledUninitOutput::<T>::new(buffers, lhs.shape().to_vec())?;
        mul_into_uninit(
            &mut out.as_uninit_view_mut()?,
            &typed_view("mul", lhs)?,
            &typed_view("mul", rhs)?,
        )
        .map_err(|err| crate::Error::backend_source("mul", err))?;
        // SAFETY: the successful wrapping multiplication kernel writes every logical destination element and retains no destination view.
        Ok(unsafe { out.assume_init()? })
    } else {
        typed_binary_with_pool("mul", buffers, lhs, rhs, |x, y| x.wrapping_mul_elem(y))
    }
}

fn typed_wrapping_mul_view_with_pool<T, L, R>(
    buffers: &mut BufferPool,
    lhs: &TypedTensorView<'_, T, L>,
    rhs: &TypedTensorView<'_, T, R>,
) -> crate::Result<TypedTensor<T>>
where
    T: WrappingIntegerElem,
    L: TensorRank,
    R: TensorRank,
{
    if lhs.shape() == rhs.shape() {
        let mut out = PooledUninitOutput::<T>::new(buffers, lhs.shape().to_vec())?;
        mul_into_uninit(
            &mut out.as_uninit_view_mut()?,
            &typed_view_from_view("mul", lhs)?,
            &typed_view_from_view("mul", rhs)?,
        )
        .map_err(|err| crate::Error::backend_source("mul", err))?;
        // SAFETY: the successful wrapping multiplication kernel writes every logical destination element and retains no destination view.
        Ok(unsafe { out.assume_init()? })
    } else {
        typed_binary_view_with_pool("mul", buffers, lhs, rhs, |x, y| x.wrapping_mul_elem(y))
    }
}

#[doc(hidden)]
pub fn typed_mul_view_with_pool<T, L, R>(
    buffers: &mut BufferPool,
    lhs: &TypedTensorView<'_, T, L>,
    rhs: &TypedTensorView<'_, T, R>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + Mul<Output = T> + PoolScalar + 'static,
    L: TensorRank,
    R: TensorRank,
{
    if lhs.shape() == rhs.shape() {
        let mut out = PooledUninitOutput::<T>::new(buffers, lhs.shape().to_vec())?;
        mul_into_uninit(
            &mut out.as_uninit_view_mut()?,
            &typed_view_from_view("mul", lhs)?,
            &typed_view_from_view("mul", rhs)?,
        )
        .map_err(|err| crate::Error::backend_source("mul", err))?;
        // SAFETY: the successful multiplication kernel writes every logical destination element and retains no destination view.
        Ok(unsafe { out.assume_init()? })
    } else {
        typed_binary_view_with_pool("mul", buffers, lhs, rhs, |x, y| x * y)
    }
}

#[doc(hidden)]
pub fn typed_div_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + Div<Output = T> + PoolScalar + 'static,
{
    typed_binary_with_pool("div", buffers, lhs, rhs, |x, y| x / y)
}

fn typed_integer_div_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: WrappingIntegerElem,
{
    let rhs_view = typed_view("div", rhs)?;
    ensure_no_zero_divisor("div", &rhs_view)?;
    typed_binary_with_pool("div", buffers, lhs, rhs, |x, y| x.wrapping_div_elem(y))
}

fn typed_integer_div_view_with_pool<T, L, R>(
    buffers: &mut BufferPool,
    lhs: &TypedTensorView<'_, T, L>,
    rhs: &TypedTensorView<'_, T, R>,
) -> crate::Result<TypedTensor<T>>
where
    T: WrappingIntegerElem,
    L: TensorRank,
    R: TensorRank,
{
    let rhs_view = typed_view_from_view("div", rhs)?;
    ensure_no_zero_divisor("div", &rhs_view)?;
    typed_binary_view_with_pool("div", buffers, lhs, rhs, |x, y| x.wrapping_div_elem(y))
}

fn typed_rem_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + StdRem<Output = T> + PoolScalar + 'static,
{
    typed_binary_with_pool("rem", buffers, lhs, rhs, |x, y| x % y)
}

fn typed_integer_rem_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: WrappingIntegerElem,
{
    let rhs_view = typed_view("rem", rhs)?;
    ensure_no_zero_divisor("rem", &rhs_view)?;
    typed_binary_with_pool("rem", buffers, lhs, rhs, |x, y| x.wrapping_rem_elem(y))
}

fn typed_integer_rem_view_with_pool<T, L, R>(
    buffers: &mut BufferPool,
    lhs: &TypedTensorView<'_, T, L>,
    rhs: &TypedTensorView<'_, T, R>,
) -> crate::Result<TypedTensor<T>>
where
    T: WrappingIntegerElem,
    L: TensorRank,
    R: TensorRank,
{
    let rhs_view = typed_view_from_view("rem", rhs)?;
    ensure_no_zero_divisor("rem", &rhs_view)?;
    typed_binary_view_with_pool("rem", buffers, lhs, rhs, |x, y| x.wrapping_rem_elem(y))
}

#[doc(hidden)]
fn typed_map_with_pool<T, O>(
    op: &'static str,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
    f: impl Fn(T) -> O + Copy + Sync,
) -> crate::Result<TypedTensor<O>>
where
    T: Copy + Send + Sync + TensorScalar + 'static,
    O: Clone + PoolScalar,
{
    let mut out = PooledUninitOutput::<O>::new(buffers, input.shape().to_vec())?;
    map_into(
        &mut out.as_uninit_view_mut()?,
        &typed_view(op, input)?,
        |x| MaybeUninit::new(f(x)),
    )
    .map_err(|err| crate::Error::backend_source(op, err))?;
    // SAFETY: the successful same-shape replay writes every logical destination element and retains no destination view.
    Ok(unsafe { out.assume_init()? })
}

#[doc(hidden)]
pub fn typed_neg_with_pool<T>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + Neg<Output = T> + PoolScalar + 'static,
{
    typed_map_with_pool("neg", buffers, input, |x| -x)
}

fn typed_wrapping_unary_with_pool<T>(
    op: &'static str,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
    f: impl Fn(T) -> T + Copy + Sync,
) -> crate::Result<TypedTensor<T>>
where
    T: WrappingIntegerElem,
{
    typed_map_with_pool(op, buffers, input, f)
}

fn typed_wrapping_neg_with_pool<T>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: WrappingIntegerElem,
{
    typed_wrapping_unary_with_pool("neg", buffers, input, |x| x.wrapping_neg_elem())
}

fn typed_wrapping_abs_with_pool<T>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: WrappingIntegerElem,
{
    typed_wrapping_unary_with_pool("abs", buffers, input, |x| x.wrapping_abs_elem())
}

#[doc(hidden)]
pub fn typed_conj_with_pool<T>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + ConjElem + PoolScalar + 'static,
{
    typed_map_with_pool("conj", buffers, input, |x| x.conj_elem())
}

#[doc(hidden)]
pub fn typed_abs_with_pool<T>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Tier2Elem + PoolScalar + 'static,
{
    typed_map_with_pool("abs", buffers, input, |x| x.abs_elem())
}

fn typed_complex_abs_with_pool<T>(
    buffers: &mut BufferPool,
    input: &TypedTensor<Complex<T>>,
) -> crate::Result<TypedTensor<T>>
where
    T: num_traits::Float + PoolScalar + 'static,
    Complex<T>: TensorScalar,
{
    typed_map_with_pool("abs", buffers, input, |x| x.norm())
}

fn typed_complex_abs_view_with_pool<T, R>(
    buffers: &mut BufferPool,
    input: &TypedTensorView<'_, Complex<T>, R>,
) -> crate::Result<TypedTensor<T>>
where
    T: num_traits::Float + PoolScalar + 'static,
    R: TensorRank,
{
    let mut out = PooledUninitOutput::<T>::new(buffers, input.shape().to_vec())?;
    map_into(
        &mut out.as_uninit_view_mut()?,
        &typed_view_from_view("abs", input)?,
        |x| MaybeUninit::new(x.norm()),
    )
    .map_err(|err| crate::Error::backend_source("abs", err))?;
    // SAFETY: the successful unary map replay writes every logical destination element and retains no destination view.
    Ok(unsafe { out.assume_init()? })
}

#[doc(hidden)]
pub fn typed_sign_with_pool<T>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Tier2Elem + PoolScalar + 'static,
{
    typed_map_with_pool("sign", buffers, input, |x| x.sign_elem())
}

fn typed_integer_sign_with_pool<T>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: WrappingIntegerElem,
{
    typed_wrapping_unary_with_pool("sign", buffers, input, |x| x.signum_elem())
}

#[doc(hidden)]
pub fn typed_maximum_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: OrderedElem + PoolScalar,
{
    if lhs.shape() != rhs.shape() {
        return Err(crate::Error::shape_mismatch(
            "maximum",
            lhs.shape().to_vec(),
            rhs.shape().to_vec(),
        ));
    }
    let mut out = PooledUninitOutput::<T>::new(buffers, lhs.shape().to_vec())?;
    zip_map2_into(
        &mut out.as_uninit_view_mut()?,
        &typed_view("maximum", lhs)?,
        &typed_view("maximum", rhs)?,
        |x, y| MaybeUninit::new(x.max_elem(y)),
    )
    .map_err(|err| crate::Error::backend_source("maximum", err))?;
    // SAFETY: the successful binary zip replay writes every logical destination element and retains no destination view.
    Ok(unsafe { out.assume_init()? })
}

#[doc(hidden)]
pub fn typed_minimum_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: OrderedElem + PoolScalar,
{
    if lhs.shape() != rhs.shape() {
        return Err(crate::Error::shape_mismatch(
            "minimum",
            lhs.shape().to_vec(),
            rhs.shape().to_vec(),
        ));
    }
    let mut out = PooledUninitOutput::<T>::new(buffers, lhs.shape().to_vec())?;
    zip_map2_into(
        &mut out.as_uninit_view_mut()?,
        &typed_view("minimum", lhs)?,
        &typed_view("minimum", rhs)?,
        |x, y| MaybeUninit::new(x.min_elem(y)),
    )
    .map_err(|err| crate::Error::backend_source("minimum", err))?;
    // SAFETY: the successful binary zip replay writes every logical destination element and retains no destination view.
    Ok(unsafe { out.assume_init()? })
}

#[doc(hidden)]
pub fn typed_compare_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    dir: &CompareDir,
) -> crate::Result<TypedTensor<bool>>
where
    T: CompareElem + TensorScalar,
{
    if lhs.shape() != rhs.shape() {
        return Err(crate::Error::shape_mismatch(
            "compare",
            lhs.shape().to_vec(),
            rhs.shape().to_vec(),
        ));
    }
    let mut out = PooledUninitOutput::<bool>::new(buffers, lhs.shape().to_vec())?;
    zip_map2_into(
        &mut out.as_uninit_view_mut()?,
        &typed_view("compare", lhs)?,
        &typed_view("compare", rhs)?,
        |x, y| MaybeUninit::new(x.compare_elem(y, dir)),
    )
    .map_err(|err| crate::Error::backend_source("compare", err))?;
    // SAFETY: the successful compare replay writes every logical destination element and retains no destination view.
    Ok(unsafe { out.assume_init()? })
}

#[doc(hidden)]
pub fn typed_select_with_pool<T>(
    buffers: &mut BufferPool,
    pred: &TypedTensor<bool>,
    on_true: &TypedTensor<T>,
    on_false: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + PoolScalar,
{
    if pred.shape() != on_true.shape() {
        return Err(crate::Error::shape_mismatch(
            "select",
            pred.shape().to_vec(),
            on_true.shape().to_vec(),
        ));
    }
    if pred.shape() != on_false.shape() {
        return Err(crate::Error::shape_mismatch(
            "select",
            pred.shape().to_vec(),
            on_false.shape().to_vec(),
        ));
    }
    let mut out = PooledUninitOutput::<T>::new(buffers, pred.shape().to_vec())?;
    zip_map3_into(
        &mut out.as_uninit_view_mut()?,
        &typed_view("select", pred)?,
        &typed_view("select", on_true)?,
        &typed_view("select", on_false)?,
        |p, t, f| MaybeUninit::new(if p { t } else { f }),
    )
    .map_err(|err| crate::Error::backend_source("select", err))?;
    // SAFETY: the successful select replay writes every logical destination element and retains no destination view.
    Ok(unsafe { out.assume_init()? })
}

#[doc(hidden)]
pub fn typed_clamp_with_pool<T>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
    lower: &TypedTensor<T>,
    upper: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: OrderedElem + PoolScalar,
{
    if input.shape() != lower.shape() {
        return Err(crate::Error::shape_mismatch(
            "clamp",
            input.shape().to_vec(),
            lower.shape().to_vec(),
        ));
    }
    if input.shape() != upper.shape() {
        return Err(crate::Error::shape_mismatch(
            "clamp",
            input.shape().to_vec(),
            upper.shape().to_vec(),
        ));
    }
    let mut out = PooledUninitOutput::<T>::new(buffers, input.shape().to_vec())?;
    zip_map3_into(
        &mut out.as_uninit_view_mut()?,
        &typed_view("clamp", input)?,
        &typed_view("clamp", lower)?,
        &typed_view("clamp", upper)?,
        |x, lo, hi| MaybeUninit::new(hi.min_elem(lo.max_elem(x))),
    )
    .map_err(|err| crate::Error::backend_source("clamp", err))?;
    // SAFETY: the successful clamp replay writes every logical destination element and retains no destination view.
    Ok(unsafe { out.assume_init()? })
}

#[cfg(test)]
mod tests;
