use num_complex::Complex;
use num_traits::Zero;
use smallvec::SmallVec;
use strided_kernel::{
    erased_broadcast_mul_into_uninit, erased_clamp_into_uninit, erased_compare_into_uninit,
    erased_map_into_uninit, erased_select_into_uninit, erased_zip_into_uninit,
    plan_lazy_outer_product, CompareOp, ErasedMapOp, ErasedRawStridedPtr, ErasedRawStridedRef,
    ErasedRawStridedUninitMut, ErasedZipOp, ExecContext, KernelStorageElement, StridedError,
    StridedView,
};

use tenferro_cpu_basic::{BufferPool, PoolScalar, PooledUninitOutput};
use tenferro_tensor::{
    col_major_strides, CompareDir, DType, Tensor, TensorRank, TensorRead, TensorScalar,
    TensorValue, TensorView, TypedTensor, TypedTensorView,
};

use super::{typed_host_data, typed_view, typed_view_from_view};
use tenferro_cpu_basic::{
    reject_complex_ordered_dtypes, reject_complex_unsupported_compare_dtypes,
};

/// The typed operands behind a same-dtype triple, or this module's refusal for one.
/// The Rust scalar type behind a preset variant name a macro received.
macro_rules! preset_scalar {
    (F32) => {
        f32
    };
    (F64) => {
        f64
    };
    (I32) => {
        i32
    };
    (I64) => {
        i64
    };
    (Bool) => {
        bool
    };
    (C32) => {
        num_complex::Complex32
    };
    (C64) => {
        num_complex::Complex64
    };
}
macro_rules! dispatch_ternary_result_with_pool {
    ($op:literal, $a:expr, $b:expr, $c:expr, |$x:ident, $y:ident, $z:ident| $body:expr) => {{
        match $a.dtype() {
            DType::F32 => {
                let ($x, $y, $z) = ternary_operands::<f32>($op, $a, $b, $c)?;
                Ok(Tensor::from_typed::<f32>($body?))
            }
            DType::F64 => {
                let ($x, $y, $z) = ternary_operands::<f64>($op, $a, $b, $c)?;
                Ok(Tensor::from_typed::<f64>($body?))
            }
            _ => Err(ternary_dtype_error(
                $op,
                [$a.dtype(), $b.dtype(), $c.dtype()],
            )),
        }
    }};
}
fn ternary_operands<'a, T: TensorScalar>(
    op: &'static str,
    a: &'a Tensor,
    b: &'a Tensor,
    c: &'a Tensor,
) -> crate::Result<(&'a TypedTensor<T>, &'a TypedTensor<T>, &'a TypedTensor<T>)> {
    let mismatch = || ternary_dtype_error(op, [a.dtype(), b.dtype(), c.dtype()]);
    let a_t = a.as_typed::<T>().ok_or_else(mismatch)?;
    let b_t = b.as_typed::<T>().ok_or_else(mismatch)?;
    let c_t = c.as_typed::<T>().ok_or_else(mismatch)?;
    Ok((a_t, b_t, c_t))
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

/// The typed operand behind `operand`, or the refusal this table's wildcard arm
/// produces for the pair.
///
/// `lhs` and `rhs` are the pair being dispatched on, in that order, so the refusal
/// reads the same way whichever operand could not be typed. Callers reach this from
/// a match on the pair's dtypes, so `None` means the table and the runtime dtype
/// disagree rather than a caller mistake.
fn pair_operand<'a, T: TensorScalar>(
    op: &'static str,
    lhs: &Tensor,
    rhs: &Tensor,
    operand: &'a Tensor,
) -> crate::Result<&'a TypedTensor<T>> {
    operand
        .as_typed::<T>()
        .ok_or_else(|| dtype_pair_error(op, lhs.dtype(), rhs.dtype()))
}

fn tensor_pair_error(op: &'static str, lhs: &Tensor, rhs: &Tensor) -> crate::Error {
    dtype_pair_error(op, lhs.dtype(), rhs.dtype())
}

fn read_pair_error(op: &'static str, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Error {
    dtype_pair_error(op, lhs.dtype(), rhs.dtype())
}

/// A host tensor operand the erased strided entries can read.
///
/// Owned tensors and borrowed views share one kernel body per operation; the
/// shape is read before any view is formed so shape errors keep their order.
trait HostOperand<T> {
    fn operand_shape(&self) -> &[usize];
    fn strided(&self, op: &'static str) -> crate::Result<StridedView<'_, T>>;
}

impl<T: Copy + TensorScalar> HostOperand<T> for TypedTensor<T> {
    fn operand_shape(&self) -> &[usize] {
        self.shape()
    }

    fn strided(&self, op: &'static str) -> crate::Result<StridedView<'_, T>> {
        typed_view(op, self)
    }
}

impl<T: Copy + 'static, R: TensorRank> HostOperand<T> for TypedTensorView<'_, T, R> {
    fn operand_shape(&self) -> &[usize] {
        self.shape()
    }

    fn strided(&self, op: &'static str) -> crate::Result<StridedView<'_, T>> {
        typed_view_from_view(op, self)
    }
}

type AxisVec<T> = SmallVec<[T; 8]>;

/// The layout of one erased input descriptor.
///
/// A rank-0 operand broadcast against a ranked output keeps its storage and
/// offset and reads through zero strides.
struct Operand<'a, T> {
    data: &'a [T],
    dims: AxisVec<usize>,
    strides: AxisVec<isize>,
    offset: isize,
}

impl<'a, T: KernelStorageElement> Operand<'a, T> {
    fn new(view: &StridedView<'a, T>) -> Self {
        Self {
            data: view.data(),
            dims: view.dims().iter().copied().collect(),
            strides: view.strides().iter().copied().collect(),
            offset: view.offset(),
        }
    }

    fn broadcast_to(view: &StridedView<'a, T>, shape: &[usize]) -> Self {
        if view.dims() == shape {
            return Self::new(view);
        }
        Self {
            data: view.data(),
            dims: shape.iter().copied().collect(),
            strides: shape.iter().map(|_| 0).collect(),
            offset: view.offset(),
        }
    }

    fn erased(&self, op: &'static str) -> crate::Result<ErasedRawStridedRef<'_>> {
        ErasedRawStridedRef::from_slice(self.data, &self.dims, &self.strides, self.offset)
            .map_err(|err| crate::Error::backend_source(op, err))
    }
}

fn strided_error(op: &'static str, dtype: DType, err: StridedError) -> crate::Error {
    match err {
        StridedError::IntegerDivisionByZero { .. } => crate::cpu_division_by_zero(op, dtype),
        err => crate::Error::backend_source(op, err),
    }
}

/// Allocate a pooled column-major output of `shape` and let one erased entry
/// fill it.
///
/// This is the only place the elementwise family finalizes pooled output.
fn run_into<O>(
    op: &'static str,
    dtype: DType,
    buffers: &mut BufferPool,
    shape: &[usize],
    write: impl FnOnce(&mut ErasedRawStridedUninitMut<'_>) -> Result<(), StridedError>,
) -> crate::Result<TypedTensor<O>>
where
    O: PoolScalar + KernelStorageElement,
{
    let strides = col_major_strides(shape)?;
    let mut out = PooledUninitOutput::<O>::new(buffers, shape.to_vec())?;
    {
        let mut dest = ErasedRawStridedUninitMut::from_uninit_slice(
            out.as_uninit_slice_mut(),
            shape,
            &strides,
            0,
        )
        .map_err(|err| crate::Error::backend_source(op, err))?;
        write(&mut dest).map_err(|err| strided_error(op, dtype, err))?;
    }
    // SAFETY: a successful erased strided entry initializes every element of its dense column-major destination, and the destination descriptor is dropped above.
    unsafe { out.assume_init() }
}

/// The output shape of a binary operation that broadcasts only rank-0 operands.
fn scalar_broadcast_shape<'a>(
    op: &'static str,
    lhs: &'a [usize],
    rhs: &'a [usize],
) -> crate::Result<&'a [usize]> {
    if lhs == rhs || rhs.is_empty() {
        Ok(lhs)
    } else if lhs.is_empty() {
        Ok(rhs)
    } else {
        Err(crate::Error::shape_mismatch(op, lhs.to_vec(), rhs.to_vec()))
    }
}

fn ensure_same_shape(op: &'static str, lhs: &[usize], rhs: &[usize]) -> crate::Result<()> {
    if lhs == rhs {
        Ok(())
    } else {
        Err(crate::Error::shape_mismatch(op, lhs.to_vec(), rhs.to_vec()))
    }
}

/// Binary arithmetic with rank-0 broadcasting on either side.
fn zip_with_pool<T, L, R>(
    op: &'static str,
    zip: ErasedZipOp,
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    lhs: &L,
    rhs: &R,
) -> crate::Result<TypedTensor<T>>
where
    T: PoolScalar + KernelStorageElement,
    L: HostOperand<T> + ?Sized,
    R: HostOperand<T> + ?Sized,
{
    let shape = scalar_broadcast_shape(op, lhs.operand_shape(), rhs.operand_shape())?;
    let lhs_view = lhs.strided(op)?;
    let rhs_view = rhs.strided(op)?;
    let lhs = Operand::broadcast_to(&lhs_view, shape);
    let rhs = Operand::broadcast_to(&rhs_view, shape);
    let lhs = lhs.erased(op)?;
    let rhs = rhs.erased(op)?;
    run_into::<T>(op, T::dtype(), buffers, shape, |dest| {
        erased_zip_into_uninit(
            T::DTYPE,
            zip,
            ctx,
            dest,
            &ErasedRawStridedPtr::from_ref(&lhs),
            &ErasedRawStridedPtr::from_ref(&rhs),
        )
    })
}

/// Integer division or remainder.
///
/// The strided entry scans the divisor it reads, which for a rank-0 divisor
/// broadcast into an empty output is nothing; the rank-0 divisor is therefore
/// checked here so a zero divisor is refused regardless of the output extent.
fn integer_division_with_pool<T, L, R>(
    op: &'static str,
    zip: ErasedZipOp,
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    lhs: &L,
    rhs: &R,
) -> crate::Result<TypedTensor<T>>
where
    T: PoolScalar + KernelStorageElement + Zero + PartialEq,
    L: HostOperand<T> + ?Sized,
    R: HostOperand<T> + ?Sized,
{
    if rhs.operand_shape().is_empty() && rhs.strided(op)?.get(&[]) == T::zero() {
        return Err(crate::cpu_division_by_zero(op, T::dtype()));
    }
    zip_with_pool(op, zip, buffers, ctx, lhs, rhs)
}

/// Binary operation over two operands of the same shape.
fn same_shape_zip_with_pool<T, L, R>(
    op: &'static str,
    zip: ErasedZipOp,
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    lhs: &L,
    rhs: &R,
) -> crate::Result<TypedTensor<T>>
where
    T: PoolScalar + KernelStorageElement,
    L: HostOperand<T> + ?Sized,
    R: HostOperand<T> + ?Sized,
{
    ensure_same_shape(op, lhs.operand_shape(), rhs.operand_shape())?;
    zip_with_pool(op, zip, buffers, ctx, lhs, rhs)
}

/// Unary operation; `O` differs from `T` only for complex absolute value.
fn map_with_pool<T, O, A>(
    op: &'static str,
    map: ErasedMapOp,
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    input: &A,
) -> crate::Result<TypedTensor<O>>
where
    T: KernelStorageElement,
    O: PoolScalar + KernelStorageElement,
    A: HostOperand<T> + ?Sized,
{
    let view = input.strided(op)?;
    let source = Operand::new(&view);
    let source = source.erased(op)?;
    run_into::<O>(op, O::dtype(), buffers, view.dims(), |dest| {
        erased_map_into_uninit(
            T::DTYPE,
            map,
            ctx,
            dest,
            &ErasedRawStridedPtr::from_ref(&source),
        )
    })
}

fn compare_op(dir: &CompareDir) -> CompareOp {
    match dir {
        CompareDir::Eq => CompareOp::Eq,
        CompareDir::Lt => CompareOp::Lt,
        CompareDir::Le => CompareOp::Le,
        CompareDir::Gt => CompareOp::Gt,
        CompareDir::Ge => CompareOp::Ge,
    }
}

fn typed_compare_with_pool<T, L, R>(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    lhs: &L,
    rhs: &R,
    dir: &CompareDir,
) -> crate::Result<TypedTensor<bool>>
where
    T: KernelStorageElement,
    L: HostOperand<T> + ?Sized,
    R: HostOperand<T> + ?Sized,
{
    let op = "compare";
    ensure_same_shape(op, lhs.operand_shape(), rhs.operand_shape())?;
    let lhs_view = lhs.strided(op)?;
    let rhs_view = rhs.strided(op)?;
    let lhs = Operand::new(&lhs_view);
    let rhs = Operand::new(&rhs_view);
    let lhs = lhs.erased(op)?;
    let rhs = rhs.erased(op)?;
    run_into::<bool>(op, DType::Bool, buffers, lhs_view.dims(), |dest| {
        erased_compare_into_uninit(
            T::DTYPE,
            compare_op(dir),
            ctx,
            dest,
            &ErasedRawStridedPtr::from_ref(&lhs),
            &ErasedRawStridedPtr::from_ref(&rhs),
        )
    })
}

fn typed_select_with_pool<T, P, A, B>(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    pred: &P,
    on_true: &A,
    on_false: &B,
) -> crate::Result<TypedTensor<T>>
where
    T: PoolScalar + KernelStorageElement,
    P: HostOperand<bool> + ?Sized,
    A: HostOperand<T> + ?Sized,
    B: HostOperand<T> + ?Sized,
{
    let op = "select";
    ensure_same_shape(op, pred.operand_shape(), on_true.operand_shape())?;
    ensure_same_shape(op, pred.operand_shape(), on_false.operand_shape())?;
    let pred_view = pred.strided(op)?;
    let true_view = on_true.strided(op)?;
    let false_view = on_false.strided(op)?;
    let pred = Operand::new(&pred_view);
    let on_true = Operand::new(&true_view);
    let on_false = Operand::new(&false_view);
    let pred = pred.erased(op)?;
    let on_true = on_true.erased(op)?;
    let on_false = on_false.erased(op)?;
    run_into::<T>(op, T::dtype(), buffers, pred_view.dims(), |dest| {
        erased_select_into_uninit(
            T::DTYPE,
            ctx,
            dest,
            &ErasedRawStridedPtr::from_ref(&pred),
            &ErasedRawStridedPtr::from_ref(&on_true),
            &ErasedRawStridedPtr::from_ref(&on_false),
        )
    })
}

fn typed_clamp_with_pool<T, I, L, U>(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    input: &I,
    lower: &L,
    upper: &U,
) -> crate::Result<TypedTensor<T>>
where
    T: PoolScalar + KernelStorageElement,
    I: HostOperand<T> + ?Sized,
    L: HostOperand<T> + ?Sized,
    U: HostOperand<T> + ?Sized,
{
    let op = "clamp";
    ensure_same_shape(op, input.operand_shape(), lower.operand_shape())?;
    ensure_same_shape(op, input.operand_shape(), upper.operand_shape())?;
    let input_view = input.strided(op)?;
    let lower_view = lower.strided(op)?;
    let upper_view = upper.strided(op)?;
    let input = Operand::new(&input_view);
    let lower = Operand::new(&lower_view);
    let upper = Operand::new(&upper_view);
    let input = input.erased(op)?;
    let lower = lower.erased(op)?;
    let upper = upper.erased(op)?;
    run_into::<T>(op, T::dtype(), buffers, input_view.dims(), |dest| {
        erased_clamp_into_uninit(
            T::DTYPE,
            ctx,
            dest,
            &ErasedRawStridedPtr::from_ref(&input),
            &ErasedRawStridedPtr::from_ref(&lower),
            &ErasedRawStridedPtr::from_ref(&upper),
        )
    })
}

// One typed entry per operation serves the float, complex, and integer arms:
// the strided entries wrap signed-integer arithmetic themselves.
macro_rules! typed_zip_entry {
    ($name:ident, $op:literal, $zip:ident, $kernel:ident) => {
        fn $name<T, L, R>(
            buffers: &mut BufferPool,
            ctx: &ExecContext,
            lhs: &L,
            rhs: &R,
        ) -> crate::Result<TypedTensor<T>>
        where
            T: PoolScalar + KernelStorageElement + Zero + PartialEq,
            L: HostOperand<T> + ?Sized,
            R: HostOperand<T> + ?Sized,
        {
            $kernel($op, ErasedZipOp::$zip, buffers, ctx, lhs, rhs)
        }
    };
}

typed_zip_entry!(typed_add_with_pool, "add", Add, zip_with_pool);
typed_zip_entry!(typed_sub_with_pool, "sub", Subtract, zip_with_pool);
typed_zip_entry!(typed_mul_with_pool, "mul", Multiply, zip_with_pool);
typed_zip_entry!(typed_div_with_pool, "div", Divide, zip_with_pool);
typed_zip_entry!(typed_rem_with_pool, "rem", Remainder, zip_with_pool);
typed_zip_entry!(
    typed_integer_div_with_pool,
    "div",
    Divide,
    integer_division_with_pool
);
typed_zip_entry!(
    typed_integer_rem_with_pool,
    "rem",
    Remainder,
    integer_division_with_pool
);
typed_zip_entry!(
    typed_maximum_with_pool,
    "maximum",
    Maximum,
    same_shape_zip_with_pool
);
typed_zip_entry!(
    typed_minimum_with_pool,
    "minimum",
    Minimum,
    same_shape_zip_with_pool
);

macro_rules! typed_map_entry {
    ($name:ident, $op:literal, $map:ident) => {
        fn $name<T, A>(
            buffers: &mut BufferPool,
            ctx: &ExecContext,
            input: &A,
        ) -> crate::Result<TypedTensor<T>>
        where
            T: PoolScalar + KernelStorageElement,
            A: HostOperand<T> + ?Sized,
        {
            map_with_pool::<T, T, A>($op, ErasedMapOp::$map, buffers, ctx, input)
        }
    };
}

typed_map_entry!(typed_neg_with_pool, "neg", Negate);
typed_map_entry!(typed_conj_with_pool, "conj", Conj);
typed_map_entry!(typed_abs_with_pool, "abs", Abs);
typed_map_entry!(typed_sign_with_pool, "sign", Sign);

fn typed_complex_abs_with_pool<T, A>(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    input: &A,
) -> crate::Result<TypedTensor<T>>
where
    T: PoolScalar + KernelStorageElement,
    Complex<T>: KernelStorageElement,
    A: HostOperand<Complex<T>> + ?Sized,
{
    map_with_pool::<Complex<T>, T, A>("abs", ErasedMapOp::Abs, buffers, ctx, input)
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
fn with_test_pool<T>(f: impl FnOnce(&mut BufferPool, &ExecContext) -> T) -> T {
    let mut buffers = BufferPool::new();
    f(&mut buffers, &ExecContext::serial())
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
    with_test_pool(|buffers, ctx| add_with_pool(buffers, ctx, lhs, rhs))
}

/// Dispatch a same-dtype binary pair to one typed kernel.
///
/// The arm is selected by the pair's dtypes, which `Tensor` derives from the same payload the
/// extraction reads, so `pair_operand` cannot fail inside it. Binding each fallible step to its
/// own `let` keeps every `?` on a line whose call executed, and keeping the body in one macro
/// means the per-preset binary tables share one covered definition instead of one closing
/// `)?))` line per arm.
macro_rules! same_dtype_binary {
    ($buffers:expr, $ctx:expr, $lhs:expr, $rhs:expr, $scalar:ty, $kernel:ident, $op:expr) => {{
        let a = pair_operand::<$scalar>($op, $lhs, $rhs, $lhs)?;
        let b = pair_operand::<$scalar>($op, $lhs, $rhs, $rhs)?;
        let out = $kernel($buffers, $ctx, a, b)?;
        Ok(Tensor::from_typed::<$scalar>(out))
    }};
}

#[doc(hidden)]
pub fn add_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    lhs: &Tensor,
    rhs: &Tensor,
) -> crate::Result<Tensor> {
    match (lhs.dtype(), rhs.dtype()) {
        (DType::F32, DType::F32) => {
            same_dtype_binary!(buffers, ctx, lhs, rhs, f32, typed_add_with_pool, "add")
        }
        (DType::F64, DType::F64) => {
            same_dtype_binary!(buffers, ctx, lhs, rhs, f64, typed_add_with_pool, "add")
        }
        (DType::I32, DType::I32) => {
            same_dtype_binary!(buffers, ctx, lhs, rhs, i32, typed_add_with_pool, "add")
        }
        (DType::I64, DType::I64) => {
            same_dtype_binary!(buffers, ctx, lhs, rhs, i64, typed_add_with_pool, "add")
        }
        (DType::C32, DType::C32) => {
            same_dtype_binary!(
                buffers,
                ctx,
                lhs,
                rhs,
                Complex<f32>,
                typed_add_with_pool,
                "add"
            )
        }
        (DType::C64, DType::C64) => {
            same_dtype_binary!(
                buffers,
                ctx,
                lhs,
                rhs,
                Complex<f64>,
                typed_add_with_pool,
                "add"
            )
        }
        (DType::F32, DType::C32)
            if pair_operand::<f32>("add", lhs, rhs, lhs)?
                .shape()
                .is_empty() =>
        {
            let value = typed_host_data("add", pair_operand::<f32>("add", lhs, rhs, lhs)?)?[0];
            let scalar = complex_scalar_tensor(value)?;
            let other = pair_operand::<Complex<f32>>("add", lhs, rhs, rhs)?;
            let out = typed_add_with_pool(buffers, ctx, &scalar, other)?;
            Ok(Tensor::from_typed::<Complex<f32>>(out))
        }
        (DType::C32, DType::F32)
            if pair_operand::<f32>("add", lhs, rhs, rhs)?
                .shape()
                .is_empty() =>
        {
            let value = typed_host_data("add", pair_operand::<f32>("add", lhs, rhs, rhs)?)?[0];
            let scalar = complex_scalar_tensor(value)?;
            let first = pair_operand::<Complex<f32>>("add", lhs, rhs, lhs)?;
            let out = typed_add_with_pool(buffers, ctx, first, &scalar)?;
            Ok(Tensor::from_typed::<Complex<f32>>(out))
        }
        (DType::F64, DType::C64)
            if pair_operand::<f64>("add", lhs, rhs, lhs)?
                .shape()
                .is_empty() =>
        {
            let value = typed_host_data("add", pair_operand::<f64>("add", lhs, rhs, lhs)?)?[0];
            let scalar = complex_scalar_tensor(value)?;
            let other = pair_operand::<Complex<f64>>("add", lhs, rhs, rhs)?;
            let out = typed_add_with_pool(buffers, ctx, &scalar, other)?;
            Ok(Tensor::from_typed::<Complex<f64>>(out))
        }
        (DType::C64, DType::F64)
            if pair_operand::<f64>("add", lhs, rhs, rhs)?
                .shape()
                .is_empty() =>
        {
            let value = typed_host_data("add", pair_operand::<f64>("add", lhs, rhs, rhs)?)?[0];
            let scalar = complex_scalar_tensor(value)?;
            let first = pair_operand::<Complex<f64>>("add", lhs, rhs, lhs)?;
            let out = typed_add_with_pool(buffers, ctx, first, &scalar)?;
            Ok(Tensor::from_typed::<Complex<f64>>(out))
        }
        _ => Err(tensor_pair_error("add", lhs, rhs)),
    }
}

/// Dispatch a same-variant read pair to that variant's typed kernel.
///
/// Every erased elementwise operation reaches the same typed kernels through
/// these declarations, so an operation adds no matching code of its own.
macro_rules! dispatch_read_same_variant {
    ($buffers:expr, $ctx:expr, $lhs:expr, $rhs:expr, $variant:ident, $func:ident) => {
        match (&$lhs, &$rhs) {
            (TensorRead::Tensor(a), TensorRead::View(TensorView::$variant(b)))
                if a.dtype()
                    == <preset_scalar!($variant) as tenferro_tensor::TensorScalar>::dtype() =>
            {
                let a = a
                    .as_typed::<preset_scalar!($variant)>()
                    .expect("the dtype guard selects this arm");
                let a = a.as_view();
                return Ok(Tensor::from_typed::<preset_scalar!($variant)>($func(
                    $buffers, $ctx, &a, b,
                )?));
            }
            (TensorRead::View(TensorView::$variant(a)), TensorRead::Tensor(b))
                if b.dtype()
                    == <preset_scalar!($variant) as tenferro_tensor::TensorScalar>::dtype() =>
            {
                let b = b
                    .as_typed::<preset_scalar!($variant)>()
                    .expect("the dtype guard selects this arm");
                let b = b.as_view();
                return Ok(Tensor::from_typed::<preset_scalar!($variant)>($func(
                    $buffers, $ctx, a, &b,
                )?));
            }
            (
                TensorRead::View(TensorView::$variant(a)),
                TensorRead::View(TensorView::$variant(b)),
            ) => {
                return Ok(Tensor::from_typed::<preset_scalar!($variant)>($func(
                    $buffers, $ctx, a, b,
                )?));
            }
            _ => {}
        }
    };
}

/// The single list of preset variants an erased elementwise operation reaches.
///
/// An operation supplies only the kernel that implements it for a float-like
/// variant and for an integer variant; the variant list itself exists once here,
/// so adding a preset scalar is a single edit rather than one edit per
/// operation.
macro_rules! dispatch_read_presets {
    ($buffers:expr, $ctx:expr, $lhs:expr, $rhs:expr, $float:ident, $integer:ident) => {
        dispatch_read_real_complex_scalar!($buffers, $ctx, $lhs, $rhs, F32, C32, $float);
        dispatch_read_real_complex_scalar!($buffers, $ctx, $lhs, $rhs, F64, C64, $float);

        dispatch_read_same_variant!($buffers, $ctx, $lhs, $rhs, F32, $float);
        dispatch_read_same_variant!($buffers, $ctx, $lhs, $rhs, F64, $float);
        dispatch_read_same_variant!($buffers, $ctx, $lhs, $rhs, I32, $integer);
        dispatch_read_same_variant!($buffers, $ctx, $lhs, $rhs, I64, $integer);
        dispatch_read_same_variant!($buffers, $ctx, $lhs, $rhs, C32, $float);
        dispatch_read_same_variant!($buffers, $ctx, $lhs, $rhs, C64, $float);
    };
}

macro_rules! dispatch_read_real_complex_scalar {
    ($buffers:expr, $ctx:expr, $lhs:expr, $rhs:expr, $real_variant:ident, $complex_variant:ident, $func:ident) => {
        match (&$lhs, &$rhs) {
            (TensorRead::Tensor(real), TensorRead::View(TensorView::$complex_variant(complex)))
                if real.dtype()
                    == <preset_scalar!($real_variant) as tenferro_tensor::TensorScalar>::dtype(
                    )
                    && real.shape().is_empty() =>
            {
                let real = real
                    .as_typed::<preset_scalar!($real_variant)>()
                    .expect("the dtype guard selects this arm");
                let scalar = complex_scalar_tensor_from_tensor(real)?;
                let scalar = scalar.as_view();
                return Ok(Tensor::from_typed::<preset_scalar!($complex_variant)>(
                    $func($buffers, $ctx, &scalar, complex)?,
                ));
            }
            (TensorRead::View(TensorView::$real_variant(real)), TensorRead::Tensor(complex))
                if complex.dtype()
                    == <preset_scalar!($complex_variant) as tenferro_tensor::TensorScalar>::dtype()
                    && real.shape().is_empty() =>
            {
                let complex = complex
                    .as_typed::<preset_scalar!($complex_variant)>()
                    .expect("the dtype guard selects this arm");
                let scalar = complex_scalar_tensor_from_view(real)?;
                let scalar = scalar.as_view();
                let complex = complex.as_view();
                return Ok(Tensor::from_typed::<preset_scalar!($complex_variant)>(
                    $func($buffers, $ctx, &scalar, &complex)?,
                ));
            }
            (
                TensorRead::View(TensorView::$real_variant(real)),
                TensorRead::View(TensorView::$complex_variant(complex)),
            ) if real.shape().is_empty() => {
                let scalar = complex_scalar_tensor_from_view(real)?;
                let scalar = scalar.as_view();
                return Ok(Tensor::from_typed::<preset_scalar!($complex_variant)>(
                    $func($buffers, $ctx, &scalar, complex)?,
                ));
            }
            (TensorRead::Tensor(complex), TensorRead::View(TensorView::$real_variant(real)))
                if complex.dtype()
                    == <preset_scalar!($complex_variant) as tenferro_tensor::TensorScalar>::dtype()
                    && real.shape().is_empty() =>
            {
                let complex = complex
                    .as_typed::<preset_scalar!($complex_variant)>()
                    .expect("the dtype guard selects this arm");
                let complex = complex.as_view();
                let scalar = complex_scalar_tensor_from_view(real)?;
                let scalar = scalar.as_view();
                return Ok(Tensor::from_typed::<preset_scalar!($complex_variant)>(
                    $func($buffers, $ctx, &complex, &scalar)?,
                ));
            }
            (TensorRead::View(TensorView::$complex_variant(complex)), TensorRead::Tensor(real))
                if real.dtype()
                    == <preset_scalar!($real_variant) as tenferro_tensor::TensorScalar>::dtype(
                    )
                    && real.shape().is_empty() =>
            {
                let real = real
                    .as_typed::<preset_scalar!($real_variant)>()
                    .expect("the dtype guard selects this arm");
                let scalar = complex_scalar_tensor_from_tensor(real)?;
                let scalar = scalar.as_view();
                return Ok(Tensor::from_typed::<preset_scalar!($complex_variant)>(
                    $func($buffers, $ctx, complex, &scalar)?,
                ));
            }
            (
                TensorRead::View(TensorView::$complex_variant(complex)),
                TensorRead::View(TensorView::$real_variant(real)),
            ) if real.shape().is_empty() => {
                let scalar = complex_scalar_tensor_from_view(real)?;
                let scalar = scalar.as_view();
                return Ok(Tensor::from_typed::<preset_scalar!($complex_variant)>(
                    $func($buffers, $ctx, complex, &scalar)?,
                ));
            }
            _ => {}
        }
    };
}

#[doc(hidden)]
pub fn add_read_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    if let (TensorRead::Tensor(lhs), TensorRead::Tensor(rhs)) = (&lhs, &rhs) {
        return add_with_pool(buffers, ctx, lhs, rhs);
    }

    dispatch_read_presets!(
        buffers,
        ctx,
        lhs,
        rhs,
        typed_add_with_pool,
        typed_add_with_pool
    );

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
    with_test_pool(|buffers, ctx| sub_with_pool(buffers, ctx, lhs, rhs))
}

#[doc(hidden)]
pub fn sub_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    lhs: &Tensor,
    rhs: &Tensor,
) -> crate::Result<Tensor> {
    match (lhs.dtype(), rhs.dtype()) {
        (DType::F32, DType::F32) => {
            same_dtype_binary!(buffers, ctx, lhs, rhs, f32, typed_sub_with_pool, "sub")
        }
        (DType::F64, DType::F64) => {
            same_dtype_binary!(buffers, ctx, lhs, rhs, f64, typed_sub_with_pool, "sub")
        }
        (DType::I32, DType::I32) => {
            same_dtype_binary!(buffers, ctx, lhs, rhs, i32, typed_sub_with_pool, "sub")
        }
        (DType::I64, DType::I64) => {
            same_dtype_binary!(buffers, ctx, lhs, rhs, i64, typed_sub_with_pool, "sub")
        }
        (DType::C32, DType::C32) => {
            same_dtype_binary!(
                buffers,
                ctx,
                lhs,
                rhs,
                Complex<f32>,
                typed_sub_with_pool,
                "sub"
            )
        }
        (DType::C64, DType::C64) => {
            same_dtype_binary!(
                buffers,
                ctx,
                lhs,
                rhs,
                Complex<f64>,
                typed_sub_with_pool,
                "sub"
            )
        }
        (DType::F32, DType::C32)
            if pair_operand::<f32>("sub", lhs, rhs, lhs)?
                .shape()
                .is_empty() =>
        {
            let value = typed_host_data("sub", pair_operand::<f32>("sub", lhs, rhs, lhs)?)?[0];
            let scalar = complex_scalar_tensor(value)?;
            let other = pair_operand::<Complex<f32>>("sub", lhs, rhs, rhs)?;
            let out = typed_sub_with_pool(buffers, ctx, &scalar, other)?;
            Ok(Tensor::from_typed::<Complex<f32>>(out))
        }
        (DType::C32, DType::F32)
            if pair_operand::<f32>("sub", lhs, rhs, rhs)?
                .shape()
                .is_empty() =>
        {
            let value = typed_host_data("sub", pair_operand::<f32>("sub", lhs, rhs, rhs)?)?[0];
            let scalar = complex_scalar_tensor(value)?;
            let first = pair_operand::<Complex<f32>>("sub", lhs, rhs, lhs)?;
            let out = typed_sub_with_pool(buffers, ctx, first, &scalar)?;
            Ok(Tensor::from_typed::<Complex<f32>>(out))
        }
        (DType::F64, DType::C64)
            if pair_operand::<f64>("sub", lhs, rhs, lhs)?
                .shape()
                .is_empty() =>
        {
            let value = typed_host_data("sub", pair_operand::<f64>("sub", lhs, rhs, lhs)?)?[0];
            let scalar = complex_scalar_tensor(value)?;
            let other = pair_operand::<Complex<f64>>("sub", lhs, rhs, rhs)?;
            let out = typed_sub_with_pool(buffers, ctx, &scalar, other)?;
            Ok(Tensor::from_typed::<Complex<f64>>(out))
        }
        (DType::C64, DType::F64)
            if pair_operand::<f64>("sub", lhs, rhs, rhs)?
                .shape()
                .is_empty() =>
        {
            let value = typed_host_data("sub", pair_operand::<f64>("sub", lhs, rhs, rhs)?)?[0];
            let scalar = complex_scalar_tensor(value)?;
            let first = pair_operand::<Complex<f64>>("sub", lhs, rhs, lhs)?;
            let out = typed_sub_with_pool(buffers, ctx, first, &scalar)?;
            Ok(Tensor::from_typed::<Complex<f64>>(out))
        }
        _ => Err(tensor_pair_error("sub", lhs, rhs)),
    }
}

#[doc(hidden)]
pub fn sub_read_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    if let (TensorRead::Tensor(lhs), TensorRead::Tensor(rhs)) = (&lhs, &rhs) {
        return sub_with_pool(buffers, ctx, lhs, rhs);
    }

    dispatch_read_presets!(
        buffers,
        ctx,
        lhs,
        rhs,
        typed_sub_with_pool,
        typed_sub_with_pool
    );

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
    with_test_pool(|buffers, ctx| mul_with_pool(buffers, ctx, lhs, rhs))
}

fn binary_read_with_pool(
    op: &'static str,
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
    f: impl FnOnce(&mut BufferPool, &ExecContext, &Tensor, &Tensor) -> crate::Result<Tensor>,
) -> crate::Result<Tensor> {
    if let (Some(lhs), Some(rhs)) = (lhs.as_tensor(), rhs.as_tensor()) {
        return f(buffers, ctx, lhs, rhs);
    }

    Err(read_pair_error(op, lhs, rhs))
}

#[doc(hidden)]
pub fn mul_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    lhs: &Tensor,
    rhs: &Tensor,
) -> crate::Result<Tensor> {
    match (lhs.dtype(), rhs.dtype()) {
        (DType::F32, DType::F32) => {
            same_dtype_binary!(buffers, ctx, lhs, rhs, f32, typed_mul_with_pool, "mul")
        }
        (DType::F64, DType::F64) => {
            same_dtype_binary!(buffers, ctx, lhs, rhs, f64, typed_mul_with_pool, "mul")
        }
        (DType::I32, DType::I32) => {
            same_dtype_binary!(buffers, ctx, lhs, rhs, i32, typed_mul_with_pool, "mul")
        }
        (DType::I64, DType::I64) => {
            same_dtype_binary!(buffers, ctx, lhs, rhs, i64, typed_mul_with_pool, "mul")
        }
        (DType::C32, DType::C32) => {
            same_dtype_binary!(
                buffers,
                ctx,
                lhs,
                rhs,
                Complex<f32>,
                typed_mul_with_pool,
                "mul"
            )
        }
        (DType::C64, DType::C64) => {
            same_dtype_binary!(
                buffers,
                ctx,
                lhs,
                rhs,
                Complex<f64>,
                typed_mul_with_pool,
                "mul"
            )
        }
        (DType::F32, DType::C32)
            if pair_operand::<f32>("mul", lhs, rhs, lhs)?
                .shape()
                .is_empty() =>
        {
            let value = typed_host_data("mul", pair_operand::<f32>("mul", lhs, rhs, lhs)?)?[0];
            let scalar = complex_scalar_tensor(value)?;
            let other = pair_operand::<Complex<f32>>("mul", lhs, rhs, rhs)?;
            let out = typed_mul_with_pool(buffers, ctx, &scalar, other)?;
            Ok(Tensor::from_typed::<Complex<f32>>(out))
        }
        (DType::C32, DType::F32)
            if pair_operand::<f32>("mul", lhs, rhs, rhs)?
                .shape()
                .is_empty() =>
        {
            let value = typed_host_data("mul", pair_operand::<f32>("mul", lhs, rhs, rhs)?)?[0];
            let scalar = complex_scalar_tensor(value)?;
            let first = pair_operand::<Complex<f32>>("mul", lhs, rhs, lhs)?;
            let out = typed_mul_with_pool(buffers, ctx, first, &scalar)?;
            Ok(Tensor::from_typed::<Complex<f32>>(out))
        }
        (DType::F64, DType::C64)
            if pair_operand::<f64>("mul", lhs, rhs, lhs)?
                .shape()
                .is_empty() =>
        {
            let value = typed_host_data("mul", pair_operand::<f64>("mul", lhs, rhs, lhs)?)?[0];
            let scalar = complex_scalar_tensor(value)?;
            let other = pair_operand::<Complex<f64>>("mul", lhs, rhs, rhs)?;
            let out = typed_mul_with_pool(buffers, ctx, &scalar, other)?;
            Ok(Tensor::from_typed::<Complex<f64>>(out))
        }
        (DType::C64, DType::F64)
            if pair_operand::<f64>("mul", lhs, rhs, rhs)?
                .shape()
                .is_empty() =>
        {
            let value = typed_host_data("mul", pair_operand::<f64>("mul", lhs, rhs, rhs)?)?[0];
            let scalar = complex_scalar_tensor(value)?;
            let first = pair_operand::<Complex<f64>>("mul", lhs, rhs, lhs)?;
            let out = typed_mul_with_pool(buffers, ctx, first, &scalar)?;
            Ok(Tensor::from_typed::<Complex<f64>>(out))
        }
        _ => Err(tensor_pair_error("mul", lhs, rhs)),
    }
}

#[doc(hidden)]
pub fn mul_read_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    if let (TensorRead::Tensor(lhs), TensorRead::Tensor(rhs)) = (&lhs, &rhs) {
        return mul_with_pool(buffers, ctx, lhs, rhs);
    }

    dispatch_read_presets!(
        buffers,
        ctx,
        lhs,
        rhs,
        typed_mul_with_pool,
        typed_mul_with_pool
    );

    binary_read_with_pool("mul", buffers, ctx, lhs, rhs, mul_with_pool)
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

/// The typed tensor behind `input`, or this module's refusal when the dtype is a caller-owned payload.
///
/// The read view has no externally defined variant, so a payload the caller owns is reported instead of
/// being unwrapped.
fn read_typed<'a, T: tenferro_tensor::TensorScalar>(
    input: &'a Tensor,
) -> crate::Result<TypedTensorView<'a, T>> {
    let dtype = input.dtype();
    input
        .as_typed::<T>()
        .map(|tensor| tensor.as_view())
        .ok_or_else(|| {
            crate::Error::unsupported_dtype(
                "read_as_cpu_view",
                dtype,
                "the CPU read view covers the preset scalars",
            )
        })
}

fn read_as_cpu_view(input: TensorRead<'_>) -> crate::Result<CpuReadView<'_>> {
    match input {
        TensorRead::Tensor(tensor) => match tensor.dtype() {
            DType::F32 => Ok(CpuReadView::F32(read_typed::<f32>(tensor)?)),
            DType::F64 => Ok(CpuReadView::F64(read_typed::<f64>(tensor)?)),
            DType::I32 => Ok(CpuReadView::I32(read_typed::<i32>(tensor)?)),
            DType::I64 => Ok(CpuReadView::I64(read_typed::<i64>(tensor)?)),
            DType::Bool => Ok(CpuReadView::Bool(read_typed::<bool>(tensor)?)),
            DType::C32 => Ok(CpuReadView::C32(read_typed::<tenferro_tensor::Complex32>(
                tensor,
            )?)),
            DType::C64 => Ok(CpuReadView::C64(read_typed::<tenferro_tensor::Complex64>(
                tensor,
            )?)),
            DType::External(..) => Err(crate::Error::unsupported_dtype(
                "read_as_cpu_view",
                tensor.dtype(),
                "the CPU read view covers the preset scalars",
            )),
        },
        TensorRead::View(TensorView::F32(view)) => Ok(CpuReadView::F32(view)),
        TensorRead::View(TensorView::F64(view)) => Ok(CpuReadView::F64(view)),
        TensorRead::View(TensorView::I32(view)) => Ok(CpuReadView::I32(view)),
        TensorRead::View(TensorView::I64(view)) => Ok(CpuReadView::I64(view)),
        TensorRead::View(TensorView::Bool(view)) => Ok(CpuReadView::Bool(view)),
        TensorRead::View(TensorView::C32(view)) => Ok(CpuReadView::C32(view)),
        TensorRead::View(TensorView::C64(view)) => Ok(CpuReadView::C64(view)),
    }
}

/// Axis-mapped broadcast multiply of two host operands into a dense output.
#[allow(clippy::too_many_arguments)]
fn typed_broadcast_mul_with_pool<T, L, R>(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    lhs: &L,
    lhs_shape: &[usize],
    lhs_dims: &[usize],
    rhs: &R,
    rhs_shape: &[usize],
    rhs_dims: &[usize],
) -> crate::Result<TypedTensor<T>>
where
    T: PoolScalar + KernelStorageElement,
    L: HostOperand<T> + ?Sized,
    R: HostOperand<T> + ?Sized,
{
    let op = "broadcast_multiply";
    ensure_same_shape(op, lhs_shape, rhs_shape)?;
    let lhs_view = lhs.strided(op)?;
    let rhs_view = rhs.strided(op)?;
    let lhs = Operand::new(&lhs_view);
    let rhs = Operand::new(&rhs_view);
    let lhs = lhs.erased(op)?;
    let rhs = rhs.erased(op)?;
    run_into::<T>(op, T::dtype(), buffers, lhs_shape, |dest| {
        erased_broadcast_mul_into_uninit(
            T::DTYPE,
            ctx,
            dest,
            &ErasedRawStridedPtr::from_ref(&lhs),
            lhs_dims,
            &ErasedRawStridedPtr::from_ref(&rhs),
            rhs_dims,
        )
    })
}

/// A broadcast multiply whose output follows the inputs' physical order.
///
/// `strided_kernel::plan_lazy_outer_product` decides eligibility and the base
/// layout; the base is filled through a destination over it with the logical
/// output dims, so the ordinary broadcast multiply writes it.
struct LazyOuterProduct<T: TensorScalar> {
    base: TypedTensor<T>,
    strides: Vec<isize>,
}

#[allow(clippy::too_many_arguments)]
fn try_lazy_outer_product_with_pool<T>(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    lhs: &TypedTensorView<'_, T>,
    lhs_shape: &[usize],
    lhs_dims: &[usize],
    rhs: &TypedTensorView<'_, T>,
    rhs_shape: &[usize],
    rhs_dims: &[usize],
) -> crate::Result<Option<LazyOuterProduct<T>>>
where
    T: PoolScalar + KernelStorageElement,
{
    let op = "broadcast_multiply";
    if lhs_shape != rhs_shape
        || lhs.shape().len() != lhs_dims.len()
        || rhs.shape().len() != rhs_dims.len()
        || lhs.backend_buffer().is_some()
        || rhs.backend_buffer().is_some()
    {
        return Ok(None);
    }
    let Some(layout) = plan_lazy_outer_product(
        lhs_shape,
        lhs.shape(),
        lhs.strides(),
        lhs_dims,
        rhs.shape(),
        rhs.strides(),
        rhs_dims,
    )
    .map_err(|err| crate::Error::backend_source(op, err))?
    else {
        return Ok(None);
    };

    let lhs_view = typed_view_from_view(op, lhs)?;
    let rhs_view = typed_view_from_view(op, rhs)?;
    let lhs_operand = Operand::new(&lhs_view);
    let rhs_operand = Operand::new(&rhs_view);
    let lhs_source = lhs_operand.erased(op)?;
    let rhs_source = rhs_operand.erased(op)?;
    let mut base = PooledUninitOutput::<T>::new(buffers, layout.base_dims.clone())?;
    {
        let mut dest = ErasedRawStridedUninitMut::from_uninit_slice(
            base.as_uninit_slice_mut(),
            lhs_shape,
            &layout.output_strides,
            0,
        )
        .map_err(|err| crate::Error::backend_source(op, err))?;
        erased_broadcast_mul_into_uninit(
            T::DTYPE,
            ctx,
            &mut dest,
            &ErasedRawStridedPtr::from_ref(&lhs_source),
            lhs_dims,
            &ErasedRawStridedPtr::from_ref(&rhs_source),
            rhs_dims,
        )
        .map_err(|err| crate::Error::backend_source(op, err))?;
    }
    // SAFETY: the planned output strides map the logical output one-to-one onto the dense base, so the successful broadcast multiply initialized every base element; the destination descriptor is dropped above.
    let base = unsafe { base.assume_init()? };
    Ok(Some(LazyOuterProduct {
        base,
        strides: layout.output_strides,
    }))
}

#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub fn broadcast_multiply_read_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    lhs: TensorRead<'_>,
    lhs_shape: &[usize],
    lhs_dims: &[usize],
    rhs: TensorRead<'_>,
    rhs_shape: &[usize],
    rhs_dims: &[usize],
) -> crate::Result<Option<Tensor>> {
    let lhs = read_as_cpu_view(lhs)?;
    let rhs = read_as_cpu_view(rhs)?;

    macro_rules! dispatch {
        ($variant:ident, $lhs:expr, $rhs:expr) => {{
            Ok(Some(Tensor::from_typed::<preset_scalar!($variant)>(
                typed_broadcast_mul_with_pool(
                    buffers, ctx, &$lhs, lhs_shape, lhs_dims, &$rhs, rhs_shape, rhs_dims,
                )?,
            )))
        }};
    }

    match (lhs, rhs) {
        (CpuReadView::F32(lhs), CpuReadView::F32(rhs)) => dispatch!(F32, lhs, rhs),
        (CpuReadView::F64(lhs), CpuReadView::F64(rhs)) => dispatch!(F64, lhs, rhs),
        (CpuReadView::I32(lhs), CpuReadView::I32(rhs)) => dispatch!(I32, lhs, rhs),
        (CpuReadView::I64(lhs), CpuReadView::I64(rhs)) => dispatch!(I64, lhs, rhs),
        (CpuReadView::C32(lhs), CpuReadView::C32(rhs)) => dispatch!(C32, lhs, rhs),
        (CpuReadView::C64(lhs), CpuReadView::C64(rhs)) => dispatch!(C64, lhs, rhs),
        _ => Ok(None),
    }
}

#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub fn broadcast_multiply_value_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    lhs: TensorRead<'_>,
    lhs_shape: &[usize],
    lhs_dims: &[usize],
    rhs: TensorRead<'_>,
    rhs_shape: &[usize],
    rhs_dims: &[usize],
) -> crate::Result<Option<TensorValue>> {
    broadcast_multiply_value_with_pool_and_tag(
        buffers,
        ctx,
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
    ctx: &ExecContext,
    lhs: TensorRead<'_>,
    lhs_shape: &[usize],
    lhs_dims: &[usize],
    rhs: TensorRead<'_>,
    rhs_shape: &[usize],
    rhs_dims: &[usize],
    mut tag_output: impl FnMut(&mut Tensor),
) -> crate::Result<Option<TensorValue>> {
    let lhs_view = read_as_cpu_view(lhs.clone())?;
    let rhs_view = read_as_cpu_view(rhs.clone())?;

    macro_rules! dispatch_lazy {
        ($variant:ident, $lhs:expr, $rhs:expr) => {{
            if let Some(out) = try_lazy_outer_product_with_pool(
                buffers, ctx, &$lhs, lhs_shape, lhs_dims, &$rhs, rhs_shape, rhs_dims,
            )? {
                let mut base = Tensor::from_typed::<preset_scalar!($variant)>(out.base);
                tag_output(&mut base);
                return Ok(Some(TensorValue::from_parts(
                    base,
                    lhs_shape.to_vec(),
                    out.strides,
                    0,
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
        buffers, ctx, lhs, lhs_shape, lhs_dims, rhs, rhs_shape, rhs_dims,
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
    with_test_pool(|buffers, ctx| div_with_pool(buffers, ctx, lhs, rhs))
}

#[doc(hidden)]
pub fn div_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    lhs: &Tensor,
    rhs: &Tensor,
) -> crate::Result<Tensor> {
    match (lhs.dtype(), rhs.dtype()) {
        (DType::F32, DType::F32) => {
            same_dtype_binary!(buffers, ctx, lhs, rhs, f32, typed_div_with_pool, "div")
        }
        (DType::F64, DType::F64) => {
            same_dtype_binary!(buffers, ctx, lhs, rhs, f64, typed_div_with_pool, "div")
        }
        (DType::I32, DType::I32) => {
            same_dtype_binary!(
                buffers,
                ctx,
                lhs,
                rhs,
                i32,
                typed_integer_div_with_pool,
                "div"
            )
        }
        (DType::I64, DType::I64) => {
            same_dtype_binary!(
                buffers,
                ctx,
                lhs,
                rhs,
                i64,
                typed_integer_div_with_pool,
                "div"
            )
        }
        (DType::C32, DType::C32) => {
            same_dtype_binary!(
                buffers,
                ctx,
                lhs,
                rhs,
                Complex<f32>,
                typed_div_with_pool,
                "div"
            )
        }
        (DType::C64, DType::C64) => {
            same_dtype_binary!(
                buffers,
                ctx,
                lhs,
                rhs,
                Complex<f64>,
                typed_div_with_pool,
                "div"
            )
        }
        (DType::F32, DType::C32)
            if pair_operand::<f32>("div", lhs, rhs, lhs)?
                .shape()
                .is_empty() =>
        {
            let value = typed_host_data("div", pair_operand::<f32>("div", lhs, rhs, lhs)?)?[0];
            let scalar = complex_scalar_tensor(value)?;
            let other = pair_operand::<Complex<f32>>("div", lhs, rhs, rhs)?;
            let out = typed_div_with_pool(buffers, ctx, &scalar, other)?;
            Ok(Tensor::from_typed::<Complex<f32>>(out))
        }
        (DType::C32, DType::F32)
            if pair_operand::<f32>("div", lhs, rhs, rhs)?
                .shape()
                .is_empty() =>
        {
            let value = typed_host_data("div", pair_operand::<f32>("div", lhs, rhs, rhs)?)?[0];
            let scalar = complex_scalar_tensor(value)?;
            let first = pair_operand::<Complex<f32>>("div", lhs, rhs, lhs)?;
            let out = typed_div_with_pool(buffers, ctx, first, &scalar)?;
            Ok(Tensor::from_typed::<Complex<f32>>(out))
        }
        (DType::F64, DType::C64)
            if pair_operand::<f64>("div", lhs, rhs, lhs)?
                .shape()
                .is_empty() =>
        {
            let value = typed_host_data("div", pair_operand::<f64>("div", lhs, rhs, lhs)?)?[0];
            let scalar = complex_scalar_tensor(value)?;
            let other = pair_operand::<Complex<f64>>("div", lhs, rhs, rhs)?;
            let out = typed_div_with_pool(buffers, ctx, &scalar, other)?;
            Ok(Tensor::from_typed::<Complex<f64>>(out))
        }
        (DType::C64, DType::F64)
            if pair_operand::<f64>("div", lhs, rhs, rhs)?
                .shape()
                .is_empty() =>
        {
            let value = typed_host_data("div", pair_operand::<f64>("div", lhs, rhs, rhs)?)?[0];
            let scalar = complex_scalar_tensor(value)?;
            let first = pair_operand::<Complex<f64>>("div", lhs, rhs, lhs)?;
            let out = typed_div_with_pool(buffers, ctx, first, &scalar)?;
            Ok(Tensor::from_typed::<Complex<f64>>(out))
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
    ctx: &ExecContext,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let lhs_dtype = lhs.dtype();
    let rhs_dtype = rhs.dtype();
    match (read_as_cpu_view(lhs)?, read_as_cpu_view(rhs)?) {
        (CpuReadView::F32(a), CpuReadView::F32(b)) => Ok(Tensor::from_typed::<f32>(
            typed_div_with_pool(buffers, ctx, &a, &b)?,
        )),
        (CpuReadView::F64(a), CpuReadView::F64(b)) => Ok(Tensor::from_typed::<f64>(
            typed_div_with_pool(buffers, ctx, &a, &b)?,
        )),
        (CpuReadView::I32(a), CpuReadView::I32(b)) => Ok(Tensor::from_typed::<i32>(
            typed_integer_div_with_pool(buffers, ctx, &a, &b)?,
        )),
        (CpuReadView::I64(a), CpuReadView::I64(b)) => Ok(Tensor::from_typed::<i64>(
            typed_integer_div_with_pool(buffers, ctx, &a, &b)?,
        )),
        (CpuReadView::C32(a), CpuReadView::C32(b)) => Ok(Tensor::from_typed::<Complex<f32>>(
            typed_div_with_pool(buffers, ctx, &a, &b)?,
        )),
        (CpuReadView::C64(a), CpuReadView::C64(b)) => Ok(Tensor::from_typed::<Complex<f64>>(
            typed_div_with_pool(buffers, ctx, &a, &b)?,
        )),
        (CpuReadView::F32(real), CpuReadView::C32(complex)) if real.shape().is_empty() => {
            let scalar = complex_scalar_tensor_from_view(&real)?;
            let scalar = scalar.as_view();
            Ok(Tensor::from_typed::<Complex<f32>>(typed_div_with_pool(
                buffers, ctx, &scalar, &complex,
            )?))
        }
        (CpuReadView::C32(complex), CpuReadView::F32(real)) if real.shape().is_empty() => {
            let scalar = complex_scalar_tensor_from_view(&real)?;
            let scalar = scalar.as_view();
            Ok(Tensor::from_typed::<Complex<f32>>(typed_div_with_pool(
                buffers, ctx, &complex, &scalar,
            )?))
        }
        (CpuReadView::F64(real), CpuReadView::C64(complex)) if real.shape().is_empty() => {
            let scalar = complex_scalar_tensor_from_view(&real)?;
            let scalar = scalar.as_view();
            Ok(Tensor::from_typed::<Complex<f64>>(typed_div_with_pool(
                buffers, ctx, &scalar, &complex,
            )?))
        }
        (CpuReadView::C64(complex), CpuReadView::F64(real)) if real.shape().is_empty() => {
            let scalar = complex_scalar_tensor_from_view(&real)?;
            let scalar = scalar.as_view();
            Ok(Tensor::from_typed::<Complex<f64>>(typed_div_with_pool(
                buffers, ctx, &complex, &scalar,
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
    with_test_pool(|buffers, ctx| rem_with_pool(buffers, ctx, lhs, rhs))
}

#[doc(hidden)]
pub fn rem_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    lhs: &Tensor,
    rhs: &Tensor,
) -> crate::Result<Tensor> {
    match (lhs.dtype(), rhs.dtype()) {
        (DType::F32, DType::F32) => {
            same_dtype_binary!(buffers, ctx, lhs, rhs, f32, typed_rem_with_pool, "rem")
        }
        (DType::F64, DType::F64) => {
            same_dtype_binary!(buffers, ctx, lhs, rhs, f64, typed_rem_with_pool, "rem")
        }
        (DType::I32, DType::I32) => {
            same_dtype_binary!(
                buffers,
                ctx,
                lhs,
                rhs,
                i32,
                typed_integer_rem_with_pool,
                "rem"
            )
        }
        (DType::I64, DType::I64) => {
            same_dtype_binary!(
                buffers,
                ctx,
                lhs,
                rhs,
                i64,
                typed_integer_rem_with_pool,
                "rem"
            )
        }
        _ => Err(tensor_pair_error("rem", lhs, rhs)),
    }
}

#[doc(hidden)]
pub fn rem_read_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let lhs_dtype = lhs.dtype();
    let rhs_dtype = rhs.dtype();
    match (read_as_cpu_view(lhs)?, read_as_cpu_view(rhs)?) {
        (CpuReadView::F32(a), CpuReadView::F32(b)) => Ok(Tensor::from_typed::<f32>(
            typed_rem_with_pool(buffers, ctx, &a, &b)?,
        )),
        (CpuReadView::F64(a), CpuReadView::F64(b)) => Ok(Tensor::from_typed::<f64>(
            typed_rem_with_pool(buffers, ctx, &a, &b)?,
        )),
        (CpuReadView::I32(a), CpuReadView::I32(b)) => Ok(Tensor::from_typed::<i32>(
            typed_integer_rem_with_pool(buffers, ctx, &a, &b)?,
        )),
        (CpuReadView::I64(a), CpuReadView::I64(b)) => Ok(Tensor::from_typed::<i64>(
            typed_integer_rem_with_pool(buffers, ctx, &a, &b)?,
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
    with_test_pool(|buffers, ctx| neg_with_pool(buffers, ctx, input))
}

#[doc(hidden)]
pub fn neg_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    input: &Tensor,
) -> crate::Result<Tensor> {
    match input.dtype() {
        DType::F32 => Ok(Tensor::from_typed::<f32>(typed_neg_with_pool(
            buffers,
            ctx,
            input.as_typed::<f32>().ok_or_else(|| {
                unary_dtype_error("neg", input.dtype(), "F32/F64/I32/I64/C32/C64", false)
            })?,
        )?)),
        DType::F64 => Ok(Tensor::from_typed::<f64>(typed_neg_with_pool(
            buffers,
            ctx,
            input.as_typed::<f64>().ok_or_else(|| {
                unary_dtype_error("neg", input.dtype(), "F32/F64/I32/I64/C32/C64", false)
            })?,
        )?)),
        DType::I32 => Ok(Tensor::from_typed::<i32>(typed_neg_with_pool(
            buffers,
            ctx,
            input.as_typed::<i32>().ok_or_else(|| {
                unary_dtype_error("neg", input.dtype(), "F32/F64/I32/I64/C32/C64", false)
            })?,
        )?)),
        DType::I64 => Ok(Tensor::from_typed::<i64>(typed_neg_with_pool(
            buffers,
            ctx,
            input.as_typed::<i64>().ok_or_else(|| {
                unary_dtype_error("neg", input.dtype(), "F32/F64/I32/I64/C32/C64", false)
            })?,
        )?)),
        DType::Bool | DType::External(_) => Err(unary_dtype_error(
            "neg",
            input.dtype(),
            "F32/F64/I32/I64/C32/C64",
            false,
        )),
        DType::C32 => Ok(Tensor::from_typed::<Complex<f32>>(typed_neg_with_pool(
            buffers,
            ctx,
            input.as_typed::<Complex<f32>>().ok_or_else(|| {
                unary_dtype_error("neg", input.dtype(), "F32/F64/I32/I64/C32/C64", false)
            })?,
        )?)),
        DType::C64 => Ok(Tensor::from_typed::<Complex<f64>>(typed_neg_with_pool(
            buffers,
            ctx,
            input.as_typed::<Complex<f64>>().ok_or_else(|| {
                unary_dtype_error("neg", input.dtype(), "F32/F64/I32/I64/C32/C64", false)
            })?,
        )?)),
    }
}

#[doc(hidden)]
pub fn neg_read_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    input: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let dtype = input.dtype();
    match read_as_cpu_view(input)? {
        CpuReadView::F32(t) => Ok(Tensor::from_typed::<f32>(typed_neg_with_pool(
            buffers, ctx, &t,
        )?)),
        CpuReadView::F64(t) => Ok(Tensor::from_typed::<f64>(typed_neg_with_pool(
            buffers, ctx, &t,
        )?)),
        CpuReadView::I32(t) => Ok(Tensor::from_typed::<i32>(typed_neg_with_pool(
            buffers, ctx, &t,
        )?)),
        CpuReadView::I64(t) => Ok(Tensor::from_typed::<i64>(typed_neg_with_pool(
            buffers, ctx, &t,
        )?)),
        CpuReadView::C32(t) => Ok(Tensor::from_typed::<Complex<f32>>(typed_neg_with_pool(
            buffers, ctx, &t,
        )?)),
        CpuReadView::C64(t) => Ok(Tensor::from_typed::<Complex<f64>>(typed_neg_with_pool(
            buffers, ctx, &t,
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
/// assert_eq!(out.as_slice::<Complex<f64>>().unwrap(), &[Complex64::new(1.0, -2.0)]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[cfg(test)]
#[doc(hidden)]
pub fn conj(input: &Tensor) -> crate::Result<Tensor> {
    with_test_pool(|buffers, ctx| conj_with_pool(buffers, ctx, input))
}

#[doc(hidden)]
pub fn conj_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    input: &Tensor,
) -> crate::Result<Tensor> {
    match input.dtype() {
        DType::F32 => Ok(Tensor::from_typed::<f32>(typed_conj_with_pool(
            buffers,
            ctx,
            input
                .as_typed::<f32>()
                .ok_or_else(|| unary_dtype_error("conj", input.dtype(), "F32/F64/C32/C64", true))?,
        )?)),
        DType::F64 => Ok(Tensor::from_typed::<f64>(typed_conj_with_pool(
            buffers,
            ctx,
            input
                .as_typed::<f64>()
                .ok_or_else(|| unary_dtype_error("conj", input.dtype(), "F32/F64/C32/C64", true))?,
        )?)),
        DType::I32 | DType::I64 | DType::Bool | DType::External(_) => Err(unary_dtype_error(
            "conj",
            input.dtype(),
            "F32/F64/C32/C64",
            true,
        )),
        DType::C32 => Ok(Tensor::from_typed::<Complex<f32>>(typed_conj_with_pool(
            buffers,
            ctx,
            input
                .as_typed::<Complex<f32>>()
                .ok_or_else(|| unary_dtype_error("conj", input.dtype(), "F32/F64/C32/C64", true))?,
        )?)),
        DType::C64 => Ok(Tensor::from_typed::<Complex<f64>>(typed_conj_with_pool(
            buffers,
            ctx,
            input
                .as_typed::<Complex<f64>>()
                .ok_or_else(|| unary_dtype_error("conj", input.dtype(), "F32/F64/C32/C64", true))?,
        )?)),
    }
}

#[doc(hidden)]
pub fn conj_read_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    input: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let dtype = input.dtype();
    match read_as_cpu_view(input)? {
        CpuReadView::F32(t) => Ok(Tensor::from_typed::<f32>(typed_conj_with_pool(
            buffers, ctx, &t,
        )?)),
        CpuReadView::F64(t) => Ok(Tensor::from_typed::<f64>(typed_conj_with_pool(
            buffers, ctx, &t,
        )?)),
        CpuReadView::C32(t) => Ok(Tensor::from_typed::<Complex<f32>>(typed_conj_with_pool(
            buffers, ctx, &t,
        )?)),
        CpuReadView::C64(t) => Ok(Tensor::from_typed::<Complex<f64>>(typed_conj_with_pool(
            buffers, ctx, &t,
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
    with_test_pool(|buffers, ctx| abs_with_pool(buffers, ctx, input))
}

#[doc(hidden)]
pub fn abs_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    input: &Tensor,
) -> crate::Result<Tensor> {
    match input.dtype() {
        DType::F32 => Ok(Tensor::from_typed::<f32>(typed_abs_with_pool(
            buffers,
            ctx,
            input.as_typed::<f32>().ok_or_else(|| {
                unary_dtype_error("abs", input.dtype(), "F32/F64/I32/I64/C32/C64", false)
            })?,
        )?)),
        DType::F64 => Ok(Tensor::from_typed::<f64>(typed_abs_with_pool(
            buffers,
            ctx,
            input.as_typed::<f64>().ok_or_else(|| {
                unary_dtype_error("abs", input.dtype(), "F32/F64/I32/I64/C32/C64", false)
            })?,
        )?)),
        DType::I32 => Ok(Tensor::from_typed::<i32>(typed_abs_with_pool(
            buffers,
            ctx,
            input.as_typed::<i32>().ok_or_else(|| {
                unary_dtype_error("abs", input.dtype(), "F32/F64/I32/I64/C32/C64", false)
            })?,
        )?)),
        DType::I64 => Ok(Tensor::from_typed::<i64>(typed_abs_with_pool(
            buffers,
            ctx,
            input.as_typed::<i64>().ok_or_else(|| {
                unary_dtype_error("abs", input.dtype(), "F32/F64/I32/I64/C32/C64", false)
            })?,
        )?)),
        DType::Bool | DType::External(_) => Err(unary_dtype_error(
            "abs",
            input.dtype(),
            "F32/F64/I32/I64/C32/C64",
            false,
        )),
        DType::C32 => Ok(Tensor::from_typed::<f32>(typed_complex_abs_with_pool(
            buffers,
            ctx,
            input.as_typed::<Complex<f32>>().ok_or_else(|| {
                unary_dtype_error("abs", input.dtype(), "F32/F64/I32/I64/C32/C64", false)
            })?,
        )?)),
        DType::C64 => Ok(Tensor::from_typed::<f64>(typed_complex_abs_with_pool(
            buffers,
            ctx,
            input.as_typed::<Complex<f64>>().ok_or_else(|| {
                unary_dtype_error("abs", input.dtype(), "F32/F64/I32/I64/C32/C64", false)
            })?,
        )?)),
    }
}

#[doc(hidden)]
pub fn abs_read_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    input: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let dtype = input.dtype();
    match read_as_cpu_view(input)? {
        CpuReadView::F32(t) => Ok(Tensor::from_typed::<f32>(typed_abs_with_pool(
            buffers, ctx, &t,
        )?)),
        CpuReadView::F64(t) => Ok(Tensor::from_typed::<f64>(typed_abs_with_pool(
            buffers, ctx, &t,
        )?)),
        CpuReadView::I32(t) => Ok(Tensor::from_typed::<i32>(typed_abs_with_pool(
            buffers, ctx, &t,
        )?)),
        CpuReadView::I64(t) => Ok(Tensor::from_typed::<i64>(typed_abs_with_pool(
            buffers, ctx, &t,
        )?)),
        CpuReadView::C32(t) => Ok(Tensor::from_typed::<f32>(typed_complex_abs_with_pool(
            buffers, ctx, &t,
        )?)),
        CpuReadView::C64(t) => Ok(Tensor::from_typed::<f64>(typed_complex_abs_with_pool(
            buffers, ctx, &t,
        )?)),
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
    with_test_pool(|buffers, ctx| sign_with_pool(buffers, ctx, input))
}

#[doc(hidden)]
pub fn sign_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    input: &Tensor,
) -> crate::Result<Tensor> {
    match input.dtype() {
        DType::F32 => Ok(Tensor::from_typed::<f32>(typed_sign_with_pool(
            buffers,
            ctx,
            input.as_typed::<f32>().ok_or_else(|| {
                unary_dtype_error("sign", input.dtype(), "F32/F64/I32/I64/C32/C64", false)
            })?,
        )?)),
        DType::F64 => Ok(Tensor::from_typed::<f64>(typed_sign_with_pool(
            buffers,
            ctx,
            input.as_typed::<f64>().ok_or_else(|| {
                unary_dtype_error("sign", input.dtype(), "F32/F64/I32/I64/C32/C64", false)
            })?,
        )?)),
        DType::I32 => Ok(Tensor::from_typed::<i32>(typed_sign_with_pool(
            buffers,
            ctx,
            input.as_typed::<i32>().ok_or_else(|| {
                unary_dtype_error("sign", input.dtype(), "F32/F64/I32/I64/C32/C64", false)
            })?,
        )?)),
        DType::I64 => Ok(Tensor::from_typed::<i64>(typed_sign_with_pool(
            buffers,
            ctx,
            input.as_typed::<i64>().ok_or_else(|| {
                unary_dtype_error("sign", input.dtype(), "F32/F64/I32/I64/C32/C64", false)
            })?,
        )?)),
        DType::Bool | DType::External(_) => Err(unary_dtype_error(
            "sign",
            input.dtype(),
            "F32/F64/I32/I64/C32/C64",
            false,
        )),
        DType::C32 => Ok(Tensor::from_typed::<Complex<f32>>(typed_sign_with_pool(
            buffers,
            ctx,
            input.as_typed::<Complex<f32>>().ok_or_else(|| {
                unary_dtype_error("sign", input.dtype(), "F32/F64/I32/I64/C32/C64", false)
            })?,
        )?)),
        DType::C64 => Ok(Tensor::from_typed::<Complex<f64>>(typed_sign_with_pool(
            buffers,
            ctx,
            input.as_typed::<Complex<f64>>().ok_or_else(|| {
                unary_dtype_error("sign", input.dtype(), "F32/F64/I32/I64/C32/C64", false)
            })?,
        )?)),
    }
}

#[doc(hidden)]
pub fn sign_read_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    input: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let dtype = input.dtype();
    match read_as_cpu_view(input)? {
        CpuReadView::F32(t) => Ok(Tensor::from_typed::<f32>(typed_sign_with_pool(
            buffers, ctx, &t,
        )?)),
        CpuReadView::F64(t) => Ok(Tensor::from_typed::<f64>(typed_sign_with_pool(
            buffers, ctx, &t,
        )?)),
        CpuReadView::I32(t) => Ok(Tensor::from_typed::<i32>(typed_sign_with_pool(
            buffers, ctx, &t,
        )?)),
        CpuReadView::I64(t) => Ok(Tensor::from_typed::<i64>(typed_sign_with_pool(
            buffers, ctx, &t,
        )?)),
        CpuReadView::C32(t) => Ok(Tensor::from_typed::<Complex<f32>>(typed_sign_with_pool(
            buffers, ctx, &t,
        )?)),
        CpuReadView::C64(t) => Ok(Tensor::from_typed::<Complex<f64>>(typed_sign_with_pool(
            buffers, ctx, &t,
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
    with_test_pool(|buffers, ctx| maximum_with_pool(buffers, ctx, lhs, rhs))
}

#[doc(hidden)]
pub fn maximum_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    lhs: &Tensor,
    rhs: &Tensor,
) -> crate::Result<Tensor> {
    reject_complex_ordered_dtypes("maximum", &[lhs.dtype(), rhs.dtype()])?;

    match (lhs.dtype(), rhs.dtype()) {
        (DType::F32, DType::F32) => {
            same_dtype_binary!(
                buffers,
                ctx,
                lhs,
                rhs,
                f32,
                typed_maximum_with_pool,
                "maximum"
            )
        }
        (DType::F64, DType::F64) => {
            same_dtype_binary!(
                buffers,
                ctx,
                lhs,
                rhs,
                f64,
                typed_maximum_with_pool,
                "maximum"
            )
        }
        (DType::I32, DType::I32) => {
            same_dtype_binary!(
                buffers,
                ctx,
                lhs,
                rhs,
                i32,
                typed_maximum_with_pool,
                "maximum"
            )
        }
        (DType::I64, DType::I64) => {
            same_dtype_binary!(
                buffers,
                ctx,
                lhs,
                rhs,
                i64,
                typed_maximum_with_pool,
                "maximum"
            )
        }
        _ => Err(tensor_pair_error("maximum", lhs, rhs)),
    }
}

#[doc(hidden)]
pub fn maximum_read_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let lhs_dtype = lhs.dtype();
    let rhs_dtype = rhs.dtype();
    reject_complex_ordered_dtypes("maximum", &[lhs_dtype, rhs_dtype])?;

    match (read_as_cpu_view(lhs)?, read_as_cpu_view(rhs)?) {
        (CpuReadView::F32(a), CpuReadView::F32(b)) => Ok(Tensor::from_typed::<f32>(
            typed_maximum_with_pool(buffers, ctx, &a, &b)?,
        )),
        (CpuReadView::F64(a), CpuReadView::F64(b)) => Ok(Tensor::from_typed::<f64>(
            typed_maximum_with_pool(buffers, ctx, &a, &b)?,
        )),
        (CpuReadView::I32(a), CpuReadView::I32(b)) => Ok(Tensor::from_typed::<i32>(
            typed_maximum_with_pool(buffers, ctx, &a, &b)?,
        )),
        (CpuReadView::I64(a), CpuReadView::I64(b)) => Ok(Tensor::from_typed::<i64>(
            typed_maximum_with_pool(buffers, ctx, &a, &b)?,
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
    with_test_pool(|buffers, ctx| minimum_with_pool(buffers, ctx, lhs, rhs))
}

#[doc(hidden)]
pub fn minimum_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    lhs: &Tensor,
    rhs: &Tensor,
) -> crate::Result<Tensor> {
    reject_complex_ordered_dtypes("minimum", &[lhs.dtype(), rhs.dtype()])?;

    match (lhs.dtype(), rhs.dtype()) {
        (DType::F32, DType::F32) => {
            same_dtype_binary!(
                buffers,
                ctx,
                lhs,
                rhs,
                f32,
                typed_minimum_with_pool,
                "minimum"
            )
        }
        (DType::F64, DType::F64) => {
            same_dtype_binary!(
                buffers,
                ctx,
                lhs,
                rhs,
                f64,
                typed_minimum_with_pool,
                "minimum"
            )
        }
        (DType::I32, DType::I32) => {
            same_dtype_binary!(
                buffers,
                ctx,
                lhs,
                rhs,
                i32,
                typed_minimum_with_pool,
                "minimum"
            )
        }
        (DType::I64, DType::I64) => {
            same_dtype_binary!(
                buffers,
                ctx,
                lhs,
                rhs,
                i64,
                typed_minimum_with_pool,
                "minimum"
            )
        }
        _ => Err(tensor_pair_error("minimum", lhs, rhs)),
    }
}

#[doc(hidden)]
pub fn minimum_read_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let lhs_dtype = lhs.dtype();
    let rhs_dtype = rhs.dtype();
    reject_complex_ordered_dtypes("minimum", &[lhs_dtype, rhs_dtype])?;

    match (read_as_cpu_view(lhs)?, read_as_cpu_view(rhs)?) {
        (CpuReadView::F32(a), CpuReadView::F32(b)) => Ok(Tensor::from_typed::<f32>(
            typed_minimum_with_pool(buffers, ctx, &a, &b)?,
        )),
        (CpuReadView::F64(a), CpuReadView::F64(b)) => Ok(Tensor::from_typed::<f64>(
            typed_minimum_with_pool(buffers, ctx, &a, &b)?,
        )),
        (CpuReadView::I32(a), CpuReadView::I32(b)) => Ok(Tensor::from_typed::<i32>(
            typed_minimum_with_pool(buffers, ctx, &a, &b)?,
        )),
        (CpuReadView::I64(a), CpuReadView::I64(b)) => Ok(Tensor::from_typed::<i64>(
            typed_minimum_with_pool(buffers, ctx, &a, &b)?,
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
    with_test_pool(|buffers, ctx| compare_with_pool(buffers, ctx, lhs, rhs, dir))
}

#[doc(hidden)]
pub fn compare_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    lhs: &Tensor,
    rhs: &Tensor,
    dir: &CompareDir,
) -> crate::Result<Tensor> {
    reject_complex_unsupported_compare_dtypes(dir, &[lhs.dtype(), rhs.dtype()])?;

    match (lhs.dtype(), rhs.dtype()) {
        (DType::F32, DType::F32) => Ok(Tensor::from_typed::<bool>(typed_compare_with_pool(
            buffers,
            ctx,
            &pair_operand::<f32>("compare", lhs, rhs, lhs)?.as_view(),
            &pair_operand::<f32>("compare", lhs, rhs, rhs)?.as_view(),
            dir,
        )?)),
        (DType::F64, DType::F64) => Ok(Tensor::from_typed::<bool>(typed_compare_with_pool(
            buffers,
            ctx,
            &pair_operand::<f64>("compare", lhs, rhs, lhs)?.as_view(),
            &pair_operand::<f64>("compare", lhs, rhs, rhs)?.as_view(),
            dir,
        )?)),
        (DType::I32, DType::I32) => Ok(Tensor::from_typed::<bool>(typed_compare_with_pool(
            buffers,
            ctx,
            &pair_operand::<i32>("compare", lhs, rhs, lhs)?.as_view(),
            &pair_operand::<i32>("compare", lhs, rhs, rhs)?.as_view(),
            dir,
        )?)),
        (DType::I64, DType::I64) => Ok(Tensor::from_typed::<bool>(typed_compare_with_pool(
            buffers,
            ctx,
            &pair_operand::<i64>("compare", lhs, rhs, lhs)?.as_view(),
            &pair_operand::<i64>("compare", lhs, rhs, rhs)?.as_view(),
            dir,
        )?)),
        (DType::Bool, DType::Bool) => Ok(Tensor::from_typed::<bool>(typed_compare_with_pool(
            buffers,
            ctx,
            &pair_operand::<bool>("compare", lhs, rhs, lhs)?.as_view(),
            &pair_operand::<bool>("compare", lhs, rhs, rhs)?.as_view(),
            dir,
        )?)),
        (DType::C32, DType::C32) => Ok(Tensor::from_typed::<bool>(typed_compare_with_pool(
            buffers,
            ctx,
            pair_operand::<Complex<f32>>("compare", lhs, rhs, lhs)?,
            pair_operand::<Complex<f32>>("compare", lhs, rhs, rhs)?,
            dir,
        )?)),
        (DType::C64, DType::C64) => Ok(Tensor::from_typed::<bool>(typed_compare_with_pool(
            buffers,
            ctx,
            pair_operand::<Complex<f64>>("compare", lhs, rhs, lhs)?,
            pair_operand::<Complex<f64>>("compare", lhs, rhs, rhs)?,
            dir,
        )?)),
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
    ctx: &ExecContext,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
    dir: &CompareDir,
) -> crate::Result<Tensor> {
    let lhs_dtype = lhs.dtype();
    let rhs_dtype = rhs.dtype();
    reject_complex_unsupported_compare_dtypes(dir, &[lhs_dtype, rhs_dtype])?;

    match (read_as_cpu_view(lhs)?, read_as_cpu_view(rhs)?) {
        (CpuReadView::F32(a), CpuReadView::F32(b)) => Ok(Tensor::from_typed::<bool>(
            typed_compare_with_pool(buffers, ctx, &a, &b, dir)?,
        )),
        (CpuReadView::F64(a), CpuReadView::F64(b)) => Ok(Tensor::from_typed::<bool>(
            typed_compare_with_pool(buffers, ctx, &a, &b, dir)?,
        )),
        (CpuReadView::I32(a), CpuReadView::I32(b)) => Ok(Tensor::from_typed::<bool>(
            typed_compare_with_pool(buffers, ctx, &a, &b, dir)?,
        )),
        (CpuReadView::I64(a), CpuReadView::I64(b)) => Ok(Tensor::from_typed::<bool>(
            typed_compare_with_pool(buffers, ctx, &a, &b, dir)?,
        )),
        (CpuReadView::Bool(a), CpuReadView::Bool(b)) => Ok(Tensor::from_typed::<bool>(
            typed_compare_with_pool(buffers, ctx, &a, &b, dir)?,
        )),
        (CpuReadView::C32(a), CpuReadView::C32(b)) => Ok(Tensor::from_typed::<bool>(
            typed_compare_with_pool(buffers, ctx, &a, &b, dir)?,
        )),
        (CpuReadView::C64(a), CpuReadView::C64(b)) => Ok(Tensor::from_typed::<bool>(
            typed_compare_with_pool(buffers, ctx, &a, &b, dir)?,
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
    with_test_pool(|buffers, ctx| select_with_pool(buffers, ctx, pred, on_true, on_false))
}

#[doc(hidden)]
pub fn select_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    pred: &Tensor,
    on_true: &Tensor,
    on_false: &Tensor,
) -> crate::Result<Tensor> {
    match (pred.dtype(), on_true.dtype(), on_false.dtype()) {
        (DType::Bool, DType::F32, DType::F32) => {
            let (p, t, f) = select_operands::<f32>("select", pred, on_true, on_false)?;
            Ok(Tensor::from_typed::<f32>(typed_select_with_pool(
                buffers, ctx, p, t, f,
            )?))
        }
        (DType::Bool, DType::F64, DType::F64) => {
            let (p, t, f) = select_operands::<f64>("select", pred, on_true, on_false)?;
            Ok(Tensor::from_typed::<f64>(typed_select_with_pool(
                buffers, ctx, p, t, f,
            )?))
        }
        (DType::Bool, DType::I32, DType::I32) => {
            let (p, t, f) = select_operands::<i32>("select", pred, on_true, on_false)?;
            Ok(Tensor::from_typed::<i32>(typed_select_with_pool(
                buffers, ctx, p, t, f,
            )?))
        }
        (DType::Bool, DType::I64, DType::I64) => {
            let (p, t, f) = select_operands::<i64>("select", pred, on_true, on_false)?;
            Ok(Tensor::from_typed::<i64>(typed_select_with_pool(
                buffers, ctx, p, t, f,
            )?))
        }
        (DType::Bool, DType::Bool, DType::Bool) => {
            let (p, t, f) = select_operands::<bool>("select", pred, on_true, on_false)?;
            Ok(Tensor::from_typed::<bool>(typed_select_with_pool(
                buffers, ctx, p, t, f,
            )?))
        }
        (DType::Bool, DType::C32, DType::C32) => {
            let (p, t, f) = select_operands::<Complex<f32>>("select", pred, on_true, on_false)?;
            Ok(Tensor::from_typed::<Complex<f32>>(typed_select_with_pool(
                buffers, ctx, p, t, f,
            )?))
        }
        (DType::Bool, DType::C64, DType::C64) => {
            let (p, t, f) = select_operands::<Complex<f64>>("select", pred, on_true, on_false)?;
            Ok(Tensor::from_typed::<Complex<f64>>(typed_select_with_pool(
                buffers, ctx, p, t, f,
            )?))
        }
        (DType::Bool, _, _) => Err(crate::Error::dtype_mismatch(
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

/// The predicate and value pair behind a select, or the refusal this table reports.
///
/// Callers reach this from a match on the three dtypes, so `None` means the tags and the runtime
/// dtypes disagree rather than a caller mistake.
fn select_operands<'a, T: TensorScalar>(
    op: &'static str,
    pred: &'a Tensor,
    on_true: &'a Tensor,
    on_false: &'a Tensor,
) -> crate::Result<(
    &'a TypedTensor<bool>,
    &'a TypedTensor<T>,
    &'a TypedTensor<T>,
)> {
    let p = pred
        .as_typed::<bool>()
        .ok_or_else(|| select_error(op, pred, on_true, on_false))?;
    let t = on_true
        .as_typed::<T>()
        .ok_or_else(|| select_error(op, pred, on_true, on_false))?;
    let f = on_false
        .as_typed::<T>()
        .ok_or_else(|| select_error(op, pred, on_true, on_false))?;
    Ok((p, t, f))
}

/// The refusal the select table reports, which is the pair's or the predicate's depending on which
/// arm would have answered.
fn select_error(
    op: &'static str,
    pred: &Tensor,
    on_true: &Tensor,
    on_false: &Tensor,
) -> crate::Error {
    if pred.dtype() != DType::Bool {
        crate::Error::dtype_mismatch(op, pred.dtype(), DType::Bool)
    } else {
        crate::Error::dtype_mismatch(op, on_true.dtype(), on_false.dtype())
    }
}

#[doc(hidden)]
pub fn select_read_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    pred: TensorRead<'_>,
    on_true: TensorRead<'_>,
    on_false: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let pred_dtype = pred.dtype();
    let true_dtype = on_true.dtype();
    let false_dtype = on_false.dtype();
    match (
        read_as_cpu_view(pred)?,
        read_as_cpu_view(on_true)?,
        read_as_cpu_view(on_false)?,
    ) {
        (CpuReadView::Bool(p), CpuReadView::F32(t), CpuReadView::F32(f)) => Ok(
            Tensor::from_typed::<f32>(typed_select_with_pool(buffers, ctx, &p, &t, &f)?),
        ),
        (CpuReadView::Bool(p), CpuReadView::F64(t), CpuReadView::F64(f)) => Ok(
            Tensor::from_typed::<f64>(typed_select_with_pool(buffers, ctx, &p, &t, &f)?),
        ),
        (CpuReadView::Bool(p), CpuReadView::I32(t), CpuReadView::I32(f)) => Ok(
            Tensor::from_typed::<i32>(typed_select_with_pool(buffers, ctx, &p, &t, &f)?),
        ),
        (CpuReadView::Bool(p), CpuReadView::I64(t), CpuReadView::I64(f)) => Ok(
            Tensor::from_typed::<i64>(typed_select_with_pool(buffers, ctx, &p, &t, &f)?),
        ),
        (CpuReadView::Bool(p), CpuReadView::Bool(t), CpuReadView::Bool(f)) => Ok(
            Tensor::from_typed::<bool>(typed_select_with_pool(buffers, ctx, &p, &t, &f)?),
        ),
        (CpuReadView::Bool(p), CpuReadView::C32(t), CpuReadView::C32(f)) => Ok(
            Tensor::from_typed::<Complex<f32>>(typed_select_with_pool(buffers, ctx, &p, &t, &f)?),
        ),
        (CpuReadView::Bool(p), CpuReadView::C64(t), CpuReadView::C64(f)) => Ok(
            Tensor::from_typed::<Complex<f64>>(typed_select_with_pool(buffers, ctx, &p, &t, &f)?),
        ),
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
    with_test_pool(|buffers, ctx| clamp_with_pool(buffers, ctx, input, lower, upper))
}

#[doc(hidden)]
pub fn clamp_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    input: &Tensor,
    lower: &Tensor,
    upper: &Tensor,
) -> crate::Result<Tensor> {
    reject_complex_ordered_dtypes("clamp", &[input.dtype(), lower.dtype(), upper.dtype()])?;

    dispatch_ternary_result_with_pool!("clamp", input, lower, upper, |x, lo, hi| {
        typed_clamp_with_pool(buffers, ctx, x, lo, hi)
    })
}

#[doc(hidden)]
pub fn clamp_read_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    input: TensorRead<'_>,
    lower: TensorRead<'_>,
    upper: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let input_dtype = input.dtype();
    let lower_dtype = lower.dtype();
    let upper_dtype = upper.dtype();
    reject_complex_ordered_dtypes("clamp", &[input_dtype, lower_dtype, upper_dtype])?;

    match (
        read_as_cpu_view(input)?,
        read_as_cpu_view(lower)?,
        read_as_cpu_view(upper)?,
    ) {
        (CpuReadView::F32(input), CpuReadView::F32(lower), CpuReadView::F32(upper)) => Ok(
            Tensor::from_typed::<f32>(typed_clamp_with_pool(buffers, ctx, &input, &lower, &upper)?),
        ),
        (CpuReadView::F64(input), CpuReadView::F64(lower), CpuReadView::F64(upper)) => Ok(
            Tensor::from_typed::<f64>(typed_clamp_with_pool(buffers, ctx, &input, &lower, &upper)?),
        ),
        _ => Err(crate::Error::dtype_mismatch(
            "clamp",
            input_dtype,
            lower_dtype,
        )),
    }
}

#[cfg(test)]
mod tests;
