use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use chainrules_core::Differentiable as _;
use num_complex::Complex;
use num_traits::Float;
use tenferro_algebra::{HasAlgebra, Scalar, Standard};
use tenferro_einsum as tf_einsum;
use tenferro_linalg::backend::CpuLinalgScalar;
use tenferro_linalg::{
    EigResult, EigenResult, LinalgScalar, LstsqResult, LuPivot, LuResult, NormKind, QrResult,
    SlogdetResult, SvdOptions, SvdResult,
};
use tenferro_prims::{CpuBackend, CpuContext, TensorPrims};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::ad_value::{AdValue, NodeId};
use crate::runtime::{with_default_runtime, RuntimeContext};
use crate::{AdTensor, Error, Result, TapeId};

pub mod ad;
mod ad_results;

pub use ad_results::{
    AdEigResult, AdEigenResult, AdLstsqResult, AdLuResult, AdQrResult, AdSlogdetResult, AdSvdResult,
};

fn with_cpu_runtime<R>(
    op: &'static str,
    f: impl FnOnce(&mut CpuContext) -> Result<R>,
) -> Result<R> {
    with_default_runtime(|runtime| match runtime {
        RuntimeContext::Cpu(ctx) => f(ctx),
        RuntimeContext::Cuda(_) => Err(Error::UnsupportedRuntimeOp {
            op,
            runtime: "cuda",
        }),
        RuntimeContext::Rocm(_) => Err(Error::UnsupportedRuntimeOp {
            op,
            runtime: "rocm",
        }),
    })
}

fn has_forward<S: Scalar>(operands: &[&AdTensor<S>]) -> bool {
    operands
        .iter()
        .any(|op| matches!(op.as_value(), AdValue::Forward { .. }))
}

fn has_reverse<S: Scalar>(operands: &[&AdTensor<S>]) -> bool {
    operands
        .iter()
        .any(|op| matches!(op.as_value(), AdValue::Reverse { .. }))
}

fn has_any_tangent<S: Scalar>(operands: &[&AdTensor<S>]) -> bool {
    operands.iter().any(|op| op.tangent().is_some())
}

fn derive_reverse_tape<S: Scalar>(operands: &[&AdTensor<S>]) -> Result<Option<TapeId>> {
    let mut tape: Option<TapeId> = None;

    for op in operands {
        if let AdValue::Reverse { tape: current, .. } = op.as_value() {
            if let Some(expected) = tape {
                if expected != *current {
                    return Err(Error::MixedReverseTape {
                        expected: expected.0,
                        found: current.0,
                    });
                }
            } else {
                tape = Some(*current);
            }
        }
    }

    Ok(tape)
}

fn derive_reverse_node<S: Scalar>(
    op_name: &str,
    operands: &[&AdTensor<S>],
    output_dims: &[usize],
    output_tag: u64,
    tape: TapeId,
) -> NodeId {
    let mut hasher = DefaultHasher::new();
    op_name.hash(&mut hasher);
    output_dims.hash(&mut hasher);
    output_tag.hash(&mut hasher);
    tape.0.hash(&mut hasher);

    for (idx, op) in operands.iter().enumerate() {
        idx.hash(&mut hasher);
        match op.as_value() {
            AdValue::Primal(_) => {
                0_u8.hash(&mut hasher);
            }
            AdValue::Forward { .. } => {
                1_u8.hash(&mut hasher);
            }
            AdValue::Reverse { node, .. } => {
                2_u8.hash(&mut hasher);
                node.0.hash(&mut hasher);
            }
        }
        op.dims().hash(&mut hasher);
    }

    NodeId(hasher.finish())
}

fn zero_like<T: Scalar>(tensor: &Tensor<T>) -> Tensor<T> {
    Tensor::zeros(
        tensor.dims(),
        tensor.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    )
}

fn wrap_ad_output<TIn: Scalar, TOut: Scalar>(
    op_name: &'static str,
    inputs: &[&AdTensor<TIn>],
    primal: Tensor<TOut>,
    tangent: Option<Tensor<TOut>>,
    output_tag: u64,
) -> Result<AdTensor<TOut>> {
    if has_reverse(inputs) {
        let tape = derive_reverse_tape(inputs)?.ok_or_else(|| Error::InvalidAdTensor {
            message: "reverse-mode output requested but no reverse tape found".to_string(),
        })?;
        let node = derive_reverse_node(op_name, inputs, primal.dims(), output_tag, tape);
        return Ok(AdTensor::new_reverse(primal, node, tape, tangent));
    }

    if has_forward(inputs) {
        let tangent = tangent.ok_or_else(|| Error::InvalidAdTensor {
            message: "forward-mode inputs must provide tangent output".to_string(),
        })?;
        return Ok(AdTensor::new_forward(primal, tangent));
    }

    Ok(AdTensor::new_primal(primal))
}

fn collect_ad_tangents<'a, S: Scalar>(operands: &[&'a AdTensor<S>]) -> Vec<Option<&'a Tensor<S>>> {
    operands
        .iter()
        .map(|op| match op.as_value() {
            AdValue::Primal(_) => None,
            AdValue::Forward { tangent, .. } => Some(tangent),
            AdValue::Reverse { tangent, .. } => tangent.as_ref(),
        })
        .collect()
}

fn sum_einsum_tangent_terms<T>(
    ctx: &mut CpuContext,
    subscripts: &str,
    primals: &[&Tensor<T>],
    tangents: &[Option<&Tensor<T>>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Option<Tensor<T>>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
{
    let mut out_tangent: Option<Tensor<T>> = None;

    for (k, tangent_opt) in tangents.iter().enumerate() {
        let Some(tangent_k) = tangent_opt else {
            continue;
        };

        let mut term_operands: Vec<&Tensor<T>> = primals.to_vec();
        term_operands[k] = tangent_k;
        let term = tf_einsum::einsum::<Standard<T>, CpuBackend>(
            ctx,
            subscripts,
            &term_operands,
            size_dict,
        )
        .map_err(Error::from)?;

        out_tangent = Some(match out_tangent {
            None => term,
            Some(existing) => Tensor::<T>::accumulate_tangent(existing, &term),
        });
    }

    Ok(out_tangent)
}

/// Builder for primal einsum.
/// # Examples
///
/// ```ignore
/// // Construct `EinsumBuilder` via its corresponding operation constructor.
/// ```
pub struct EinsumBuilder<'a, T>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
{
    subscripts: &'a str,
    operands: &'a [&'a Tensor<T>],
    size_dict: Option<&'a HashMap<u32, usize>>,
}

impl<'a, T> EinsumBuilder<'a, T>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
{
    /// Sets optional size dictionary for output-only labels.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.size_dict(&size_dict);
    /// ```
    pub fn size_dict(mut self, size_dict: &'a HashMap<u32, usize>) -> Self {
        self.size_dict = Some(size_dict);
        self
    }

    /// Executes the operation with the default runtime.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_cpu_runtime("einsum", |ctx| {
            tf_einsum::einsum::<Standard<T>, CpuBackend>(
                ctx,
                self.subscripts,
                self.operands,
                self.size_dict,
            )
            .map_err(Error::from)
        })
    }
}

/// Creates a builder for primal einsum.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{einsum, set_default_runtime, RuntimeContext};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
/// let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
/// let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
/// let out = einsum("ij,jk->ik", &[&a, &b]).run().unwrap();
/// assert_eq!(out.dims(), &[2, 2]);
/// ```
pub fn einsum<'a, T>(subscripts: &'a str, operands: &'a [&'a Tensor<T>]) -> EinsumBuilder<'a, T>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
{
    EinsumBuilder {
        subscripts,
        operands,
        size_dict: None,
    }
}

/// Builder for AD einsum.
/// # Examples
///
/// ```ignore
/// // Construct `EinsumAdBuilder` via its corresponding operation constructor.
/// ```
pub struct EinsumAdBuilder<'a, T>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
{
    subscripts: &'a str,
    operands: &'a [&'a AdTensor<T>],
    size_dict: Option<&'a HashMap<u32, usize>>,
}

impl<'a, T> EinsumAdBuilder<'a, T>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
{
    /// Sets optional size dictionary for output-only labels.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.size_dict(&size_dict);
    /// ```
    pub fn size_dict(mut self, size_dict: &'a HashMap<u32, usize>) -> Self {
        self.size_dict = Some(size_dict);
        self
    }

    /// Executes AD einsum with mode propagation.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        with_cpu_runtime("einsum_ad", |ctx| {
            let primals: Vec<&Tensor<T>> = self.operands.iter().map(|op| op.primal()).collect();
            let primal_out = tf_einsum::einsum::<Standard<T>, CpuBackend>(
                ctx,
                self.subscripts,
                &primals,
                self.size_dict,
            )
            .map_err(Error::from)?;

            let tangents = collect_ad_tangents(self.operands);
            let tangent_out = if has_forward(self.operands) || has_any_tangent(self.operands) {
                sum_einsum_tangent_terms(ctx, self.subscripts, &primals, &tangents, self.size_dict)?
            } else {
                None
            };

            wrap_ad_output("einsum_ad", self.operands, primal_out, tangent_out, 0)
        })
    }
}

/// Creates a builder for AD einsum.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{einsum_ad, set_default_runtime, AdTensor, RuntimeContext};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
/// let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
/// let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
/// let ad_a = AdTensor::new_primal(a);
/// let ad_b = AdTensor::new_primal(b);
/// let out = einsum_ad("ij,jk->ik", &[&ad_a, &ad_b]).run().unwrap();
/// assert_eq!(out.dims(), &[2, 2]);
/// ```
pub fn einsum_ad<'a, T>(
    subscripts: &'a str,
    operands: &'a [&'a AdTensor<T>],
) -> EinsumAdBuilder<'a, T>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
{
    EinsumAdBuilder {
        subscripts,
        operands,
        size_dict: None,
    }
}

/// Builder for SVD.
/// # Examples
///
/// ```ignore
/// // Construct `SvdBuilder` via its corresponding operation constructor.
/// ```
pub struct SvdBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
    options: Option<&'a SvdOptions>,
}

impl<'a, T> SvdBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Sets optional SVD options.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.options(&options);
    /// ```
    pub fn options(mut self, options: &'a SvdOptions) -> Self {
        self.options = Some(options);
        self
    }

    /// Executes SVD.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<SvdResult<T, T::Real>> {
        with_cpu_runtime("svd", |ctx| {
            tenferro_linalg::svd::<T, CpuContext>(ctx, self.tensor, self.options)
                .map_err(Error::from)
        })
    }
}

/// Creates an SVD builder.
/// # Examples
///
/// ```ignore
/// let _ = svd(/* ... */);
/// ```
pub fn svd<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> SvdBuilder<'a, T> {
    SvdBuilder {
        tensor,
        options: None,
    }
}

/// Builder for QR.
/// # Examples
///
/// ```ignore
/// // Construct `QrBuilder` via its corresponding operation constructor.
/// ```
pub struct QrBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> QrBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes QR.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<QrResult<T>> {
        with_cpu_runtime("qr", |ctx| {
            tenferro_linalg::qr::<T, CpuContext>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates a QR builder.
/// # Examples
///
/// ```ignore
/// let _ = qr(/* ... */);
/// ```
pub fn qr<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> QrBuilder<'a, T> {
    QrBuilder { tensor }
}

/// Builder for LU.
/// # Examples
///
/// ```ignore
/// // Construct `LuBuilder` via its corresponding operation constructor.
/// ```
pub struct LuBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
    pivot: LuPivot,
}

impl<'a, T> LuBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Sets LU pivoting policy.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.pivot(pivot);
    /// ```
    pub fn pivot(mut self, pivot: LuPivot) -> Self {
        self.pivot = pivot;
        self
    }

    /// Executes LU.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<LuResult<T>> {
        with_cpu_runtime("lu", |ctx| {
            tenferro_linalg::lu::<T, CpuContext>(ctx, self.tensor, self.pivot).map_err(Error::from)
        })
    }
}

/// Creates an LU builder.
/// # Examples
///
/// ```ignore
/// let _ = lu(/* ... */);
/// ```
pub fn lu<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> LuBuilder<'a, T> {
    LuBuilder {
        tensor,
        pivot: LuPivot::Partial,
    }
}

/// Builder for eigen decomposition.
/// # Examples
///
/// ```ignore
/// // Construct `EigenBuilder` via its corresponding operation constructor.
/// ```
pub struct EigenBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> EigenBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes eigen decomposition.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<EigenResult<T, T::Real>> {
        with_cpu_runtime("eigen", |ctx| {
            tenferro_linalg::eigen::<T, CpuContext>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates an eigen builder.
/// # Examples
///
/// ```ignore
/// let _ = eigen(/* ... */);
/// ```
pub fn eigen<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> EigenBuilder<'a, T> {
    EigenBuilder { tensor }
}

/// Builder for least squares solve.
/// # Examples
///
/// ```ignore
/// // Construct `LstsqBuilder` via its corresponding operation constructor.
/// ```
pub struct LstsqBuilder<'a, T: LinalgScalar> {
    a: &'a Tensor<T>,
    b: &'a Tensor<T>,
}

impl<'a, T> LstsqBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes least squares.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<LstsqResult<T>> {
        with_cpu_runtime("lstsq", |ctx| {
            tenferro_linalg::lstsq::<T, CpuContext>(ctx, self.a, self.b).map_err(Error::from)
        })
    }
}

/// Creates an lstsq builder.
/// # Examples
///
/// ```ignore
/// let _ = lstsq(/* ... */);
/// ```
pub fn lstsq<'a, T: LinalgScalar>(a: &'a Tensor<T>, b: &'a Tensor<T>) -> LstsqBuilder<'a, T> {
    LstsqBuilder { a, b }
}

/// Builder for Cholesky.
/// # Examples
///
/// ```ignore
/// // Construct `CholeskyBuilder` via its corresponding operation constructor.
/// ```
pub struct CholeskyBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> CholeskyBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes Cholesky.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_cpu_runtime("cholesky", |ctx| {
            tenferro_linalg::cholesky::<T, CpuContext>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates a cholesky builder.
/// # Examples
///
/// ```ignore
/// let _ = cholesky(/* ... */);
/// ```
pub fn cholesky<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> CholeskyBuilder<'a, T> {
    CholeskyBuilder { tensor }
}

/// Builder for solve.
/// # Examples
///
/// ```ignore
/// // Construct `SolveBuilder` via its corresponding operation constructor.
/// ```
pub struct SolveBuilder<'a, T: LinalgScalar> {
    a: &'a Tensor<T>,
    b: &'a Tensor<T>,
}

impl<'a, T> SolveBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes linear solve.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_cpu_runtime("solve", |ctx| {
            tenferro_linalg::solve::<T, CpuContext>(ctx, self.a, self.b).map_err(Error::from)
        })
    }
}

/// Creates a solve builder.
/// # Examples
///
/// ```ignore
/// let _ = solve(/* ... */);
/// ```
pub fn solve<'a, T: LinalgScalar>(a: &'a Tensor<T>, b: &'a Tensor<T>) -> SolveBuilder<'a, T> {
    SolveBuilder { a, b }
}

/// Builder for matrix inverse.
/// # Examples
///
/// ```ignore
/// // Construct `InvBuilder` via its corresponding operation constructor.
/// ```
pub struct InvBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> InvBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes matrix inverse.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_cpu_runtime("inv", |ctx| {
            tenferro_linalg::inv::<T, CpuContext>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates an inv builder.
/// # Examples
///
/// ```ignore
/// let _ = inv(/* ... */);
/// ```
pub fn inv<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> InvBuilder<'a, T> {
    InvBuilder { tensor }
}

/// Builder for determinant.
/// # Examples
///
/// ```ignore
/// // Construct `DetBuilder` via its corresponding operation constructor.
/// ```
pub struct DetBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> DetBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Executes determinant.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_cpu_runtime("det", |ctx| {
            tenferro_linalg::det::<T, CpuContext>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates a det builder.
/// # Examples
///
/// ```ignore
/// let _ = det(/* ... */);
/// ```
pub fn det<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> DetBuilder<'a, T> {
    DetBuilder { tensor }
}

/// Builder for slogdet.
/// # Examples
///
/// ```ignore
/// // Construct `SlogdetBuilder` via its corresponding operation constructor.
/// ```
pub struct SlogdetBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> SlogdetBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Executes slogdet.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<SlogdetResult<T>> {
        with_cpu_runtime("slogdet", |ctx| {
            tenferro_linalg::slogdet::<T, CpuContext>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates an slogdet builder.
/// # Examples
///
/// ```ignore
/// let _ = slogdet(/* ... */);
/// ```
pub fn slogdet<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> SlogdetBuilder<'a, T> {
    SlogdetBuilder { tensor }
}

/// Builder for general eigendecomposition.
/// # Examples
///
/// ```ignore
/// // Construct `EigBuilder` via its corresponding operation constructor.
/// ```
pub struct EigBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> EigBuilder<'a, T>
where
    T: LinalgScalar<Real = T, Complex = Complex<T>> + Float + CpuLinalgScalar,
{
    /// Executes eig.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<EigResult<T>> {
        with_cpu_runtime("eig", |ctx| {
            tenferro_linalg::eig::<T, CpuContext>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates an eig builder.
/// # Examples
///
/// ```ignore
/// let _ = eig(/* ... */);
/// ```
pub fn eig<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> EigBuilder<'a, T> {
    EigBuilder { tensor }
}

/// Builder for pseudoinverse.
/// # Examples
///
/// ```ignore
/// // Construct `PinvBuilder` via its corresponding operation constructor.
/// ```
pub struct PinvBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
    rcond: Option<f64>,
}

impl<'a, T> PinvBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Sets rcond.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.rcond(1e-12);
    /// ```
    pub fn rcond(mut self, rcond: f64) -> Self {
        self.rcond = Some(rcond);
        self
    }

    /// Executes pseudoinverse.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_cpu_runtime("pinv", |ctx| {
            tenferro_linalg::pinv::<T, CpuContext>(ctx, self.tensor, self.rcond)
                .map_err(Error::from)
        })
    }
}

/// Creates a pinv builder.
/// # Examples
///
/// ```ignore
/// let _ = pinv(/* ... */);
/// ```
pub fn pinv<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> PinvBuilder<'a, T> {
    PinvBuilder {
        tensor,
        rcond: None,
    }
}

/// Builder for matrix exponential.
/// # Examples
///
/// ```ignore
/// // Construct `MatrixExpBuilder` via its corresponding operation constructor.
/// ```
pub struct MatrixExpBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> MatrixExpBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes matrix exponential.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_cpu_runtime("matrix_exp", |ctx| {
            tenferro_linalg::matrix_exp::<T, CpuContext>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates a matrix_exp builder.
/// # Examples
///
/// ```ignore
/// let _ = matrix_exp(/* ... */);
/// ```
pub fn matrix_exp<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> MatrixExpBuilder<'a, T> {
    MatrixExpBuilder { tensor }
}

/// Builder for triangular solve.
/// # Examples
///
/// ```ignore
/// // Construct `SolveTriangularBuilder` via its corresponding operation constructor.
/// ```
pub struct SolveTriangularBuilder<'a, T: LinalgScalar> {
    a: &'a Tensor<T>,
    b: &'a Tensor<T>,
    upper: bool,
}

impl<'a, T> SolveTriangularBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Sets whether the matrix is upper triangular.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.upper(true);
    /// ```
    pub fn upper(mut self, upper: bool) -> Self {
        self.upper = upper;
        self
    }

    /// Executes triangular solve.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_cpu_runtime("solve_triangular", |ctx| {
            tenferro_linalg::solve_triangular::<T, CpuContext>(ctx, self.a, self.b, self.upper)
                .map_err(Error::from)
        })
    }
}

/// Creates a solve_triangular builder.
/// # Examples
///
/// ```ignore
/// let _ = solve_triangular(/* ... */);
/// ```
pub fn solve_triangular<'a, T: LinalgScalar>(
    a: &'a Tensor<T>,
    b: &'a Tensor<T>,
) -> SolveTriangularBuilder<'a, T> {
    SolveTriangularBuilder { a, b, upper: true }
}

/// Builder for norm.
/// # Examples
///
/// ```ignore
/// // Construct `NormBuilder` via its corresponding operation constructor.
/// ```
pub struct NormBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
    kind: NormKind,
}

impl<'a, T> NormBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Sets norm kind.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.kind(kind);
    /// ```
    pub fn kind(mut self, kind: NormKind) -> Self {
        self.kind = kind;
        self
    }

    /// Executes norm.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_cpu_runtime("norm", |ctx| {
            tenferro_linalg::norm::<T, CpuContext>(ctx, self.tensor, self.kind).map_err(Error::from)
        })
    }
}

/// Creates a norm builder.
/// # Examples
///
/// ```ignore
/// let _ = norm(/* ... */);
/// ```
pub fn norm<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> NormBuilder<'a, T> {
    NormBuilder {
        tensor,
        kind: NormKind::Fro,
    }
}

fn run_unary_tensor_ad<T, FPrimal, FFrule>(
    op_name: &'static str,
    input: &AdTensor<T>,
    primal_fn: FPrimal,
    frule_fn: FFrule,
) -> Result<AdTensor<T>>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
    FPrimal: FnOnce(&mut CpuContext, &Tensor<T>) -> Result<Tensor<T>>,
    FFrule: FnOnce(
        &mut CpuContext,
        &Tensor<T>,
        &Tensor<T>,
    )
        -> std::result::Result<(Tensor<T>, Tensor<T>), chainrules_core::AutodiffError>,
{
    let operands = [input];
    let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);

    let (primal, tangent) = if needs_tangent {
        let in_tangent = input
            .tangent()
            .cloned()
            .unwrap_or_else(|| zero_like(input.primal()));
        let (p, d) = with_cpu_runtime(op_name, |ctx| {
            frule_fn(ctx, input.primal(), &in_tangent).map_err(Error::from)
        })?;
        (p, Some(d))
    } else {
        (
            with_cpu_runtime(op_name, |ctx| primal_fn(ctx, input.primal()))?,
            None,
        )
    };

    wrap_ad_output(op_name, &operands, primal, tangent, 0)
}

fn run_binary_tensor_ad<T, FPrimal, FFrule>(
    op_name: &'static str,
    a: &AdTensor<T>,
    b: &AdTensor<T>,
    primal_fn: FPrimal,
    frule_fn: FFrule,
) -> Result<AdTensor<T>>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
    FPrimal: FnOnce(&mut CpuContext, &Tensor<T>, &Tensor<T>) -> Result<Tensor<T>>,
    FFrule: FnOnce(
        &mut CpuContext,
        &Tensor<T>,
        &Tensor<T>,
        &Tensor<T>,
        &Tensor<T>,
    )
        -> std::result::Result<(Tensor<T>, Tensor<T>), chainrules_core::AutodiffError>,
{
    let operands = [a, b];
    let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);

    let (primal, tangent) = if needs_tangent {
        let ta = a
            .tangent()
            .cloned()
            .unwrap_or_else(|| zero_like(a.primal()));
        let tb = b
            .tangent()
            .cloned()
            .unwrap_or_else(|| zero_like(b.primal()));
        let (p, d) = with_cpu_runtime(op_name, |ctx| {
            frule_fn(ctx, a.primal(), b.primal(), &ta, &tb).map_err(Error::from)
        })?;
        (p, Some(d))
    } else {
        (
            with_cpu_runtime(op_name, |ctx| primal_fn(ctx, a.primal(), b.primal()))?,
            None,
        )
    };

    wrap_ad_output(op_name, &operands, primal, tangent, 0)
}

/// Builder for AD SVD.
/// # Examples
///
/// ```ignore
/// // Construct `SvdAdBuilder` via its corresponding operation constructor.
/// ```
pub struct SvdAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
    options: Option<&'a SvdOptions>,
}

impl<'a, T> SvdAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Sets optional SVD options.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.options(&options);
    /// ```
    pub fn options(mut self, options: &'a SvdOptions) -> Self {
        self.options = Some(options);
        self
    }

    /// Executes AD SVD.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdSvdResult<T>> {
        let operands = [self.tensor];
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);

        let (primal, tangent) = if needs_tangent {
            let dt = self
                .tensor
                .tangent()
                .cloned()
                .unwrap_or_else(|| zero_like(self.tensor.primal()));
            let (p, d) = with_cpu_runtime("svd_ad", |ctx| {
                tenferro_linalg::svd_frule::<T>(ctx, self.tensor.primal(), &dt, self.options)
                    .map_err(Error::from)
            })?;
            (p, Some(d))
        } else {
            (
                with_cpu_runtime("svd_ad", |ctx| {
                    tenferro_linalg::svd::<T, CpuContext>(ctx, self.tensor.primal(), self.options)
                        .map_err(Error::from)
                })?,
                None,
            )
        };

        let (du, ds, dvt) = if let Some(d) = tangent {
            (Some(d.u), Some(d.s), Some(d.vt))
        } else {
            (None, None, None)
        };

        Ok(AdSvdResult {
            u: wrap_ad_output("svd_ad", &operands, primal.u, du, 1)?,
            s: wrap_ad_output("svd_ad", &operands, primal.s, ds, 2)?,
            vt: wrap_ad_output("svd_ad", &operands, primal.vt, dvt, 3)?,
        })
    }
}

/// Creates an AD SVD builder.
/// # Examples
///
/// ```ignore
/// let _ = svd_ad(/* ... */);
/// ```
pub fn svd_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> SvdAdBuilder<'a, T> {
    SvdAdBuilder {
        tensor,
        options: None,
    }
}

/// Builder for AD QR.
/// # Examples
///
/// ```ignore
/// // Construct `QrAdBuilder` via its corresponding operation constructor.
/// ```
pub struct QrAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> QrAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Executes AD QR.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdQrResult<T>> {
        let operands = [self.tensor];
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);

        let (primal, tangent) = if needs_tangent {
            let dt = self
                .tensor
                .tangent()
                .cloned()
                .unwrap_or_else(|| zero_like(self.tensor.primal()));
            let (p, d) = with_cpu_runtime("qr_ad", |ctx| {
                tenferro_linalg::qr_frule::<T>(ctx, self.tensor.primal(), &dt).map_err(Error::from)
            })?;
            (p, Some(d))
        } else {
            (
                with_cpu_runtime("qr_ad", |ctx| {
                    tenferro_linalg::qr::<T, CpuContext>(ctx, self.tensor.primal())
                        .map_err(Error::from)
                })?,
                None,
            )
        };

        let (dq, dr) = if let Some(d) = tangent {
            (Some(d.q), Some(d.r))
        } else {
            (None, None)
        };

        Ok(AdQrResult {
            q: wrap_ad_output("qr_ad", &operands, primal.q, dq, 1)?,
            r: wrap_ad_output("qr_ad", &operands, primal.r, dr, 2)?,
        })
    }
}

/// Creates an AD QR builder.
/// # Examples
///
/// ```ignore
/// let _ = qr_ad(/* ... */);
/// ```
pub fn qr_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> QrAdBuilder<'a, T> {
    QrAdBuilder { tensor }
}

/// Builder for AD LU.
/// # Examples
///
/// ```ignore
/// // Construct `LuAdBuilder` via its corresponding operation constructor.
/// ```
pub struct LuAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
    pivot: LuPivot,
}

impl<'a, T> LuAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Sets LU pivot policy.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.pivot(pivot);
    /// ```
    pub fn pivot(mut self, pivot: LuPivot) -> Self {
        self.pivot = pivot;
        self
    }

    /// Executes AD LU.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdLuResult<T>> {
        let operands = [self.tensor];
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);

        let (primal, tangent) = if needs_tangent {
            let dt = self
                .tensor
                .tangent()
                .cloned()
                .unwrap_or_else(|| zero_like(self.tensor.primal()));
            let (p, d) = with_cpu_runtime("lu_ad", |ctx| {
                tenferro_linalg::lu_frule::<T>(ctx, self.tensor.primal(), &dt, self.pivot)
                    .map_err(Error::from)
            })?;
            (p, Some(d))
        } else {
            (
                with_cpu_runtime("lu_ad", |ctx| {
                    tenferro_linalg::lu::<T, CpuContext>(ctx, self.tensor.primal(), self.pivot)
                        .map_err(Error::from)
                })?,
                None,
            )
        };

        let (dl, du) = if let Some(d) = tangent {
            (Some(d.l), Some(d.u))
        } else {
            (None, None)
        };

        Ok(AdLuResult {
            p: primal.p,
            l: wrap_ad_output("lu_ad", &operands, primal.l, dl, 1)?,
            u: wrap_ad_output("lu_ad", &operands, primal.u, du, 2)?,
        })
    }
}

/// Creates an AD LU builder.
/// # Examples
///
/// ```ignore
/// let _ = lu_ad(/* ... */);
/// ```
pub fn lu_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> LuAdBuilder<'a, T> {
    LuAdBuilder {
        tensor,
        pivot: LuPivot::Partial,
    }
}

/// Builder for AD eigen decomposition.
/// # Examples
///
/// ```ignore
/// // Construct `EigenAdBuilder` via its corresponding operation constructor.
/// ```
pub struct EigenAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> EigenAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Executes AD eigen decomposition.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdEigenResult<T>> {
        let operands = [self.tensor];
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);

        let (primal, tangent) = if needs_tangent {
            let dt = self
                .tensor
                .tangent()
                .cloned()
                .unwrap_or_else(|| zero_like(self.tensor.primal()));
            let (p, d) = with_cpu_runtime("eigen_ad", |ctx| {
                tenferro_linalg::eigen_frule::<T>(ctx, self.tensor.primal(), &dt)
                    .map_err(Error::from)
            })?;
            (p, Some(d))
        } else {
            (
                with_cpu_runtime("eigen_ad", |ctx| {
                    tenferro_linalg::eigen::<T, CpuContext>(ctx, self.tensor.primal())
                        .map_err(Error::from)
                })?,
                None,
            )
        };

        let (dvalues, dvectors) = if let Some(d) = tangent {
            (Some(d.values), Some(d.vectors))
        } else {
            (None, None)
        };

        Ok(AdEigenResult {
            values: wrap_ad_output("eigen_ad", &operands, primal.values, dvalues, 1)?,
            vectors: wrap_ad_output("eigen_ad", &operands, primal.vectors, dvectors, 2)?,
        })
    }
}

/// Creates an AD eigen builder.
/// # Examples
///
/// ```ignore
/// let _ = eigen_ad(/* ... */);
/// ```
pub fn eigen_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> EigenAdBuilder<'a, T> {
    EigenAdBuilder { tensor }
}

/// Builder for AD least squares.
/// # Examples
///
/// ```ignore
/// // Construct `LstsqAdBuilder` via its corresponding operation constructor.
/// ```
pub struct LstsqAdBuilder<'a, T: Scalar> {
    a: &'a AdTensor<T>,
    b: &'a AdTensor<T>,
}

impl<'a, T> LstsqAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Executes AD least squares.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdLstsqResult<T>> {
        let operands = [self.a, self.b];
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);

        let (primal, tangent) = if needs_tangent {
            let da = self
                .a
                .tangent()
                .cloned()
                .unwrap_or_else(|| zero_like(self.a.primal()));
            let db = self
                .b
                .tangent()
                .cloned()
                .unwrap_or_else(|| zero_like(self.b.primal()));
            let (p, d) = with_cpu_runtime("lstsq_ad", |ctx| {
                tenferro_linalg::lstsq_frule::<T, CpuContext>(
                    ctx,
                    self.a.primal(),
                    self.b.primal(),
                    &da,
                    &db,
                )
                .map_err(Error::from)
            })?;
            (p, Some(d))
        } else {
            (
                with_cpu_runtime("lstsq_ad", |ctx| {
                    tenferro_linalg::lstsq::<T, CpuContext>(ctx, self.a.primal(), self.b.primal())
                        .map_err(Error::from)
                })?,
                None,
            )
        };

        let (dx, dresidual) = if let Some(d) = tangent {
            (Some(d.x), Some(d.residual))
        } else {
            (None, None)
        };

        Ok(AdLstsqResult {
            x: wrap_ad_output("lstsq_ad", &operands, primal.x, dx, 1)?,
            residual: wrap_ad_output("lstsq_ad", &operands, primal.residual, dresidual, 2)?,
        })
    }
}

/// Creates an AD lstsq builder.
/// # Examples
///
/// ```ignore
/// let _ = lstsq_ad(/* ... */);
/// ```
pub fn lstsq_ad<'a, T: Scalar>(a: &'a AdTensor<T>, b: &'a AdTensor<T>) -> LstsqAdBuilder<'a, T> {
    LstsqAdBuilder { a, b }
}

/// Builder for AD Cholesky.
/// # Examples
///
/// ```ignore
/// // Construct `CholeskyAdBuilder` via its corresponding operation constructor.
/// ```
pub struct CholeskyAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> CholeskyAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Executes AD Cholesky.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_unary_tensor_ad(
            "cholesky_ad",
            self.tensor,
            |ctx, t| tenferro_linalg::cholesky::<T, CpuContext>(ctx, t).map_err(Error::from),
            |ctx, t, dt| tenferro_linalg::cholesky_frule::<T>(ctx, t, dt),
        )
    }
}

/// Creates an AD cholesky builder.
/// # Examples
///
/// ```ignore
/// let _ = cholesky_ad(/* ... */);
/// ```
pub fn cholesky_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> CholeskyAdBuilder<'a, T> {
    CholeskyAdBuilder { tensor }
}

/// Builder for AD solve.
/// # Examples
///
/// ```ignore
/// // Construct `SolveAdBuilder` via its corresponding operation constructor.
/// ```
pub struct SolveAdBuilder<'a, T: Scalar> {
    a: &'a AdTensor<T>,
    b: &'a AdTensor<T>,
}

impl<'a, T> SolveAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Executes AD solve.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_binary_tensor_ad(
            "solve_ad",
            self.a,
            self.b,
            |ctx, a, b| tenferro_linalg::solve::<T, CpuContext>(ctx, a, b).map_err(Error::from),
            |ctx, a, b, da, db| tenferro_linalg::solve_frule::<T>(ctx, a, b, da, db),
        )
    }
}

/// Creates an AD solve builder.
/// # Examples
///
/// ```ignore
/// let _ = solve_ad(/* ... */);
/// ```
pub fn solve_ad<'a, T: Scalar>(a: &'a AdTensor<T>, b: &'a AdTensor<T>) -> SolveAdBuilder<'a, T> {
    SolveAdBuilder { a, b }
}

/// Builder for AD inverse.
/// # Examples
///
/// ```ignore
/// // Construct `InvAdBuilder` via its corresponding operation constructor.
/// ```
pub struct InvAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> InvAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Executes AD inverse.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_unary_tensor_ad(
            "inv_ad",
            self.tensor,
            |ctx, t| tenferro_linalg::inv::<T, CpuContext>(ctx, t).map_err(Error::from),
            |ctx, t, dt| tenferro_linalg::inv_frule::<T, CpuContext>(ctx, t, dt),
        )
    }
}

/// Creates an AD inv builder.
/// # Examples
///
/// ```ignore
/// let _ = inv_ad(/* ... */);
/// ```
pub fn inv_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> InvAdBuilder<'a, T> {
    InvAdBuilder { tensor }
}

/// Builder for AD det.
/// # Examples
///
/// ```ignore
/// // Construct `DetAdBuilder` via its corresponding operation constructor.
/// ```
pub struct DetAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> DetAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Executes AD determinant.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_unary_tensor_ad(
            "det_ad",
            self.tensor,
            |ctx, t| tenferro_linalg::det::<T, CpuContext>(ctx, t).map_err(Error::from),
            |ctx, t, dt| tenferro_linalg::det_frule::<T, CpuContext>(ctx, t, dt),
        )
    }
}

/// Creates an AD det builder.
/// # Examples
///
/// ```ignore
/// let _ = det_ad(/* ... */);
/// ```
pub fn det_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> DetAdBuilder<'a, T> {
    DetAdBuilder { tensor }
}

/// Builder for AD slogdet.
/// # Examples
///
/// ```ignore
/// // Construct `SlogdetAdBuilder` via its corresponding operation constructor.
/// ```
pub struct SlogdetAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> SlogdetAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Executes AD slogdet.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdSlogdetResult<T>> {
        let operands = [self.tensor];
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);

        let (primal, tangent) = if needs_tangent {
            let dt = self
                .tensor
                .tangent()
                .cloned()
                .unwrap_or_else(|| zero_like(self.tensor.primal()));
            let (p, d) = with_cpu_runtime("slogdet_ad", |ctx| {
                tenferro_linalg::slogdet_frule::<T, CpuContext>(ctx, self.tensor.primal(), &dt)
                    .map_err(Error::from)
            })?;
            (p, Some(d))
        } else {
            (
                with_cpu_runtime("slogdet_ad", |ctx| {
                    tenferro_linalg::slogdet::<T, CpuContext>(ctx, self.tensor.primal())
                        .map_err(Error::from)
                })?,
                None,
            )
        };

        let (dsign, dlogabsdet) = if let Some(d) = tangent {
            (Some(d.sign), Some(d.logabsdet))
        } else {
            (None, None)
        };

        Ok(AdSlogdetResult {
            sign: wrap_ad_output("slogdet_ad", &operands, primal.sign, dsign, 1)?,
            logabsdet: wrap_ad_output("slogdet_ad", &operands, primal.logabsdet, dlogabsdet, 2)?,
        })
    }
}

/// Creates an AD slogdet builder.
/// # Examples
///
/// ```ignore
/// let _ = slogdet_ad(/* ... */);
/// ```
pub fn slogdet_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> SlogdetAdBuilder<'a, T> {
    SlogdetAdBuilder { tensor }
}

/// Builder for AD eig.
/// # Examples
///
/// ```ignore
/// // Construct `EigAdBuilder` via its corresponding operation constructor.
/// ```
pub struct EigAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> EigAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T, Complex = Complex<T>> + Float + CpuLinalgScalar,
    Complex<T>: Scalar,
{
    /// Executes AD eig.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdEigResult<T>> {
        let operands = [self.tensor];
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);

        let (primal, tangent) = if needs_tangent {
            let dt = self
                .tensor
                .tangent()
                .cloned()
                .unwrap_or_else(|| zero_like(self.tensor.primal()));
            let (p, d) = with_cpu_runtime("eig_ad", |ctx| {
                tenferro_linalg::eig_frule::<T>(ctx, self.tensor.primal(), &dt).map_err(Error::from)
            })?;
            (p, Some(d))
        } else {
            (
                with_cpu_runtime("eig_ad", |ctx| {
                    tenferro_linalg::eig::<T, CpuContext>(ctx, self.tensor.primal())
                        .map_err(Error::from)
                })?,
                None,
            )
        };

        let (dvalues, dvectors) = if let Some(d) = tangent {
            (Some(d.values), Some(d.vectors))
        } else {
            (None, None)
        };

        Ok(AdEigResult {
            values: wrap_ad_output("eig_ad", &operands, primal.values, dvalues, 1)?,
            vectors: wrap_ad_output("eig_ad", &operands, primal.vectors, dvectors, 2)?,
        })
    }
}

/// Creates an AD eig builder.
/// # Examples
///
/// ```ignore
/// let _ = eig_ad(/* ... */);
/// ```
pub fn eig_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> EigAdBuilder<'a, T> {
    EigAdBuilder { tensor }
}

/// Builder for AD pinv.
/// # Examples
///
/// ```ignore
/// // Construct `PinvAdBuilder` via its corresponding operation constructor.
/// ```
pub struct PinvAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
    rcond: Option<f64>,
}

impl<'a, T> PinvAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Sets rcond.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.rcond(1e-12);
    /// ```
    pub fn rcond(mut self, rcond: f64) -> Self {
        self.rcond = Some(rcond);
        self
    }

    /// Executes AD pinv.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_unary_tensor_ad(
            "pinv_ad",
            self.tensor,
            |ctx, t| {
                tenferro_linalg::pinv::<T, CpuContext>(ctx, t, self.rcond).map_err(Error::from)
            },
            |ctx, t, dt| tenferro_linalg::pinv_frule::<T, CpuContext>(ctx, t, dt, self.rcond),
        )
    }
}

/// Creates an AD pinv builder.
/// # Examples
///
/// ```ignore
/// let _ = pinv_ad(/* ... */);
/// ```
pub fn pinv_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> PinvAdBuilder<'a, T> {
    PinvAdBuilder {
        tensor,
        rcond: None,
    }
}

/// Builder for AD matrix exponential.
/// # Examples
///
/// ```ignore
/// // Construct `MatrixExpAdBuilder` via its corresponding operation constructor.
/// ```
pub struct MatrixExpAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> MatrixExpAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Executes AD matrix exponential.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_unary_tensor_ad(
            "matrix_exp_ad",
            self.tensor,
            |ctx, t| tenferro_linalg::matrix_exp::<T, CpuContext>(ctx, t).map_err(Error::from),
            |ctx, t, dt| tenferro_linalg::matrix_exp_frule::<T, CpuContext>(ctx, t, dt),
        )
    }
}

/// Creates an AD matrix_exp builder.
/// # Examples
///
/// ```ignore
/// let _ = matrix_exp_ad(/* ... */);
/// ```
pub fn matrix_exp_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> MatrixExpAdBuilder<'a, T> {
    MatrixExpAdBuilder { tensor }
}

/// Builder for AD solve_triangular.
/// # Examples
///
/// ```ignore
/// // Construct `SolveTriangularAdBuilder` via its corresponding operation constructor.
/// ```
pub struct SolveTriangularAdBuilder<'a, T: Scalar> {
    a: &'a AdTensor<T>,
    b: &'a AdTensor<T>,
    upper: bool,
}

impl<'a, T> SolveTriangularAdBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Sets whether the matrix is upper triangular.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.upper(true);
    /// ```
    pub fn upper(mut self, upper: bool) -> Self {
        self.upper = upper;
        self
    }

    /// Executes AD triangular solve.
    ///
    /// Forward/reverse inputs are currently unsupported.
    ///
    /// Upstream tracking issue: `tensor4all/tenferro-rs#257`.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        let operands = [self.a, self.b];
        if has_forward(&operands) || has_reverse(&operands) {
            return Err(Error::UnsupportedAdOp {
                op: "solve_triangular_ad",
            });
        }

        with_cpu_runtime("solve_triangular_ad", |ctx| {
            let out = tenferro_linalg::solve_triangular::<T, CpuContext>(
                ctx,
                self.a.primal(),
                self.b.primal(),
                self.upper,
            )
            .map_err(Error::from)?;
            Ok(AdTensor::new_primal(out))
        })
    }
}

/// Creates an AD solve_triangular builder.
/// # Examples
///
/// ```ignore
/// let _ = solve_triangular_ad(/* ... */);
/// ```
pub fn solve_triangular_ad<'a, T: Scalar>(
    a: &'a AdTensor<T>,
    b: &'a AdTensor<T>,
) -> SolveTriangularAdBuilder<'a, T> {
    SolveTriangularAdBuilder { a, b, upper: true }
}

/// Builder for AD norm.
/// # Examples
///
/// ```ignore
/// // Construct `NormAdBuilder` via its corresponding operation constructor.
/// ```
pub struct NormAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
    kind: NormKind,
}

impl<'a, T> NormAdBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Sets norm kind.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.kind(kind);
    /// ```
    pub fn kind(mut self, kind: NormKind) -> Self {
        self.kind = kind;
        self
    }

    /// Executes AD norm.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_unary_tensor_ad(
            "norm_ad",
            self.tensor,
            |ctx, t| tenferro_linalg::norm::<T, CpuContext>(ctx, t, self.kind).map_err(Error::from),
            |ctx, t, dt| tenferro_linalg::norm_frule::<T, CpuContext>(ctx, t, dt, self.kind),
        )
    }
}

/// Creates an AD norm builder.
/// # Examples
///
/// ```ignore
/// let _ = norm_ad(/* ... */);
/// ```
pub fn norm_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> NormAdBuilder<'a, T> {
    NormAdBuilder {
        tensor,
        kind: NormKind::Fro,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenferro_prims::{CudaContext, RocmContext};
    use tenferro_tensor::MemoryOrder;

    fn as_slice<T: Scalar>(t: &Tensor<T>) -> &[T] {
        t.buffer()
            .as_slice()
            .unwrap_or_else(|| panic!("expected CPU-backed contiguous tensor"))
    }

    #[test]
    fn run_requires_runtime() {
        let t = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
        let err = qr(&t).run().err();
        assert!(matches!(err, Some(Error::RuntimeNotConfigured)));
    }

    #[test]
    fn run_with_cuda_runtime_returns_unsupported_runtime_error() {
        let _guard = crate::set_default_runtime(RuntimeContext::Cuda(CudaContext::new()));
        let t = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
        let err = qr(&t).run().err();
        assert!(matches!(
            err,
            Some(Error::UnsupportedRuntimeOp {
                op: "qr",
                runtime: "cuda"
            })
        ));
    }

    #[test]
    fn run_with_rocm_runtime_returns_unsupported_runtime_error() {
        let _guard = crate::set_default_runtime(RuntimeContext::Rocm(RocmContext::new()));
        let t = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
        let err = qr(&t).run().err();
        assert!(matches!(
            err,
            Some(Error::UnsupportedRuntimeOp {
                op: "qr",
                runtime: "rocm"
            })
        ));
    }

    #[test]
    fn primal_einsum_builder_runs() {
        let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
        let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
        let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
        let out = einsum("ij,jk->ik", &[&a, &b]).run().unwrap();
        assert_eq!(out.dims(), &[2, 2]);
        assert_eq!(as_slice(&out).len(), 4);
    }

    #[test]
    fn primal_qr_builder_runs() {
        let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
        let t = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
        let out = qr(&t).run().unwrap();
        assert_eq!(out.q.dims(), &[2, 2]);
        assert_eq!(out.r.dims(), &[2, 2]);
    }

    #[test]
    fn solve_triangular_ad_rejects_non_primal() {
        let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
        let a = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
        let b = Tensor::<f64>::from_slice(&[1.0, 1.0], &[2], MemoryOrder::ColumnMajor).unwrap();
        let da =
            Tensor::<f64>::from_slice(&[0.1, 0.0, 0.0, 0.1], &[2, 2], MemoryOrder::ColumnMajor)
                .unwrap();

        let ad_a = AdTensor::new_forward(a, da);
        let ad_b = AdTensor::new_primal(b);
        let err = solve_triangular_ad(&ad_a, &ad_b).run().err();
        assert!(matches!(err, Some(Error::UnsupportedAdOp { .. })));
    }

    fn assert_primal_mode(t: &AdTensor<f64>) {
        assert!(matches!(t.as_value(), AdValue::Primal(_)));
    }

    #[test]
    fn primal_linalg_builders_cover_all_ops() {
        let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

        let a = Tensor::<f64>::from_slice(&[4.0, 1.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
        let b = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
        let tri =
            Tensor::<f64>::from_slice(&[2.0, 0.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
                .unwrap();
        let a_general =
            Tensor::<f64>::from_slice(&[0.0, 1.0, -1.0, 0.0], &[2, 2], MemoryOrder::ColumnMajor)
                .unwrap();
        let a_rect = Tensor::<f64>::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let a_ls = Tensor::<f64>::from_slice(
            &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            &[3, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let b_ls =
            Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();

        let out_svd = svd(&a).run().unwrap();
        assert_eq!(out_svd.s.dims(), &[2]);
        let out_qr = qr(&a).run().unwrap();
        assert_eq!(out_qr.q.dims(), &[2, 2]);
        let out_lu = lu(&a).pivot(LuPivot::Partial).run().unwrap();
        assert_eq!(out_lu.l.dims(), &[2, 2]);
        let out_eigen = eigen(&a).run().unwrap();
        assert_eq!(out_eigen.values.dims(), &[2]);
        let out_lstsq = lstsq(&a_ls, &b_ls).run().unwrap();
        assert_eq!(out_lstsq.x.dims(), &[2]);
        let out_cholesky = cholesky(&a).run().unwrap();
        assert_eq!(out_cholesky.dims(), &[2, 2]);
        let out_solve = solve(&a, &b).run().unwrap();
        assert_eq!(out_solve.dims(), &[2]);
        let out_inv = inv(&a).run().unwrap();
        assert_eq!(out_inv.dims(), &[2, 2]);
        let out_det = det(&a).run().unwrap();
        assert_eq!(out_det.dims(), &[]);
        let out_slogdet = slogdet(&a).run().unwrap();
        assert_eq!(out_slogdet.sign.dims(), &[]);
        let out_eig = eig(&a_general).run().unwrap();
        assert_eq!(out_eig.values.dims(), &[2]);
        let out_pinv = pinv(&a_rect).rcond(1e-12).run().unwrap();
        assert_eq!(out_pinv.dims(), &[3, 2]);
        let out_exp = matrix_exp(&a).run().unwrap();
        assert_eq!(out_exp.dims(), &[2, 2]);
        let out_tri = solve_triangular(&tri, &b).upper(true).run().unwrap();
        assert_eq!(out_tri.dims(), &[2]);
        let out_norm = norm(&a).kind(NormKind::Fro).run().unwrap();
        assert_eq!(out_norm.dims(), &[]);
    }

    #[test]
    fn ad_linalg_builders_cover_all_ops_in_primal_mode() {
        let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

        let a = Tensor::<f64>::from_slice(&[4.0, 1.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
        let b = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
        let tri =
            Tensor::<f64>::from_slice(&[2.0, 0.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
                .unwrap();
        let a_general =
            Tensor::<f64>::from_slice(&[0.0, 1.0, -1.0, 0.0], &[2, 2], MemoryOrder::ColumnMajor)
                .unwrap();
        let a_rect = Tensor::<f64>::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let a_ls = Tensor::<f64>::from_slice(
            &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            &[3, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let b_ls =
            Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();

        let ad_a = AdTensor::new_primal(a);
        let ad_b = AdTensor::new_primal(b);
        let ad_tri = AdTensor::new_primal(tri);
        let ad_general = AdTensor::new_primal(a_general);
        let ad_rect = AdTensor::new_primal(a_rect);
        let ad_ls_a = AdTensor::new_primal(a_ls);
        let ad_ls_b = AdTensor::new_primal(b_ls);

        let out_svd = svd_ad(&ad_a).run().unwrap();
        assert_primal_mode(&out_svd.u);
        assert_primal_mode(&out_svd.s);
        assert_primal_mode(&out_svd.vt);

        let out_qr = qr_ad(&ad_a).run().unwrap();
        assert_primal_mode(&out_qr.q);
        assert_primal_mode(&out_qr.r);

        let out_lu = lu_ad(&ad_a).run().unwrap();
        assert_primal_mode(&out_lu.l);
        assert_primal_mode(&out_lu.u);

        let out_eigen = eigen_ad(&ad_a).run().unwrap();
        assert_primal_mode(&out_eigen.values);
        assert_primal_mode(&out_eigen.vectors);

        let out_lstsq = lstsq_ad(&ad_ls_a, &ad_ls_b).run().unwrap();
        assert_primal_mode(&out_lstsq.x);
        assert_primal_mode(&out_lstsq.residual);

        assert_primal_mode(&cholesky_ad(&ad_a).run().unwrap());
        assert_primal_mode(&solve_ad(&ad_a, &ad_b).run().unwrap());
        assert_primal_mode(&inv_ad(&ad_a).run().unwrap());
        assert_primal_mode(&det_ad(&ad_a).run().unwrap());

        let out_slogdet = slogdet_ad(&ad_a).run().unwrap();
        assert_primal_mode(&out_slogdet.sign);
        assert_primal_mode(&out_slogdet.logabsdet);

        let out_eig = eig_ad(&ad_general).run().unwrap();
        assert!(matches!(out_eig.values.as_value(), AdValue::Primal(_)));
        assert!(matches!(out_eig.vectors.as_value(), AdValue::Primal(_)));

        assert_primal_mode(&pinv_ad(&ad_rect).run().unwrap());
        assert_primal_mode(&matrix_exp_ad(&ad_a).run().unwrap());
        assert_primal_mode(&solve_triangular_ad(&ad_tri, &ad_b).run().unwrap());
        assert_primal_mode(&norm_ad(&ad_a).kind(NormKind::Fro).run().unwrap());
    }

    #[test]
    fn ad_mode_propagation_forward_and_reverse() {
        let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

        let a = Tensor::<f64>::from_slice(&[4.0, 1.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
        let da =
            Tensor::<f64>::from_slice(&[0.1, 0.0, 0.0, 0.1], &[2, 2], MemoryOrder::ColumnMajor)
                .unwrap();
        let b = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();

        let ad_a_fwd = AdTensor::new_forward(a.clone(), da);
        let ad_b = AdTensor::new_primal(b);
        let out_fwd = solve_ad(&ad_a_fwd, &ad_b).run().unwrap();
        assert!(matches!(out_fwd.as_value(), AdValue::Forward { .. }));

        let ad_a_rev = AdTensor::new_reverse(a.clone(), NodeId(1), TapeId(11), None);
        let ad_b_rev = AdTensor::new_reverse(a, NodeId(2), TapeId(11), None);
        let out_rev = einsum_ad("ij,jk->ik", &[&ad_a_rev, &ad_b_rev])
            .run()
            .unwrap();
        assert!(matches!(out_rev.as_value(), AdValue::Reverse { tape, .. } if *tape == TapeId(11)));
    }
}
