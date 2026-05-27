//! FFT extension operations for tenferro.
//!
//! This crate is an out-of-tree `ExtensionOp` package. The initial
//! implementation executes on host tensors through `rustfft`; it does not add
//! FFT to the core `tenferro` backend trait surface.
//!
//! # Examples
//!
//! ```
//! use num_complex::Complex64;
//! use tenferro_runtime::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};
//! use tenferro_fft::{traced_tensor::fft, FftNorm};
//!
//! let x = TracedTensor::from_vec_col_major(
//!     vec![4],
//!     vec![
//!         Complex64::new(1.0, 0.0),
//!         Complex64::new(2.0, 0.0),
//!         Complex64::new(3.0, 0.0),
//!         Complex64::new(4.0, 0.0),
//!     ],
//! );
//! let y = fft(&x, None, -1, FftNorm::Backward).unwrap();
//!
//! let mut compiler = GraphCompiler::new();
//! let program = compiler.compile(&y).unwrap();
//! let mut executor = GraphExecutor::new(CpuBackend::new());
//! executor.register_extension(tenferro_fft::register_runtime).unwrap();
//! let out = executor.run(&program).unwrap();
//! assert_eq!(out.shape(), &[4]);
//! assert_eq!(out.as_slice::<Complex64>().unwrap()[0], Complex64::new(10.0, 0.0));
//! ```

use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

#[cfg(feature = "autodiff")]
use chainrules_core::{ADRuleError, ADRuleKind, ADRuleResult};
#[cfg(feature = "autodiff")]
use computegraph::fragment::FragmentBuilder;
#[cfg(feature = "autodiff")]
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
#[cfg(feature = "autodiff")]
use computegraph::OpEmitter;
use num_complex::Complex;
use num_traits::{Float, FromPrimitive, Zero};
use rustfft::{FftNum, FftPlanner};
#[cfg(feature = "autodiff")]
use tenferro_ad::extension::ExtensionAdRuleTrait;
use tenferro_extension_macros::define_extension_runtime;
#[cfg(feature = "autodiff")]
use tenferro_extension_macros::define_idempotent_rule_registration;
#[cfg(feature = "autodiff")]
use tenferro_ops::std_tensor_op::StdTensorOp;
#[cfg(feature = "autodiff")]
use tenferro_ops::ShapeGuardContext;
use tenferro_ops::SymDim;
use tenferro_runtime::extension::{apply, ExtensionExecutionContext, ExtensionOpTrait};
use tenferro_runtime::{Error, Result, TracedTensor};
use tenferro_tensor::{DType, Tensor, TensorBackend, TypedTensor};

/// Extension family id used by the tenferro FFT extension.
///
/// # Examples
///
/// ```
/// assert_eq!(
///     tenferro_fft::FFT_EXTENSION_FAMILY_ID,
///     "tenferro-fft.fft.v1"
/// );
/// ```
pub const FFT_EXTENSION_FAMILY_ID: &str = "tenferro-fft.fft.v1";

/// Traced tensor FFT operations.
///
/// This module is the canonical traced tensor namespace for the FFT extension
/// crate. Root-level functions remain available for compatibility.
pub mod traced_tensor {
    pub use crate::{fft, ifft, irfft, rfft};
}

/// FFT normalization convention.
///
/// `Backward` matches NumPy, JAX, and PyTorch defaults: the forward transform
/// is unscaled and the inverse transform is scaled by `1 / n`.
///
/// # Examples
///
/// ```
/// use tenferro_fft::FftNorm;
///
/// assert_eq!(FftNorm::default(), FftNorm::Backward);
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FftNorm {
    /// Scale inverse transforms by `1 / n`.
    #[default]
    Backward,
    /// Scale forward transforms by `1 / n`.
    Forward,
    /// Scale both forward and inverse transforms by `1 / sqrt(n)`.
    Ortho,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FftKind {
    C2C { forward: bool },
    R2C { onesided: bool },
    C2R,
}

#[derive(Clone, Debug, PartialEq)]
struct FftOp {
    kind: FftKind,
    axis: usize,
    n: Option<usize>,
    norm: FftNorm,
}

impl FftOp {
    fn new(kind: FftKind, axis: usize, n: Option<usize>, norm: FftNorm) -> Self {
        Self {
            kind,
            axis,
            n,
            norm,
        }
    }
}

impl ExtensionOpTrait for FftOp {
    fn family_id(&self) -> &'static str {
        FFT_EXTENSION_FAMILY_ID
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        let kind = match self.kind {
            FftKind::C2C { forward: true } => 0,
            FftKind::C2C { forward: false } => 1,
            FftKind::R2C { onesided: true } => 2,
            FftKind::R2C { onesided: false } => 3,
            FftKind::C2R => 4,
        };
        hasher.write_u8(kind);
        hasher.write_usize(self.axis);
        match self.n {
            Some(n) => {
                hasher.write_u8(1);
                hasher.write_usize(n);
            }
            None => hasher.write_u8(0),
        }
        let norm = match self.norm {
            FftNorm::Backward => 0,
            FftNorm::Forward => 1,
            FftNorm::Ortho => 2,
        };
        hasher.write_u8(norm);
    }

    fn payload_eq(&self, other: &dyn ExtensionOpTrait) -> bool {
        other
            .as_any()
            .downcast_ref::<FftOp>()
            .is_some_and(|that| self == that)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOpTrait> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn n_outputs(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        input_dtypes: &[DType],
        input_shapes: &[&[SymDim]],
    ) -> Vec<(DType, Vec<SymDim>)> {
        assert_eq!(
            input_dtypes.len(),
            1,
            "tenferro-fft expects exactly one input"
        );
        assert_eq!(
            input_shapes.len(),
            1,
            "tenferro-fft expects exactly one input shape"
        );
        assert!(
            self.axis < input_shapes[0].len(),
            "tenferro-fft axis {} out of bounds for rank {}",
            self.axis,
            input_shapes[0].len()
        );

        let mut out_shape = input_shapes[0].to_vec();
        let output_dtype = match self.kind {
            FftKind::C2C { .. } => {
                assert!(
                    matches!(input_dtypes[0], DType::C32 | DType::C64),
                    "fft/ifft expect C32 or C64 input for complex-to-complex transforms; got {:?}",
                    input_dtypes[0]
                );
                input_dtypes[0]
            }
            FftKind::R2C { onesided } => {
                assert!(
                    matches!(input_dtypes[0], DType::F32 | DType::F64),
                    "rfft expects F32 or F64 input; got {:?}",
                    input_dtypes[0]
                );
                let len = transform_len_dim(self.n, &input_shapes[0][self.axis]);
                out_shape[self.axis] = if onesided { len / 2usize + 1usize } else { len };
                match input_dtypes[0] {
                    DType::F32 => DType::C32,
                    DType::F64 => DType::C64,
                    _ => unreachable!(),
                }
            }
            FftKind::C2R => {
                assert!(
                    matches!(input_dtypes[0], DType::C32 | DType::C64),
                    "irfft expects C32 or C64 input; got {:?}",
                    input_dtypes[0]
                );
                out_shape[self.axis] = match self.n {
                    Some(n) => SymDim::from(n),
                    None => (input_shapes[0][self.axis].clone() - 1usize) * 2usize,
                };
                match input_dtypes[0] {
                    DType::C32 => DType::F32,
                    DType::C64 => DType::F64,
                    _ => unreachable!(),
                }
            }
        };

        if matches!(self.kind, FftKind::C2C { .. }) {
            out_shape[self.axis] = transform_len_dim(self.n, &input_shapes[0][self.axis]);
        }

        vec![(output_dtype, out_shape)]
    }

    fn eager_execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(tenferro_tensor::Error::InvalidConfig {
                op: "tenferro-fft",
                message: format!("expected 1 input, got {}", inputs.len()),
            });
        }

        let output = match (self.kind, inputs[0]) {
            (FftKind::C2C { forward }, Tensor::C64(input)) => {
                Tensor::C64(TypedTensor::from_vec_col_major(
                    output_shape_c2c(input.shape.as_slice(), self.axis, self.n)?,
                    execute_c2c(input, self.axis, self.n, forward, self.norm)?,
                ))
            }
            (FftKind::C2C { forward }, Tensor::C32(input)) => {
                Tensor::C32(TypedTensor::from_vec_col_major(
                    output_shape_c2c(input.shape.as_slice(), self.axis, self.n)?,
                    execute_c2c(input, self.axis, self.n, forward, self.norm)?,
                ))
            }
            (FftKind::R2C { onesided }, Tensor::F64(input)) => {
                Tensor::C64(TypedTensor::from_vec_col_major(
                    output_shape_r2c(input.shape.as_slice(), self.axis, self.n, onesided)?,
                    execute_r2c(input, self.axis, self.n, onesided, self.norm)?,
                ))
            }
            (FftKind::R2C { onesided }, Tensor::F32(input)) => {
                Tensor::C32(TypedTensor::from_vec_col_major(
                    output_shape_r2c(input.shape.as_slice(), self.axis, self.n, onesided)?,
                    execute_r2c(input, self.axis, self.n, onesided, self.norm)?,
                ))
            }
            (FftKind::C2R, Tensor::C64(input)) => Tensor::F64(TypedTensor::from_vec_col_major(
                output_shape_c2r(input.shape.as_slice(), self.axis, self.n)?,
                execute_c2r(input, self.axis, self.n, self.norm)?,
            )),
            (FftKind::C2R, Tensor::C32(input)) => Tensor::F32(TypedTensor::from_vec_col_major(
                output_shape_c2r(input.shape.as_slice(), self.axis, self.n)?,
                execute_c2r(input, self.axis, self.n, self.norm)?,
            )),
            (kind, other) => {
                return Err(tenferro_tensor::Error::DTypeMismatch {
                    op: match kind {
                        FftKind::C2C { .. } => "fft",
                        FftKind::R2C { .. } => "rfft",
                        FftKind::C2R => "irfft",
                    },
                    lhs: expected_dtype_for(kind),
                    rhs: other.dtype(),
                });
            }
        };
        Ok(vec![output])
    }
}

#[cfg(feature = "autodiff")]
#[derive(Debug)]
struct FftAdRule;

#[cfg(feature = "autodiff")]
impl ExtensionAdRuleTrait for FftAdRule {
    fn family_id(&self) -> &'static str {
        FFT_EXTENSION_FAMILY_ID
    }

    fn linearize(
        &self,
        op: &dyn ExtensionOpTrait,
        builder: &mut FragmentBuilder<StdTensorOp>,
        _primal_in: &[GlobalValKey<StdTensorOp>],
        _primal_out: &[GlobalValKey<StdTensorOp>],
        tangent_in: &[Option<LocalValId>],
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>> {
        let fft_op = fft_payload(op, ADRuleKind::Linearize)?;
        if !matches!(fft_op.kind, FftKind::C2C { .. }) {
            return Err(ADRuleError::unsupported(
                FFT_EXTENSION_FAMILY_ID,
                ADRuleKind::Linearize,
            ));
        }

        match tangent_in[0] {
            Some(dx) => {
                let outputs = builder.add_op(
                    StdTensorOp::Extension(Arc::new(fft_op.clone())),
                    vec![ValRef::Local(dx)],
                    OpMode::Linear {
                        active_mask: vec![true],
                    },
                );
                Ok(vec![Some(outputs[0])])
            }
            None => Ok(vec![None]),
        }
    }

    fn transpose_rule(
        &self,
        op: &dyn ExtensionOpTrait,
        emitter: &mut dyn OpEmitter<StdTensorOp>,
        cotangent_out: &[Option<LocalValId>],
        _inputs: &[ValRef<StdTensorOp>],
        _mode: &OpMode,
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>> {
        let fft_op = fft_payload(op, ADRuleKind::Transpose)?;
        if !matches!(fft_op.kind, FftKind::C2C { .. }) {
            return Err(ADRuleError::unsupported(
                FFT_EXTENSION_FAMILY_ID,
                ADRuleKind::Transpose,
            ));
        }

        match cotangent_out[0] {
            Some(ct) => {
                let outputs = emitter.add_op(
                    StdTensorOp::Extension(Arc::new(fft_op.clone())),
                    vec![ValRef::Local(ct)],
                    OpMode::Linear {
                        active_mask: vec![true],
                    },
                );
                Ok(vec![Some(outputs[0])])
            }
            None => Ok(vec![None]),
        }
    }
}

#[cfg(feature = "autodiff")]
define_idempotent_rule_registration! {
    register_fn = register_fft,
    rule_type = FftAdRule,
}

fn execute_fft_extension<B: TensorBackend + 'static>(
    op: &FftOp,
    inputs: &[&Tensor],
    _ctx: &mut ExtensionExecutionContext<'_, B>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    op.eager_execute(inputs)
}

define_extension_runtime! {
    runtime = FftRuntime,
    family_id = FFT_EXTENSION_FAMILY_ID,
    op_type = FftOp,
    execute = execute_fft_extension,
    register_fn = register_runtime,
}

/// Build a one-dimensional FFT along `axis`.
///
/// Complex inputs use a complex-to-complex transform. Real inputs use a
/// real-to-complex transform that returns the full complex spectrum.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use tenferro_runtime::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};
/// use tenferro_fft::{traced_tensor, FftNorm};
///
/// let x = TracedTensor::from_vec_col_major(vec![2], vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)]);
/// let y = traced_tensor::fft(&x, None, -1, FftNorm::Backward).unwrap();
///
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&y).unwrap();
/// let mut executor = GraphExecutor::new(CpuBackend::new());
/// executor.register_extension(tenferro_fft::register_runtime).unwrap();
/// let out = executor.run(&program).unwrap();
/// assert_eq!(out.as_slice::<Complex64>().unwrap()[0], Complex64::new(3.0, 0.0));
/// ```
pub fn fft(
    input: &TracedTensor,
    n: Option<usize>,
    axis: isize,
    norm: FftNorm,
) -> Result<TracedTensor> {
    let kind = match input.dtype {
        DType::C32 | DType::C64 => FftKind::C2C { forward: true },
        DType::F32 | DType::F64 => FftKind::R2C { onesided: false },
        DType::I32 | DType::I64 | DType::Bool => {
            return Err(fft_config_error(
                "fft",
                format!(
                    "fft expects real or complex floating input, got {:?}",
                    input.dtype
                ),
            ))
        }
    };
    apply_unary_fft("fft", input, kind, n, axis, norm)
}

/// Build a one-dimensional inverse FFT along `axis`.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use tenferro_runtime::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};
/// use tenferro_fft::{traced_tensor, FftNorm};
///
/// let spectrum = TracedTensor::from_vec_col_major(vec![2], vec![Complex64::new(3.0, 0.0), Complex64::new(-1.0, 0.0)]);
/// let y = traced_tensor::ifft(&spectrum, None, -1, FftNorm::Backward).unwrap();
///
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&y).unwrap();
/// let mut executor = GraphExecutor::new(CpuBackend::new());
/// executor.register_extension(tenferro_fft::register_runtime).unwrap();
/// let out = executor.run(&program).unwrap();
/// assert_eq!(out.as_slice::<Complex64>().unwrap()[0], Complex64::new(1.0, 0.0));
/// ```
pub fn ifft(
    input: &TracedTensor,
    n: Option<usize>,
    axis: isize,
    norm: FftNorm,
) -> Result<TracedTensor> {
    if !matches!(input.dtype, DType::C32 | DType::C64) {
        return Err(fft_config_error(
            "ifft",
            format!("ifft expects C32 or C64 input; got {:?}", input.dtype),
        ));
    }
    apply_unary_fft(
        "ifft",
        input,
        FftKind::C2C { forward: false },
        n,
        axis,
        norm,
    )
}

/// Build a one-dimensional real FFT along `axis`.
///
/// The output keeps only the Hermitian one-sided spectrum with axis length
/// `n / 2 + 1`.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use tenferro_runtime::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};
/// use tenferro_fft::{traced_tensor, FftNorm};
///
/// let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
/// let y = traced_tensor::rfft(&x, None, -1, FftNorm::Backward).unwrap();
///
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&y).unwrap();
/// let mut executor = GraphExecutor::new(CpuBackend::new());
/// executor.register_extension(tenferro_fft::register_runtime).unwrap();
/// let out = executor.run(&program).unwrap();
/// assert_eq!(out.shape(), &[2]);
/// assert_eq!(out.as_slice::<Complex64>().unwrap()[0], Complex64::new(3.0, 0.0));
/// ```
pub fn rfft(
    input: &TracedTensor,
    n: Option<usize>,
    axis: isize,
    norm: FftNorm,
) -> Result<TracedTensor> {
    if !matches!(input.dtype, DType::F32 | DType::F64) {
        return Err(fft_config_error(
            "rfft",
            format!("rfft expects F32 or F64 input; got {:?}", input.dtype),
        ));
    }
    apply_unary_fft(
        "rfft",
        input,
        FftKind::R2C { onesided: true },
        n,
        axis,
        norm,
    )
}

/// Build a one-dimensional inverse real FFT along `axis`.
///
/// If `n` is `None`, the output length is inferred as
/// `2 * (input_len - 1)`.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use tenferro_runtime::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};
/// use tenferro_fft::{traced_tensor, FftNorm};
///
/// let spectrum = TracedTensor::from_vec_col_major(
///     vec![2],
///     vec![Complex64::new(3.0, 0.0), Complex64::new(-1.0, 0.0)],
/// );
/// let y = traced_tensor::irfft(&spectrum, Some(2), -1, FftNorm::Backward).unwrap();
///
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&y).unwrap();
/// let mut executor = GraphExecutor::new(CpuBackend::new());
/// executor.register_extension(tenferro_fft::register_runtime).unwrap();
/// let out = executor.run(&program).unwrap();
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
/// ```
pub fn irfft(
    input: &TracedTensor,
    n: Option<usize>,
    axis: isize,
    norm: FftNorm,
) -> Result<TracedTensor> {
    if !matches!(input.dtype, DType::C32 | DType::C64) {
        return Err(fft_config_error(
            "irfft",
            format!("irfft expects C32 or C64 input; got {:?}", input.dtype),
        ));
    }
    apply_unary_fft("irfft", input, FftKind::C2R, n, axis, norm)
}

fn apply_unary_fft(
    op_name: &'static str,
    input: &TracedTensor,
    kind: FftKind,
    n: Option<usize>,
    axis: isize,
    norm: FftNorm,
) -> Result<TracedTensor> {
    #[cfg(feature = "autodiff")]
    register_fft().map_err(|err| Error::Internal(err.to_string()))?;
    validate_n(op_name, n)?;
    let axis = normalize_axis(op_name, axis, input.rank)?;
    let op = Arc::new(FftOp::new(kind, axis, n, norm));
    let mut outputs = apply(op, &[input]);
    outputs
        .pop()
        .ok_or_else(|| Error::Internal("FFT extension declares exactly one output".into()))
}

fn normalize_axis(op: &'static str, axis: isize, rank: usize) -> Result<usize> {
    if rank == 0 {
        return Err(fft_config_error(op, "tenferro-fft requires rank >= 1"));
    }
    let rank_isize = rank as isize;
    let normalized = if axis < 0 { rank_isize + axis } else { axis };
    if normalized < 0 || normalized >= rank_isize {
        return Err(fft_config_error(
            op,
            format!("tenferro-fft axis {axis} out of bounds for rank {rank}"),
        ));
    }
    Ok(normalized as usize)
}

fn validate_n(op: &'static str, n: Option<usize>) -> Result<()> {
    if n == Some(0) {
        return Err(fft_config_error(
            op,
            "tenferro-fft transform length n must be positive",
        ));
    }
    Ok(())
}

fn fft_config_error(op: &'static str, message: impl std::fmt::Display) -> Error {
    Error::TensorRuntime(tenferro_tensor::Error::InvalidConfig {
        op,
        message: message.to_string(),
    })
}

fn transform_len_dim(n: Option<usize>, input_dim: &SymDim) -> SymDim {
    n.map(SymDim::from).unwrap_or_else(|| input_dim.clone())
}

fn expected_dtype_for(kind: FftKind) -> DType {
    match kind {
        FftKind::C2C { .. } | FftKind::C2R => DType::C64,
        FftKind::R2C { .. } => DType::F64,
    }
}

#[cfg(feature = "autodiff")]
fn fft_payload<'a>(op: &'a dyn ExtensionOpTrait, rule: ADRuleKind) -> ADRuleResult<&'a FftOp> {
    op.as_any()
        .downcast_ref::<FftOp>()
        .ok_or_else(|| ADRuleError::unsupported(FFT_EXTENSION_FAMILY_ID, rule))
}

fn output_shape_c2c(
    shape: &[usize],
    axis: usize,
    n: Option<usize>,
) -> tenferro_tensor::Result<Vec<usize>> {
    let len = transform_len(shape, axis, n)?;
    let mut out_shape = shape.to_vec();
    out_shape[axis] = len;
    Ok(out_shape)
}

fn output_shape_r2c(
    shape: &[usize],
    axis: usize,
    n: Option<usize>,
    onesided: bool,
) -> tenferro_tensor::Result<Vec<usize>> {
    let len = transform_len(shape, axis, n)?;
    let mut out_shape = shape.to_vec();
    out_shape[axis] = if onesided { len / 2 + 1 } else { len };
    Ok(out_shape)
}

fn output_shape_c2r(
    shape: &[usize],
    axis: usize,
    n: Option<usize>,
) -> tenferro_tensor::Result<Vec<usize>> {
    validate_axis("irfft", shape, axis)?;
    let input_len = shape[axis];
    if input_len == 0 {
        return Err(tenferro_tensor::Error::InvalidConfig {
            op: "irfft",
            message: "input spectrum axis length must be positive".to_string(),
        });
    }
    let len = n.unwrap_or_else(|| 2 * (input_len - 1));
    if len == 0 {
        return Err(tenferro_tensor::Error::InvalidConfig {
            op: "irfft",
            message: "output length must be positive".to_string(),
        });
    }
    let mut out_shape = shape.to_vec();
    out_shape[axis] = len;
    Ok(out_shape)
}

fn transform_len(shape: &[usize], axis: usize, n: Option<usize>) -> tenferro_tensor::Result<usize> {
    validate_axis("fft", shape, axis)?;
    let len = n.unwrap_or(shape[axis]);
    if len == 0 {
        return Err(tenferro_tensor::Error::InvalidConfig {
            op: "fft",
            message: "transform length must be positive".to_string(),
        });
    }
    Ok(len)
}

fn validate_axis(op: &'static str, shape: &[usize], axis: usize) -> tenferro_tensor::Result<()> {
    if axis >= shape.len() {
        return Err(tenferro_tensor::Error::AxisOutOfBounds {
            op,
            axis,
            rank: shape.len(),
        });
    }
    Ok(())
}

fn execute_c2c<T>(
    input: &TypedTensor<Complex<T>>,
    axis: usize,
    n: Option<usize>,
    forward: bool,
    norm: FftNorm,
) -> tenferro_tensor::Result<Vec<Complex<T>>>
where
    T: FftNum + Float + FromPrimitive,
{
    let in_shape = input.shape.as_slice();
    let fft_len = transform_len(in_shape, axis, n)?;
    let out_shape = output_shape_c2c(in_shape, axis, n)?;
    let out_axis_len = out_shape[axis];
    let input_data = input.host_data();
    let mut output = vec![Complex::zero(); out_shape.iter().product()];
    let mut planner = FftPlanner::<T>::new();
    let fft_plan = if forward {
        planner.plan_fft_forward(fft_len)
    } else {
        planner.plan_fft_inverse(fft_len)
    };
    let scale: T = scale_for(norm, forward, fft_len);
    let mut lane = vec![Complex::zero(); fft_len];

    for_axis_lane(in_shape, axis, out_axis_len, |lane_ctx| {
        lane.fill(Complex::zero());
        let copy_len = lane_ctx.in_axis_len.min(fft_len);
        for (k, slot) in lane.iter_mut().take(copy_len).enumerate() {
            *slot = input_data[lane_ctx.input_offset(k)];
        }
        fft_plan.process(&mut lane);
        if scale != T::one() {
            for value in &mut lane {
                *value = *value * scale;
            }
        }
        for (k, value) in lane.iter().take(out_axis_len).copied().enumerate() {
            output[lane_ctx.output_offset(k)] = value;
        }
    });

    Ok(output)
}

fn execute_r2c<T>(
    input: &TypedTensor<T>,
    axis: usize,
    n: Option<usize>,
    onesided: bool,
    norm: FftNorm,
) -> tenferro_tensor::Result<Vec<Complex<T>>>
where
    T: FftNum + Float + FromPrimitive,
{
    let in_shape = input.shape.as_slice();
    let fft_len = transform_len(in_shape, axis, n)?;
    let out_shape = output_shape_r2c(in_shape, axis, n, onesided)?;
    let out_axis_len = out_shape[axis];
    let input_data = input.host_data();
    let mut output = vec![Complex::zero(); out_shape.iter().product()];
    let mut planner = FftPlanner::<T>::new();
    let fft_plan = planner.plan_fft_forward(fft_len);
    let scale: T = scale_for(norm, true, fft_len);
    let mut lane = vec![Complex::zero(); fft_len];

    for_axis_lane(in_shape, axis, out_axis_len, |lane_ctx| {
        lane.fill(Complex::zero());
        let copy_len = lane_ctx.in_axis_len.min(fft_len);
        for (k, slot) in lane.iter_mut().take(copy_len).enumerate() {
            *slot = Complex::new(input_data[lane_ctx.input_offset(k)], T::zero());
        }
        fft_plan.process(&mut lane);
        if scale != T::one() {
            for value in &mut lane {
                *value = *value * scale;
            }
        }
        for (k, value) in lane.iter().take(out_axis_len).copied().enumerate() {
            output[lane_ctx.output_offset(k)] = value;
        }
    });

    Ok(output)
}

fn execute_c2r<T>(
    input: &TypedTensor<Complex<T>>,
    axis: usize,
    n: Option<usize>,
    norm: FftNorm,
) -> tenferro_tensor::Result<Vec<T>>
where
    T: FftNum + Float + FromPrimitive,
{
    let in_shape = input.shape.as_slice();
    let out_shape = output_shape_c2r(in_shape, axis, n)?;
    let out_axis_len = out_shape[axis];
    let expected_half = out_axis_len / 2 + 1;
    let input_data = input.host_data();
    let mut output = vec![T::zero(); out_shape.iter().product()];
    let mut planner = FftPlanner::<T>::new();
    let fft_plan = planner.plan_fft_inverse(out_axis_len);
    let scale: T = scale_for(norm, false, out_axis_len);
    let mut lane = vec![Complex::zero(); out_axis_len];

    for_axis_lane(in_shape, axis, out_axis_len, |lane_ctx| {
        lane.fill(Complex::zero());
        let copy_len = lane_ctx.in_axis_len.min(expected_half);
        for (k, slot) in lane.iter_mut().take(copy_len).enumerate() {
            *slot = input_data[lane_ctx.input_offset(k)];
        }
        for k in expected_half..out_axis_len {
            let mirror = out_axis_len - k;
            if mirror < lane.len() {
                lane[k] = lane[mirror].conj();
            }
        }
        fft_plan.process(&mut lane);
        for (k, value) in lane.iter().take(out_axis_len).enumerate() {
            output[lane_ctx.output_offset(k)] = value.re * scale;
        }
    });

    Ok(output)
}

fn scale_for<T>(norm: FftNorm, forward: bool, n: usize) -> T
where
    T: Float + FromPrimitive,
{
    let len = T::from_usize(n).expect("FFT length must convert to scalar");
    match (norm, forward) {
        (FftNorm::Backward, true) | (FftNorm::Forward, false) => T::one(),
        (FftNorm::Backward, false) | (FftNorm::Forward, true) => T::one() / len,
        (FftNorm::Ortho, _) => T::one() / len.sqrt(),
    }
}

#[derive(Clone, Copy)]
struct LaneContext {
    input_base: usize,
    output_base: usize,
    axis_stride: usize,
    in_axis_len: usize,
}

impl LaneContext {
    fn input_offset(self, k: usize) -> usize {
        self.input_base + k * self.axis_stride
    }

    fn output_offset(self, k: usize) -> usize {
        self.output_base + k * self.axis_stride
    }
}

fn for_axis_lane(
    in_shape: &[usize],
    axis: usize,
    out_axis_len: usize,
    mut f: impl FnMut(LaneContext),
) {
    let in_axis_len = in_shape[axis];
    let axis_stride: usize = in_shape[..axis].iter().product();
    let outer: usize = in_shape[axis + 1..].iter().product();
    let in_block = axis_stride * in_axis_len;
    let out_block = axis_stride * out_axis_len;

    for outer_idx in 0..outer {
        let in_outer_base = outer_idx * in_block;
        let out_outer_base = outer_idx * out_block;
        for inner in 0..axis_stride {
            f(LaneContext {
                input_base: in_outer_base + inner,
                output_base: out_outer_base + inner,
                axis_stride,
                in_axis_len,
            });
        }
    }
}
