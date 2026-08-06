use crate::config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use crate::types::{
    TensorRank, TensorScalar, TensorView, TensorViewMut, TypedTensor, TypedTensorView,
    TypedTensorViewMut,
};
use crate::validate::validate_convert_dtype;
use crate::{
    AllocationDomainId, AllocationId, DType, Error, RuntimeCacheControl, ShapeMismatch, Tensor,
    TensorRead, TensorValue, TensorWrite, ValidationError,
};
use num_complex::{Complex32, Complex64};
use smallvec::SmallVec;
use std::any::TypeId;
use std::ptr::NonNull;
use strided_kernel::{
    erased_map_into, erased_zip_into, ErasedMapOp, ErasedRawStridedMut, ErasedRawStridedPtr,
    ErasedZipOp, ExecContext, KernelDType,
};

#[cfg(test)]
mod tests;

fn read_boundary_error(op: &'static str) -> crate::Error {
    crate::Error::unsupported(
        op,
        "backend does not accept borrowed tensor views at this execution boundary",
    )
}

fn validation(op: &'static str, source: ValidationError) -> crate::Error {
    Error::validation(op, source)
}

fn invalid_argument(op: &'static str, argument: &'static str, message: impl Into<String>) -> Error {
    Error::invalid_argument(op, argument, message)
}

fn read_tensor<'a>(op: &'static str, input: TensorRead<'a>) -> crate::Result<&'a Tensor> {
    input.as_tensor().ok_or_else(|| read_boundary_error(op))
}

fn validate_axis_list(
    op: &'static str,
    role: &'static str,
    axes: &[usize],
    rank: usize,
) -> crate::Result<()> {
    let mut seen = vec![false; rank];
    for &axis in axes {
        if axis >= rank {
            return Err(validation(
                op,
                ValidationError::AxisOutOfBounds { axis, rank },
            ));
        }
        if seen[axis] {
            return Err(validation(
                op,
                ValidationError::DuplicateAxis { axis, role },
            ));
        }
        seen[axis] = true;
    }
    Ok(())
}

fn validate_role_disjoint(
    op: &'static str,
    first_role: &'static str,
    first_axes: &[usize],
    second_role: &'static str,
    second_axes: &[usize],
) -> crate::Result<()> {
    for &axis in first_axes {
        if second_axes.contains(&axis) {
            return Err(validation(
                op,
                ValidationError::AxisRoleConflict {
                    axis,
                    first_role,
                    second_role,
                },
            ));
        }
    }
    Ok(())
}

/// Infer the output shape for a validated dot-general operation.
#[doc(hidden)]
pub fn dot_general_output_shape(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    config: &DotGeneralConfig,
    op: &'static str,
) -> crate::Result<Vec<usize>> {
    if config.lhs_contracting_dims.len() != config.rhs_contracting_dims.len() {
        return Err(invalid_argument(
            op,
            "contracting_dims",
            "lhs/rhs contracting dim counts differ",
        ));
    }
    if config.lhs_batch_dims.len() != config.rhs_batch_dims.len() {
        return Err(invalid_argument(
            op,
            "batch_dims",
            "lhs/rhs batch dim counts differ",
        ));
    }

    let lhs_rank = lhs_shape.len();
    let rhs_rank = rhs_shape.len();
    validate_axis_list(
        op,
        "lhs_contracting",
        &config.lhs_contracting_dims,
        lhs_rank,
    )?;
    validate_axis_list(
        op,
        "rhs_contracting",
        &config.rhs_contracting_dims,
        rhs_rank,
    )?;
    validate_axis_list(op, "lhs_batch", &config.lhs_batch_dims, lhs_rank)?;
    validate_axis_list(op, "rhs_batch", &config.rhs_batch_dims, rhs_rank)?;
    validate_role_disjoint(
        op,
        "lhs_contracting",
        &config.lhs_contracting_dims,
        "lhs_batch",
        &config.lhs_batch_dims,
    )?;
    validate_role_disjoint(
        op,
        "rhs_contracting",
        &config.rhs_contracting_dims,
        "rhs_batch",
        &config.rhs_batch_dims,
    )?;

    for (&lhs_axis, &rhs_axis) in config
        .lhs_contracting_dims
        .iter()
        .zip(&config.rhs_contracting_dims)
    {
        if lhs_shape[lhs_axis] != rhs_shape[rhs_axis] {
            return Err(validation(
                op,
                ShapeMismatch::ContractedDimensions {
                    lhs_axis,
                    lhs_size: lhs_shape[lhs_axis],
                    rhs_axis,
                    rhs_size: rhs_shape[rhs_axis],
                }
                .into(),
            ));
        }
    }
    for (&lhs_axis, &rhs_axis) in config.lhs_batch_dims.iter().zip(&config.rhs_batch_dims) {
        if lhs_shape[lhs_axis] != rhs_shape[rhs_axis] {
            return Err(validation(
                op,
                ShapeMismatch::ContractedDimensions {
                    lhs_axis,
                    lhs_size: lhs_shape[lhs_axis],
                    rhs_axis,
                    rhs_size: rhs_shape[rhs_axis],
                }
                .into(),
            ));
        }
    }

    let lhs_free = (0..lhs_rank)
        .filter(|axis| {
            !config.lhs_contracting_dims.contains(axis) && !config.lhs_batch_dims.contains(axis)
        })
        .map(|axis| lhs_shape[axis]);
    let rhs_free = (0..rhs_rank)
        .filter(|axis| {
            !config.rhs_contracting_dims.contains(axis) && !config.rhs_batch_dims.contains(axis)
        })
        .map(|axis| rhs_shape[axis]);
    let batch = config.lhs_batch_dims.iter().map(|&axis| lhs_shape[axis]);

    Ok(lhs_free.chain(rhs_free).chain(batch).collect())
}

/// Validate output dtype and shape for dot-general read-into dispatch.
#[doc(hidden)]
pub fn validate_dot_general_read_into(
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    config: &DotGeneralConfig,
    out: &TensorWrite<'_>,
    op: &'static str,
) -> crate::Result<Vec<usize>> {
    if lhs.dtype() != rhs.dtype() {
        return Err(validation(
            op,
            ValidationError::DTypeMismatch {
                expected: crate::core_dtype(lhs.dtype()),
                actual: crate::core_dtype(rhs.dtype()),
            },
        ));
    }
    if lhs.dtype() != out.dtype() {
        return Err(validation(
            op,
            ValidationError::DTypeMismatch {
                expected: crate::core_dtype(lhs.dtype()),
                actual: crate::core_dtype(out.dtype()),
            },
        ));
    }
    let expected = dot_general_output_shape(lhs.shape(), rhs.shape(), config, op)?;
    if out.shape() != expected.as_slice() {
        return Err(validation(
            op,
            ShapeMismatch::ExpectedActual {
                expected: expected.clone().into(),
                actual: out.shape().to_vec().into(),
            }
            .into(),
        ));
    }
    Ok(expected)
}

/// Scalar coefficient accepted by contraction accumulation backends.
///
/// `ContractionScalar` is intentionally narrower than [`crate::TensorScalar`]:
/// dot-general accumulation is only defined for floating and complex tensor
/// dtypes.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{ContractionScalar, DType};
///
/// let alpha = ContractionScalar::F64(2.0);
/// assert_eq!(alpha.dtype(), DType::F64);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ContractionScalar {
    F32(f32),
    F64(f64),
    C32(Complex32),
    C64(Complex64),
}

impl ContractionScalar {
    /// Return this scalar's tensor dtype.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{ContractionScalar, DType};
    ///
    /// assert_eq!(ContractionScalar::F32(1.0).dtype(), DType::F32);
    /// ```
    pub fn dtype(self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::C32(_) => DType::C32,
            Self::C64(_) => DType::C64,
        }
    }

    /// Return the multiplicative identity for a supported contraction dtype.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{ContractionScalar, DType};
    ///
    /// assert_eq!(ContractionScalar::one(DType::F64).unwrap(), ContractionScalar::F64(1.0));
    /// assert!(ContractionScalar::one(DType::I32).is_err());
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    pub fn one(dtype: DType) -> crate::Result<Self> {
        match dtype {
            DType::F32 => Ok(Self::F32(1.0)),
            DType::F64 => Ok(Self::F64(1.0)),
            DType::C32 => Ok(Self::C32(Complex32::new(1.0, 0.0))),
            DType::C64 => Ok(Self::C64(Complex64::new(1.0, 0.0))),
            DType::I32 | DType::I64 | DType::Bool => Err(validation(
                "dot_general",
                ValidationError::DTypeMismatch {
                    expected: crate::core_dtype(dtype),
                    actual: crate::core_dtype(DType::F32),
                },
            )),
        }
    }

    /// Return the additive identity for a supported contraction dtype.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{ContractionScalar, DType};
    ///
    /// assert_eq!(ContractionScalar::zero(DType::F64).unwrap(), ContractionScalar::F64(0.0));
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    pub fn zero(dtype: DType) -> crate::Result<Self> {
        match dtype {
            DType::F32 => Ok(Self::F32(0.0)),
            DType::F64 => Ok(Self::F64(0.0)),
            DType::C32 => Ok(Self::C32(Complex32::new(0.0, 0.0))),
            DType::C64 => Ok(Self::C64(Complex64::new(0.0, 0.0))),
            DType::I32 | DType::I64 | DType::Bool => Err(validation(
                "dot_general",
                ValidationError::DTypeMismatch {
                    expected: crate::core_dtype(dtype),
                    actual: crate::core_dtype(DType::F32),
                },
            )),
        }
    }
}

/// Output-update semantics for dot-general accumulation.
///
/// This keeps contraction axes in [`DotGeneralConfig`] and output update
/// semantics here, so cached and non-cached backend traits can share the same
/// accumulation contract.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{ContractionScalar, DotGeneralAccumulation, DType};
///
/// let accum = DotGeneralAccumulation::overwrite(DType::F64).unwrap();
/// assert_eq!(accum.alpha, ContractionScalar::F64(1.0));
/// assert_eq!(accum.beta, ContractionScalar::F64(0.0));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DotGeneralAccumulation {
    pub lhs_conj: bool,
    pub rhs_conj: bool,
    pub alpha: ContractionScalar,
    pub beta: ContractionScalar,
}

/// One matrix multiply in a grouped GEMM over shared flat buffers.
///
/// Offsets are element offsets into the corresponding shared lhs, rhs, and
/// output buffers. Each job computes a column-major `rows x cols` output block
/// from a column-major `rows x contracted` lhs block and a column-major
/// `contracted x cols` rhs block.
///
/// Provider implementations receive these descriptors through the public
/// grouped-GEMM request accessor. The engine validates ranges and pairwise
/// output disjointness before provider entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GroupedGemmJob {
    out_offset: usize,
    lhs_offset: usize,
    rhs_offset: usize,
    rows: usize,
    contracted: usize,
    cols: usize,
}

impl GroupedGemmJob {
    /// Construct a column-major grouped-GEMM job over shared flat buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        out_offset: usize,
        lhs_offset: usize,
        rhs_offset: usize,
        rows: usize,
        contracted: usize,
        cols: usize,
    ) -> Self {
        Self {
            out_offset,
            lhs_offset,
            rhs_offset,
            rows,
            contracted,
            cols,
        }
    }

    /// Return the output element offset.
    pub fn out_offset(&self) -> usize {
        self.out_offset
    }

    /// Return the left-input element offset.
    pub fn lhs_offset(&self) -> usize {
        self.lhs_offset
    }

    /// Return the right-input element offset.
    pub fn rhs_offset(&self) -> usize {
        self.rhs_offset
    }

    /// Return the output row count.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Return the contracted dimension.
    pub fn contracted(&self) -> usize {
        self.contracted
    }

    /// Return the output column count.
    pub fn cols(&self) -> usize {
        self.cols
    }
}

/// Shared scalar/update metadata for grouped GEMM execution.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupedGemmConfig<'a> {
    jobs: &'a [GroupedGemmJob],
    accumulation: DotGeneralAccumulation,
}

impl<'a> GroupedGemmConfig<'a> {
    pub fn new(jobs: &'a [GroupedGemmJob], accumulation: DotGeneralAccumulation) -> Self {
        Self { jobs, accumulation }
    }

    pub fn jobs(&self) -> &'a [GroupedGemmJob] {
        self.jobs
    }

    pub fn accumulation(&self) -> DotGeneralAccumulation {
        self.accumulation
    }
}

impl DotGeneralAccumulation {
    /// Return overwrite semantics, `out = lhs dot rhs`, for `dtype`.
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    pub fn overwrite(dtype: DType) -> crate::Result<Self> {
        Ok(Self {
            lhs_conj: false,
            rhs_conj: false,
            alpha: ContractionScalar::one(dtype)?,
            beta: ContractionScalar::zero(dtype)?,
        })
    }

    /// Return additive update semantics, `out += lhs dot rhs`, for `dtype`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{ContractionScalar, DType, DotGeneralAccumulation};
    ///
    /// let accum = DotGeneralAccumulation::add_to(DType::F64)?;
    /// assert_eq!(accum.alpha, ContractionScalar::F64(1.0));
    /// assert_eq!(accum.beta, ContractionScalar::F64(1.0));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    pub fn add_to(dtype: DType) -> crate::Result<Self> {
        Ok(Self {
            lhs_conj: false,
            rhs_conj: false,
            alpha: ContractionScalar::one(dtype)?,
            beta: ContractionScalar::one(dtype)?,
        })
    }

    /// Return scaled update semantics, `out = alpha * lhs dot rhs + beta * out`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{ContractionScalar, DotGeneralAccumulation};
    ///
    /// let accum = DotGeneralAccumulation::scaled(
    ///     ContractionScalar::F32(0.5),
    ///     ContractionScalar::F32(2.0),
    /// )?;
    /// assert_eq!(accum.alpha, ContractionScalar::F32(0.5));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    pub fn scaled(alpha: ContractionScalar, beta: ContractionScalar) -> crate::Result<Self> {
        if alpha.dtype() != beta.dtype() {
            return Err(validation(
                "dot_general",
                ValidationError::DTypeMismatch {
                    expected: crate::core_dtype(alpha.dtype()),
                    actual: crate::core_dtype(beta.dtype()),
                },
            ));
        }
        Ok(Self {
            lhs_conj: false,
            rhs_conj: false,
            alpha,
            beta,
        })
    }

    fn validate_for_dtype(self, dtype: DType) -> crate::Result<()> {
        for scalar in [self.alpha, self.beta] {
            if scalar.dtype() != dtype {
                return Err(validation(
                    "dot_general",
                    ValidationError::DTypeMismatch {
                        expected: crate::core_dtype(scalar.dtype()),
                        actual: crate::core_dtype(dtype),
                    },
                ));
            }
        }
        Ok(())
    }
}

#[doc(hidden)]
pub fn validate_dot_general_accumulation(
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    config: &DotGeneralConfig,
    accumulation: DotGeneralAccumulation,
    out: &TensorWrite<'_>,
    op: &'static str,
) -> crate::Result<Vec<usize>> {
    let shape = validate_dot_general_read_into(lhs, rhs, config, out, op)?;
    accumulation.validate_for_dtype(lhs.dtype())?;
    Ok(shape)
}

#[doc(hidden)]
pub fn dot_general_accum_via_temp<B: TensorDot + ?Sized>(
    backend: &mut B,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
    config: &DotGeneralConfig,
    accumulation: DotGeneralAccumulation,
    mut out: TensorWrite<'_>,
) -> crate::Result<()> {
    validate_dot_general_accumulation(&lhs, &rhs, config, accumulation, &out, "dot_general")?;
    let dot = backend.dot_general_with_conj_read(
        lhs,
        rhs,
        config,
        accumulation.lhs_conj,
        accumulation.rhs_conj,
    )?;
    accumulate_dot_result_into(&dot, accumulation, &mut out)
}

fn grouped_checked_product(
    op: &'static str,
    role: &'static str,
    dims: &[usize],
) -> crate::Result<usize> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            invalid_argument(
                op,
                role,
                format!("logical element count overflows usize for shape {dims:?}"),
            )
        })
    })
}

fn checked_gemm_span(
    op: &'static str,
    role: &'static str,
    offset: usize,
    rows: usize,
    cols: usize,
) -> crate::Result<Option<std::ops::Range<usize>>> {
    let len = rows.checked_mul(cols).ok_or_else(|| {
        invalid_argument(
            op,
            role,
            format!("matrix element count overflows usize: rows={rows} cols={cols}"),
        )
    })?;
    if len == 0 {
        return Ok(None);
    }
    let end = offset.checked_add(len).ok_or_else(|| {
        invalid_argument(
            op,
            role,
            format!("matrix range overflows usize: offset={offset} len={len}"),
        )
    })?;
    Ok(Some(offset..end))
}

fn validate_grouped_gemm_range(
    op: &'static str,
    role: &'static str,
    len: usize,
    range: Option<std::ops::Range<usize>>,
) -> crate::Result<()> {
    let Some(range) = range else {
        return Ok(());
    };
    if range.end > len {
        return Err(invalid_argument(
            op,
            role,
            format!(
                "matrix range {}..{} exceeds shared buffer logical length {len}",
                range.start, range.end
            ),
        ));
    }
    Ok(())
}

#[doc(hidden)]
pub fn validate_grouped_gemm(
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    out: &TensorWrite<'_>,
    config: &GroupedGemmConfig<'_>,
    op: &'static str,
) -> crate::Result<()> {
    if lhs.dtype() != rhs.dtype() {
        return Err(validation(
            op,
            ValidationError::DTypeMismatch {
                expected: crate::core_dtype(lhs.dtype()),
                actual: crate::core_dtype(rhs.dtype()),
            },
        ));
    }
    if lhs.dtype() != out.dtype() {
        return Err(validation(
            op,
            ValidationError::DTypeMismatch {
                expected: crate::core_dtype(lhs.dtype()),
                actual: crate::core_dtype(out.dtype()),
            },
        ));
    }
    config.accumulation.validate_for_dtype(lhs.dtype())?;

    let lhs_len = grouped_checked_product(op, "lhs", lhs.shape())?;
    let rhs_len = grouped_checked_product(op, "rhs", rhs.shape())?;
    let out_len = grouped_checked_product(op, "out", out.shape())?;
    // Grouped GEMM job count is runtime-controlled and can be large. Keep the
    // validation ranges in a reserved Vec, not SmallVec, so arbitrary batches
    // avoid inline-capacity tuning and can be sorted for O(n log n) overlap
    // validation.
    let mut out_ranges = Vec::<(usize, std::ops::Range<usize>)>::with_capacity(config.jobs.len());
    for (idx, job) in config.jobs.iter().enumerate() {
        validate_grouped_gemm_range(
            op,
            "lhs",
            lhs_len,
            checked_gemm_span(op, "lhs", job.lhs_offset, job.rows, job.contracted)?,
        )?;
        validate_grouped_gemm_range(
            op,
            "rhs",
            rhs_len,
            checked_gemm_span(op, "rhs", job.rhs_offset, job.contracted, job.cols)?,
        )?;
        let out_range = checked_gemm_span(op, "out", job.out_offset, job.rows, job.cols)?;
        validate_grouped_gemm_range(op, "out", out_len, out_range.clone())?;
        if let Some(out_range) = out_range {
            out_ranges.push((idx, out_range));
        }
    }
    out_ranges.sort_unstable_by_key(|(_, range)| range.start);
    for pair in out_ranges.windows(2) {
        let (prev_idx, previous) = &pair[0];
        let (idx, current) = &pair[1];
        if previous.end > current.start {
            return Err(invalid_argument(
                op,
                "jobs",
                format!(
                    "grouped GEMM output range for job {idx} overlaps job {prev_idx} range {}..{}",
                    previous.start, previous.end
                ),
            ));
        }
    }
    Ok(())
}

fn add_element_offsets(
    op: &'static str,
    base: isize,
    offset: usize,
    role: &'static str,
) -> crate::Result<isize> {
    let offset = isize::try_from(offset).map_err(|_| {
        invalid_argument(op, role, format!("offset {offset} does not fit in isize"))
    })?;
    base.checked_add(offset).ok_or_else(|| {
        invalid_argument(
            op,
            role,
            format!("offset overflows isize: base={base} offset={offset}"),
        )
    })
}

fn dim_stride(op: &'static str, dim: usize, role: &'static str) -> crate::Result<isize> {
    isize::try_from(dim).map_err(|_| {
        invalid_argument(
            op,
            role,
            format!("leading dimension {dim} does not fit in isize"),
        )
    })
}

fn typed_read_storage<'a, T: crate::TensorScalar>(
    tensor: &'a TypedTensor<T>,
    op: &'static str,
) -> crate::Result<(&'a [T], isize)> {
    tensor.host_data().map(|data| (data, 0)).map_err(|_| {
        crate::Error::runtime_state(
            op,
            "grouped GEMM default path requires host-backed tensor storage",
        )
    })
}

fn grouped_gemm_default_config() -> DotGeneralConfig {
    // DotGeneralConfig owns Vec fields, so this rank-2 fallback config follows
    // that API boundary rather than introducing SmallVec locally.
    DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: Vec::new(),
        rhs_batch_dims: Vec::new(),
    }
}

trait GroupedGemmDType<T> {
    fn wrap_read(view: TypedTensorView<'_, T>) -> TensorView<'_>;
    fn wrap_write(view: TypedTensorViewMut<'_, T>) -> TensorViewMut<'_>;
}

struct GroupedF32;
struct GroupedF64;
struct GroupedC32;
struct GroupedC64;

impl GroupedGemmDType<f32> for GroupedF32 {
    fn wrap_read(view: TypedTensorView<'_, f32>) -> TensorView<'_> {
        TensorView::F32(view)
    }

    fn wrap_write(view: TypedTensorViewMut<'_, f32>) -> TensorViewMut<'_> {
        TensorViewMut::F32(view)
    }
}

impl GroupedGemmDType<f64> for GroupedF64 {
    fn wrap_read(view: TypedTensorView<'_, f64>) -> TensorView<'_> {
        TensorView::F64(view)
    }

    fn wrap_write(view: TypedTensorViewMut<'_, f64>) -> TensorViewMut<'_> {
        TensorViewMut::F64(view)
    }
}

impl GroupedGemmDType<Complex32> for GroupedC32 {
    fn wrap_read(view: TypedTensorView<'_, Complex32>) -> TensorView<'_> {
        TensorView::C32(view)
    }

    fn wrap_write(view: TypedTensorViewMut<'_, Complex32>) -> TensorViewMut<'_> {
        TensorViewMut::C32(view)
    }
}

impl GroupedGemmDType<Complex64> for GroupedC64 {
    fn wrap_read(view: TypedTensorView<'_, Complex64>) -> TensorView<'_> {
        TensorView::C64(view)
    }

    fn wrap_write(view: TypedTensorViewMut<'_, Complex64>) -> TensorViewMut<'_> {
        TensorViewMut::C64(view)
    }
}

#[allow(clippy::too_many_arguments)]
fn grouped_gemm_default_loop<B, T, V>(
    backend: &mut B,
    lhs_data: &[T],
    lhs_base: isize,
    rhs_data: &[T],
    rhs_base: isize,
    out_view: &mut TypedTensorViewMut<'_, T>,
    config: &GroupedGemmConfig<'_>,
) -> crate::Result<()>
where
    B: TensorDot + ?Sized,
    T: 'static,
    V: GroupedGemmDType<T>,
{
    let op = "grouped_gemm";
    let dot_config = grouped_gemm_default_config();
    for job in config.jobs {
        let lhs_offset = add_element_offsets(op, lhs_base, job.lhs_offset, "lhs")?;
        let rhs_offset = add_element_offsets(op, rhs_base, job.rhs_offset, "rhs")?;
        let out_offset = add_element_offsets(op, out_view.offset(), job.out_offset, "out")?;
        let lhs_rows = dim_stride(op, job.rows, "lhs")?;
        let rhs_rows = dim_stride(op, job.contracted, "rhs")?;
        let out_rows = dim_stride(op, job.rows, "out")?;
        // TypedTensorView constructors own Vec shape/stride metadata. These
        // fallback rank-2 views are short-lived, but SmallVec is not usable
        // without changing the view API.
        let lhs_matrix = TypedTensorView::from_slice(
            vec![job.rows, job.contracted],
            vec![1, lhs_rows],
            lhs_offset,
            lhs_data,
        )?;
        let rhs_matrix = TypedTensorView::from_slice(
            vec![job.contracted, job.cols],
            vec![1, rhs_rows],
            rhs_offset,
            rhs_data,
        )?;
        let out_storage = out_view.host_storage_mut()?;
        let out_matrix = TypedTensorViewMut::from_slice(
            vec![job.rows, job.cols],
            vec![1, out_rows],
            out_offset,
            out_storage,
        )?;
        backend.dot_general_read_into_accum(
            TensorRead::from_view(V::wrap_read(lhs_matrix)),
            TensorRead::from_view(V::wrap_read(rhs_matrix)),
            &dot_config,
            config.accumulation,
            TensorWrite::from_view(V::wrap_write(out_matrix)),
        )?;
    }
    Ok(())
}

#[doc(hidden)]
pub fn grouped_gemm_via_sequential<B>(
    backend: &mut B,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
    config: &GroupedGemmConfig<'_>,
    mut out: TensorWrite<'_>,
) -> crate::Result<()>
where
    B: TensorDot + ?Sized,
{
    validate_grouped_gemm(&lhs, &rhs, &out, config, "grouped_gemm")?;
    macro_rules! dispatch {
        ($variant:ident, $wrapper:ty) => {
            match (&lhs, &rhs, &mut out) {
                (
                    TensorRead::Tensor(Tensor::$variant(a)),
                    TensorRead::Tensor(Tensor::$variant(b)),
                    TensorWrite::Tensor(Tensor::$variant(c)),
                ) => {
                    let (a_data, a_base) = typed_read_storage(a, "grouped_gemm")?;
                    let (b_data, b_base) = typed_read_storage(b, "grouped_gemm")?;
                    let mut c_view = c.as_view_mut();
                    return grouped_gemm_default_loop::<_, _, $wrapper>(
                        backend,
                        a_data,
                        a_base,
                        b_data,
                        b_base,
                        &mut c_view,
                        config,
                    );
                }
                (
                    TensorRead::Tensor(Tensor::$variant(a)),
                    TensorRead::View(TensorView::$variant(b)),
                    TensorWrite::Tensor(Tensor::$variant(c)),
                ) => {
                    let (a_data, a_base) = typed_read_storage(a, "grouped_gemm")?;
                    let mut c_view = c.as_view_mut();
                    return grouped_gemm_default_loop::<_, _, $wrapper>(
                        backend,
                        a_data,
                        a_base,
                        b.host_storage()?,
                        b.offset(),
                        &mut c_view,
                        config,
                    );
                }
                (
                    TensorRead::View(TensorView::$variant(a)),
                    TensorRead::Tensor(Tensor::$variant(b)),
                    TensorWrite::Tensor(Tensor::$variant(c)),
                ) => {
                    let (b_data, b_base) = typed_read_storage(b, "grouped_gemm")?;
                    let mut c_view = c.as_view_mut();
                    return grouped_gemm_default_loop::<_, _, $wrapper>(
                        backend,
                        a.host_storage()?,
                        a.offset(),
                        b_data,
                        b_base,
                        &mut c_view,
                        config,
                    );
                }
                (
                    TensorRead::View(TensorView::$variant(a)),
                    TensorRead::View(TensorView::$variant(b)),
                    TensorWrite::Tensor(Tensor::$variant(c)),
                ) => {
                    let mut c_view = c.as_view_mut();
                    return grouped_gemm_default_loop::<_, _, $wrapper>(
                        backend,
                        a.host_storage()?,
                        a.offset(),
                        b.host_storage()?,
                        b.offset(),
                        &mut c_view,
                        config,
                    );
                }
                (
                    TensorRead::Tensor(Tensor::$variant(a)),
                    TensorRead::Tensor(Tensor::$variant(b)),
                    TensorWrite::View(TensorViewMut::$variant(c)),
                ) => {
                    let (a_data, a_base) = typed_read_storage(a, "grouped_gemm")?;
                    let (b_data, b_base) = typed_read_storage(b, "grouped_gemm")?;
                    return grouped_gemm_default_loop::<_, _, $wrapper>(
                        backend, a_data, a_base, b_data, b_base, c, config,
                    );
                }
                (
                    TensorRead::Tensor(Tensor::$variant(a)),
                    TensorRead::View(TensorView::$variant(b)),
                    TensorWrite::View(TensorViewMut::$variant(c)),
                ) => {
                    let (a_data, a_base) = typed_read_storage(a, "grouped_gemm")?;
                    return grouped_gemm_default_loop::<_, _, $wrapper>(
                        backend,
                        a_data,
                        a_base,
                        b.host_storage()?,
                        b.offset(),
                        c,
                        config,
                    );
                }
                (
                    TensorRead::View(TensorView::$variant(a)),
                    TensorRead::Tensor(Tensor::$variant(b)),
                    TensorWrite::View(TensorViewMut::$variant(c)),
                ) => {
                    let (b_data, b_base) = typed_read_storage(b, "grouped_gemm")?;
                    return grouped_gemm_default_loop::<_, _, $wrapper>(
                        backend,
                        a.host_storage()?,
                        a.offset(),
                        b_data,
                        b_base,
                        c,
                        config,
                    );
                }
                (
                    TensorRead::View(TensorView::$variant(a)),
                    TensorRead::View(TensorView::$variant(b)),
                    TensorWrite::View(TensorViewMut::$variant(c)),
                ) => {
                    return grouped_gemm_default_loop::<_, _, $wrapper>(
                        backend,
                        a.host_storage()?,
                        a.offset(),
                        b.host_storage()?,
                        b.offset(),
                        c,
                        config,
                    );
                }
                _ => {}
            }
        };
    }

    dispatch!(F32, GroupedF32);
    dispatch!(F64, GroupedF64);
    dispatch!(C32, GroupedC32);
    dispatch!(C64, GroupedC64);
    Err(validation(
        "grouped_gemm",
        ValidationError::DTypeMismatch {
            expected: crate::core_dtype(lhs.dtype()),
            actual: crate::core_dtype(out.dtype()),
        },
    ))
}

fn grouped_gemm_default<B>(
    backend: &mut B,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
    config: &GroupedGemmConfig<'_>,
    out: TensorWrite<'_>,
) -> crate::Result<()>
where
    B: TensorDot + ?Sized,
{
    grouped_gemm_via_sequential(backend, lhs, rhs, config, out)
}

#[doc(hidden)]
pub fn accumulate_dot_result_into(
    dot: &Tensor,
    accumulation: DotGeneralAccumulation,
    out: &mut TensorWrite<'_>,
) -> crate::Result<()> {
    macro_rules! dispatch {
        ($variant:ident, $ty:ty) => {
            if let (
                Tensor::$variant(dot),
                ContractionScalar::$variant(alpha),
                ContractionScalar::$variant(beta),
            ) = (dot, accumulation.alpha, accumulation.beta)
            {
                match out {
                    TensorWrite::Tensor(Tensor::$variant(out)) => {
                        let mut out = out.as_view_mut();
                        accumulate_typed(dot.as_slice()?, alpha, beta, &mut out)?;
                        return Ok(());
                    }
                    TensorWrite::View(crate::TensorViewMut::$variant(out)) => {
                        accumulate_typed(dot.as_slice()?, alpha, beta, out)?;
                        return Ok(());
                    }
                    _ => {}
                }
            }
        };
    }

    dispatch!(F32, f32);
    dispatch!(F64, f64);
    dispatch!(C32, Complex32);
    dispatch!(C64, Complex64);

    Err(validation(
        "dot_general",
        ValidationError::DTypeMismatch {
            expected: crate::core_dtype(accumulation.alpha.dtype()),
            actual: crate::core_dtype(dot.dtype()),
        },
    ))
}

fn accumulate_typed<T>(
    dot: &[T],
    alpha: T,
    beta: T,
    out: &mut TypedTensorViewMut<'_, T>,
) -> crate::Result<()>
where
    T: Copy
        + PartialEq
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + num_traits::Zero
        + 'static,
{
    let beta_is_zero = beta == T::zero();
    if let Some(output) = compact_host_accumulation_slice(out, dot.len())? {
        for (output, dot_value) in output.iter_mut().zip(dot.iter().copied()) {
            // INVARIANT: beta == 0 follows BLAS GEMM semantics and does not read
            // the existing output element; beta != 0 requires an initialized
            // TensorWrite target and performs a read-modify-write update.
            *output = if beta_is_zero {
                alpha * dot_value
            } else {
                alpha * dot_value + beta * *output
            };
        }
        return Ok(());
    }

    for (linear, dot_value) in dot.iter().copied().enumerate() {
        let indices = flat_to_multi_for_shape(out.shape(), linear);
        let output = out.get_mut(&indices).ok_or_else(|| {
            invalid_argument(
                "dot_general",
                "output",
                format!("index {indices:?} is outside accumulation target"),
            )
        })?;
        // INVARIANT: beta == 0 follows BLAS GEMM semantics and does not read
        // the existing output element; beta != 0 requires an initialized
        // TensorWrite target and performs a read-modify-write update.
        *output = if beta_is_zero {
            alpha * dot_value
        } else {
            alpha * dot_value + beta * *output
        };
    }
    Ok(())
}

fn compact_host_accumulation_slice<'a, T: 'static>(
    out: &'a mut TypedTensorViewMut<'_, T>,
    expected_len: usize,
) -> crate::Result<Option<&'a mut [T]>> {
    if out.backend_buffer().is_some()
        || out.n_elements() != expected_len
        || !out.is_col_major_contiguous()?
    {
        return Ok(None);
    }

    let start = usize::try_from(out.offset()).map_err(|_| {
        invalid_argument("dot_general", "output", "compact output offset is negative")
    })?;
    let end = start
        .checked_add(expected_len)
        .ok_or_else(|| validation("dot_general", ValidationError::IntegerOverflow))?;
    out.host_storage_mut()?
        .get_mut(start..end)
        .map(Some)
        .ok_or_else(|| {
            invalid_argument(
                "dot_general",
                "output",
                "compact output is outside its backing storage",
            )
        })
}

fn flat_to_multi_for_shape(shape: &[usize], mut linear: usize) -> Vec<usize> {
    let mut indices = Vec::with_capacity(shape.len());
    for &dim in shape {
        if dim == 0 {
            indices.push(0);
        } else {
            indices.push(linear % dim);
            linear /= dim;
        }
    }
    indices
}

/// Canonical elementwise fusion plan shared between segmented execution and backends.
#[doc(hidden)]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ElementwiseFusionPlan {
    dtype: crate::DType,
    input_count: usize,
    // Keep view metadata in Vecs. A/B benchmarking on the broadcast_mul
    // path showed SmallVec made this metadata path about 6-7% slower.
    input_views: Vec<ElementwiseFusionInputView>,
    outputs: Vec<usize>,
    ops: Vec<ElementwiseFusionInst>,
}

/// Metadata-only view applied to one backend fusion input.
#[doc(hidden)]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum ElementwiseFusionInputView {
    Identity,
    BroadcastInDim {
        // Vec is intentional here; see ElementwiseFusionPlan::input_views.
        shape: Vec<usize>,
        dims: Vec<usize>,
    },
}

/// One node in a canonical elementwise fusion plan.
#[doc(hidden)]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ElementwiseFusionInst {
    op: ElementwiseFusionOp,
    inputs: Vec<usize>,
}

tenferro_core_ops::define_elementwise_fusion_op!();

impl ElementwiseFusionPlan {
    /// Build a backend elementwise fusion plan.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::backend::{
    ///     ElementwiseFusionInst, ElementwiseFusionOp, ElementwiseFusionPlan,
    /// };
    /// use tenferro_tensor::DType;
    ///
    /// let plan = ElementwiseFusionPlan::new(
    ///     DType::F64,
    ///     2,
    ///     vec![2],
    ///     vec![ElementwiseFusionInst::new(ElementwiseFusionOp::Add, vec![0, 1])],
    /// );
    /// assert_eq!(plan.input_count(), 2);
    /// ```
    pub fn new(
        dtype: crate::DType,
        input_count: usize,
        outputs: Vec<usize>,
        ops: Vec<ElementwiseFusionInst>,
    ) -> Self {
        Self::with_input_views(
            dtype,
            vec![ElementwiseFusionInputView::Identity; input_count],
            outputs,
            ops,
        )
    }

    /// Build a backend elementwise fusion plan with input view metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::backend::{
    ///     ElementwiseFusionInputView, ElementwiseFusionInst, ElementwiseFusionOp,
    ///     ElementwiseFusionPlan,
    /// };
    /// use tenferro_tensor::DType;
    ///
    /// let plan = ElementwiseFusionPlan::with_input_views(
    ///     DType::F64,
    ///     vec![ElementwiseFusionInputView::broadcast_in_dim(vec![2, 3], vec![0])],
    ///     vec![1],
    ///     vec![ElementwiseFusionInst::new(ElementwiseFusionOp::Negate, vec![0])],
    /// );
    /// assert_eq!(plan.input_count(), 1);
    /// ```
    pub fn with_input_views(
        dtype: crate::DType,
        input_views: impl IntoIterator<Item = ElementwiseFusionInputView>,
        outputs: Vec<usize>,
        ops: Vec<ElementwiseFusionInst>,
    ) -> Self {
        let input_views = input_views.into_iter().collect::<Vec<_>>();
        let input_count = input_views.len();
        Self {
            dtype,
            input_count,
            input_views,
            outputs,
            ops,
        }
    }

    /// Return the scalar dtype expected by this fusion plan.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::backend::ElementwiseFusionPlan;
    /// use tenferro_tensor::DType;
    ///
    /// let plan = ElementwiseFusionPlan::new(DType::F32, 0, Vec::new(), Vec::new());
    /// assert_eq!(plan.dtype(), DType::F32);
    /// ```
    pub fn dtype(&self) -> crate::DType {
        self.dtype
    }

    /// Return the number of input tensors expected by this plan.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::backend::ElementwiseFusionPlan;
    /// use tenferro_tensor::DType;
    ///
    /// let plan = ElementwiseFusionPlan::new(DType::F64, 3, Vec::new(), Vec::new());
    /// assert_eq!(plan.input_count(), 3);
    /// ```
    pub fn input_count(&self) -> usize {
        self.input_count
    }

    /// Return metadata views applied to fusion inputs before executing ops.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::backend::ElementwiseFusionPlan;
    /// use tenferro_tensor::DType;
    ///
    /// let plan = ElementwiseFusionPlan::new(DType::F64, 2, Vec::new(), Vec::new());
    /// assert_eq!(plan.input_views().len(), 2);
    /// ```
    pub fn input_views(&self) -> &[ElementwiseFusionInputView] {
        &self.input_views
    }

    /// Return the value ids selected as fusion outputs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::backend::ElementwiseFusionPlan;
    /// use tenferro_tensor::DType;
    ///
    /// let plan = ElementwiseFusionPlan::new(DType::F64, 0, vec![0], Vec::new());
    /// assert_eq!(plan.outputs(), &[0]);
    /// ```
    pub fn outputs(&self) -> &[usize] {
        &self.outputs
    }

    /// Return the fused elementwise instruction sequence.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::backend::{
    ///     ElementwiseFusionInst, ElementwiseFusionOp, ElementwiseFusionPlan,
    /// };
    /// use tenferro_tensor::DType;
    ///
    /// let inst = ElementwiseFusionInst::new(ElementwiseFusionOp::Negate, vec![0]);
    /// let plan = ElementwiseFusionPlan::new(DType::F64, 1, vec![1], vec![inst]);
    /// assert_eq!(plan.ops().len(), 1);
    /// ```
    pub fn ops(&self) -> &[ElementwiseFusionInst] {
        &self.ops
    }
}

impl ElementwiseFusionInputView {
    /// Build metadata for a `BroadcastInDim` fusion input view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::backend::ElementwiseFusionInputView;
    ///
    /// let view = ElementwiseFusionInputView::broadcast_in_dim(vec![2, 3], vec![0]);
    /// assert!(matches!(view, ElementwiseFusionInputView::BroadcastInDim { .. }));
    /// ```
    pub fn broadcast_in_dim(
        shape: impl IntoIterator<Item = usize>,
        dims: impl IntoIterator<Item = usize>,
    ) -> Self {
        Self::BroadcastInDim {
            shape: shape.into_iter().collect(),
            dims: dims.into_iter().collect(),
        }
    }

    /// Return true when this fusion input is an identity view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::backend::ElementwiseFusionInputView;
    ///
    /// assert!(ElementwiseFusionInputView::Identity.is_identity());
    /// ```
    pub fn is_identity(&self) -> bool {
        matches!(self, Self::Identity)
    }
}

impl ElementwiseFusionInst {
    /// Build a backend elementwise fusion instruction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::backend::{ElementwiseFusionInst, ElementwiseFusionOp};
    ///
    /// let inst = ElementwiseFusionInst::new(ElementwiseFusionOp::Add, vec![0, 1]);
    /// assert_eq!(inst.inputs(), &[0, 1]);
    /// ```
    pub fn new(op: ElementwiseFusionOp, inputs: Vec<usize>) -> Self {
        Self { op, inputs }
    }

    /// Return the elementwise op executed by this instruction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::backend::{ElementwiseFusionInst, ElementwiseFusionOp};
    ///
    /// let inst = ElementwiseFusionInst::new(ElementwiseFusionOp::Negate, vec![0]);
    /// assert_eq!(inst.op(), ElementwiseFusionOp::Negate);
    /// ```
    pub fn op(&self) -> ElementwiseFusionOp {
        self.op
    }

    /// Return this instruction's input value ids.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::backend::{ElementwiseFusionInst, ElementwiseFusionOp};
    ///
    /// let inst = ElementwiseFusionInst::new(ElementwiseFusionOp::Multiply, vec![2, 0]);
    /// assert_eq!(inst.inputs(), &[2, 0]);
    /// ```
    pub fn inputs(&self) -> &[usize] {
        &self.inputs
    }
}

/// Runtime operation selected by [`TensorElementwise::elementwise_read_into`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ElementwiseReadOp {
    /// Binary addition.
    Add,
    /// Binary subtraction.
    Subtract,
    /// Binary multiplication.
    Multiply,
    /// Unary negation.
    Negate,
    /// Unary conjugation.
    Conj,
    /// Binary division.
    Divide,
}

impl ElementwiseReadOp {
    fn label(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Subtract => "sub",
            Self::Multiply => "mul",
            Self::Negate => "neg",
            Self::Conj => "conj",
            Self::Divide => "div",
        }
    }

    fn arity(self) -> usize {
        match self {
            Self::Negate | Self::Conj => 1,
            Self::Add | Self::Subtract | Self::Multiply | Self::Divide => 2,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum StorageIdentity {
    Host {
        start: usize,
        end: usize,
    },
    Backend {
        domain: Option<AllocationDomainId>,
        allocation: Option<AllocationId>,
        family: &'static str,
        object: usize,
    },
}

fn host_storage_identity<T>(data: &[T]) -> StorageIdentity {
    let start = data.as_ptr() as usize;
    let bytes = std::mem::size_of_val(data);
    StorageIdentity::Host {
        start,
        end: start.saturating_add(bytes),
    }
}

fn backend_storage_identity<T: 'static>(buffer: &dyn crate::BackendStorage<T>) -> StorageIdentity {
    StorageIdentity::Backend {
        domain: buffer.allocation_domain(),
        allocation: buffer.allocation_id(),
        family: buffer.backend_family(),
        // INVARIANT: every backend buffer is borrowed from the single Box-owned
        // root allocation; the data pointer of this trait object is stable for
        // that owner and is used only as a fallback when provider identity is
        // unavailable.
        object: buffer as *const dyn crate::BackendStorage<T> as *const () as usize,
    }
}

fn typed_tensor_storage_identity<T: crate::TensorScalar>(
    tensor: &TypedTensor<T>,
) -> crate::Result<StorageIdentity> {
    if tensor.backend_buffer().is_some() {
        let buffer = tensor.backend_buffer().ok_or_else(|| {
            crate::Error::runtime_state("typed_tensor_storage_identity", "backend buffer missing")
        })?;
        Ok(backend_storage_identity(buffer))
    } else {
        Ok(host_storage_identity(tensor.host_data()?))
    }
}

fn typed_view_storage_identity<T: crate::TensorScalar + 'static>(
    view: &TypedTensorView<'_, T>,
) -> crate::Result<StorageIdentity> {
    match view.backend_buffer() {
        Some(buffer) => Ok(backend_storage_identity(buffer)),
        None => view.host_storage().map(host_storage_identity),
    }
}

fn tensor_read_storage_identity(input: &TensorRead<'_>) -> crate::Result<StorageIdentity> {
    macro_rules! typed_identity {
        ($value:expr) => {
            match $value {
                Tensor::F32(value) => typed_tensor_storage_identity(value),
                Tensor::F64(value) => typed_tensor_storage_identity(value),
                Tensor::I32(value) => typed_tensor_storage_identity(value),
                Tensor::I64(value) => typed_tensor_storage_identity(value),
                Tensor::Bool(value) => typed_tensor_storage_identity(value),
                Tensor::C32(value) => typed_tensor_storage_identity(value),
                Tensor::C64(value) => typed_tensor_storage_identity(value),
            }
        };
    }
    macro_rules! view_identity {
        ($value:expr) => {
            match $value {
                TensorView::F32(value) => typed_view_storage_identity(value),
                TensorView::F64(value) => typed_view_storage_identity(value),
                TensorView::I32(value) => typed_view_storage_identity(value),
                TensorView::I64(value) => typed_view_storage_identity(value),
                TensorView::Bool(value) => typed_view_storage_identity(value),
                TensorView::C32(value) => typed_view_storage_identity(value),
                TensorView::C64(value) => typed_view_storage_identity(value),
            }
        };
    }

    match input {
        TensorRead::Tensor(tensor) => typed_identity!(tensor),
        TensorRead::View(view) => view_identity!(view),
    }
}

fn storage_overlaps(lhs: StorageIdentity, rhs: StorageIdentity) -> bool {
    match (lhs, rhs) {
        (
            StorageIdentity::Host {
                start: lhs_start,
                end: lhs_end,
            },
            StorageIdentity::Host {
                start: rhs_start,
                end: rhs_end,
            },
        ) => lhs_start < rhs_end && rhs_start < lhs_end,
        (
            StorageIdentity::Backend {
                domain: lhs_domain,
                allocation: lhs_allocation,
                family: lhs_family,
                object: lhs_object,
            },
            StorageIdentity::Backend {
                domain: rhs_domain,
                allocation: rhs_allocation,
                family: rhs_family,
                object: rhs_object,
            },
        ) => {
            lhs_object == rhs_object
                || matches!(
                    (lhs_domain, rhs_domain, lhs_allocation, rhs_allocation),
                    (Some(lhs_domain), Some(rhs_domain), Some(lhs), Some(rhs))
                        if lhs_domain == rhs_domain && lhs == rhs
                )
                || matches!(
                    (lhs_domain, rhs_domain, lhs_allocation, rhs_allocation),
                    (None, None, Some(lhs), Some(rhs)) if lhs_family == rhs_family && lhs == rhs
                )
        }
        _ => false,
    }
}

fn validate_elementwise_output_disjoint(
    op: ElementwiseReadOp,
    inputs: &[TensorRead<'_>],
    out: &TensorWrite<'_>,
) -> crate::Result<()> {
    validate_read_into_destination(op.label(), inputs, out)
}

/// Validate that a caller-owned destination does not overlap any read input.
///
/// The check is intentionally conservative for host views: two views backed by
/// the same host allocation are treated as overlapping because the allocation
/// identity is the only stable boundary contract available to erased backend
/// code. Backend allocations use their domain/allocation identity when the
/// provider exposes it.
///
/// # Errors
///
/// Returns `tenferro_tensor_core::ValidationError::InvalidArgument` when the
/// destination storage overlaps an input, or `Error::RuntimeState` when
/// storage identity cannot be established safely.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{Tensor, TensorRead, TensorWrite};
/// use tenferro_tensor::backend::validate_read_into_destination;
///
/// let input = Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?;
/// let mut output = Tensor::from_vec_col_major(vec![1], vec![0.0_f64])?;
/// validate_read_into_destination(
///     "example",
///     &[TensorRead::from_tensor(&input)],
///     &TensorWrite::from_tensor(&mut output),
/// )?;
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub fn validate_read_into_destination(
    op: &'static str,
    inputs: &[TensorRead<'_>],
    out: &TensorWrite<'_>,
) -> crate::Result<()> {
    let output_identity = tensor_read_storage_identity(&out.as_read())?;
    for (index, input) in inputs.iter().enumerate() {
        if storage_overlaps(tensor_read_storage_identity(input)?, output_identity) {
            return Err(Error::invalid_argument(
                op,
                "out",
                format!("destination storage overlaps input {index}"),
            ));
        }
    }
    Ok(())
}

fn read_is_host(input: &TensorRead<'_>) -> bool {
    match input {
        TensorRead::Tensor(tensor) => !tensor.is_backend_buffer(),
        TensorRead::View(view) => match view {
            TensorView::F32(view) => view.backend_buffer().is_none(),
            TensorView::F64(view) => view.backend_buffer().is_none(),
            TensorView::I32(view) => view.backend_buffer().is_none(),
            TensorView::I64(view) => view.backend_buffer().is_none(),
            TensorView::Bool(view) => view.backend_buffer().is_none(),
            TensorView::C32(view) => view.backend_buffer().is_none(),
            TensorView::C64(view) => view.backend_buffer().is_none(),
        },
    }
}

fn write_is_host(out: &TensorWrite<'_>) -> bool {
    read_is_host(&out.as_read())
}

fn one_shot_supports(op: ElementwiseReadOp, dtype: DType) -> bool {
    match op {
        ElementwiseReadOp::Conj => true,
        ElementwiseReadOp::Add
        | ElementwiseReadOp::Subtract
        | ElementwiseReadOp::Multiply
        | ElementwiseReadOp::Divide
        | ElementwiseReadOp::Negate => !matches!(dtype, DType::Bool),
    }
}

fn one_shot_eligible(
    op: ElementwiseReadOp,
    inputs: &[TensorRead<'_>],
    out: &TensorWrite<'_>,
) -> bool {
    let dtype = out.dtype();
    write_is_host(out)
        && one_shot_supports(op, dtype)
        && inputs.iter().all(|input| {
            read_is_host(input) && input.dtype() == dtype && input.shape() == out.shape()
        })
}

fn tensor_write_view(out: TensorWrite<'_>) -> TensorViewMut<'_> {
    match out {
        TensorWrite::Tensor(tensor) => match tensor {
            Tensor::F32(tensor) => TensorViewMut::F32(tensor.as_view_mut()),
            Tensor::F64(tensor) => TensorViewMut::F64(tensor.as_view_mut()),
            Tensor::I32(tensor) => TensorViewMut::I32(tensor.as_view_mut()),
            Tensor::I64(tensor) => TensorViewMut::I64(tensor.as_view_mut()),
            Tensor::Bool(tensor) => TensorViewMut::Bool(tensor.as_view_mut()),
            Tensor::C32(tensor) => TensorViewMut::C32(tensor.as_view_mut()),
            Tensor::C64(tensor) => TensorViewMut::C64(tensor.as_view_mut()),
        },
        TensorWrite::View(view) => view,
    }
}

fn non_null_bytes<T>(data: &[T]) -> NonNull<u8> {
    NonNull::new(data.as_ptr().cast_mut().cast()).unwrap_or_else(NonNull::dangling)
}

fn typed_bytes<T>(data: &[T]) -> &[u8] {
    // SAFETY: u8 has alignment one and the returned bytes retain the shared
    // lifetime of the typed source slice.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), std::mem::size_of_val(data)) }
}

fn typed_bytes_mut<T>(data: &mut [T]) -> &mut [u8] {
    let len = std::mem::size_of_val(data);
    // SAFETY: u8 has alignment one and the returned bytes retain the unique
    // lifetime of the typed destination slice.
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), len) }
}

fn erased_raw_strided_ptr<'a>(
    dtype: KernelDType,
    data: &'a [u8],
    dims: &'a [usize],
    strides: &'a [isize],
    offset: isize,
) -> strided_kernel::Result<ErasedRawStridedPtr<'a>> {
    // SAFETY: callers derive `data` from initialized typed host storage and
    // retain the backing borrow for the returned descriptor lifetime.
    unsafe {
        ErasedRawStridedPtr::from_raw_parts(
            dtype,
            non_null_bytes(data),
            data.len(),
            dims,
            strides,
            offset,
        )
    }
}

fn erased_raw_strided_mut<'a>(
    dtype: KernelDType,
    data: &'a mut [u8],
    dims: &'a [usize],
    strides: &'a [isize],
    offset: isize,
) -> strided_kernel::Result<ErasedRawStridedMut<'a>> {
    let data_ptr = NonNull::new(data.as_mut_ptr()).unwrap_or_else(NonNull::dangling);
    // SAFETY: callers derive `data` from a uniquely borrowed initialized host
    // destination and retain that borrow for the returned descriptor lifetime.
    unsafe {
        ErasedRawStridedMut::from_raw_parts(dtype, data_ptr, data.len(), dims, strides, offset)
    }
}

fn execute_one_shot_map<T: 'static>(
    dtype: KernelDType,
    op: ErasedMapOp,
    ctx: &ExecContext,
    input: TypedTensorView<'_, T>,
    mut out: TypedTensorViewMut<'_, T>,
) -> crate::Result<()> {
    let input_data = input.host_storage()?;
    // INVARIANT: dtype and layout come from the same validated typed view, and
    // its host storage remains borrowed until replay returns.
    // SAFETY: input_data supplies the pointer and exact byte length; the view
    // owns the matching shape, signed strides, and in-bounds offset.
    let input_descriptor = erased_raw_strided_ptr(
        dtype,
        typed_bytes(input_data),
        input.shape(),
        input.strides(),
        input.offset(),
    )
    .map_err(|error| Error::backend_source("elementwise_read_into", error))?;

    let out_dims = SmallVec::<[usize; 8]>::from_slice(out.shape());
    let out_strides = SmallVec::<[isize; 8]>::from_slice(out.strides());
    let out_offset = out.offset();
    let out_data = out.host_storage_mut()?;
    // INVARIANT: the copied output layout describes this uniquely borrowed
    // host storage, already validated as disjoint from every input.
    let mut out_descriptor = erased_raw_strided_mut(
        dtype,
        typed_bytes_mut(out_data),
        &out_dims,
        &out_strides,
        out_offset,
    )
    .map_err(|error| Error::backend_source("elementwise_read_into", error))?;
    erased_map_into(dtype, op, ctx, &mut out_descriptor, &input_descriptor)
        .map_err(|error| Error::backend_source("elementwise_read_into", error))
}

fn execute_one_shot_zip<T: 'static>(
    dtype: KernelDType,
    op: ErasedZipOp,
    ctx: &ExecContext,
    lhs: TypedTensorView<'_, T>,
    rhs: TypedTensorView<'_, T>,
    mut out: TypedTensorViewMut<'_, T>,
) -> crate::Result<()> {
    let lhs_data = lhs.host_storage()?;
    // INVARIANT: dtype and layout come from the same validated typed view, and
    // its host storage remains borrowed until replay returns.
    // SAFETY: lhs_data supplies the pointer and exact byte length; the view
    // owns the matching shape, signed strides, and in-bounds offset.
    let lhs_descriptor = erased_raw_strided_ptr(
        dtype,
        typed_bytes(lhs_data),
        lhs.shape(),
        lhs.strides(),
        lhs.offset(),
    )
    .map_err(|error| Error::backend_source("elementwise_read_into", error))?;
    let rhs_data = rhs.host_storage()?;
    // INVARIANT: dtype and layout come from the same validated typed view, and
    // its host storage remains borrowed until replay returns.
    // SAFETY: rhs_data supplies the pointer and exact byte length; the view
    // owns the matching shape, signed strides, and in-bounds offset.
    let rhs_descriptor = erased_raw_strided_ptr(
        dtype,
        typed_bytes(rhs_data),
        rhs.shape(),
        rhs.strides(),
        rhs.offset(),
    )
    .map_err(|error| Error::backend_source("elementwise_read_into", error))?;

    let out_dims = SmallVec::<[usize; 8]>::from_slice(out.shape());
    let out_strides = SmallVec::<[isize; 8]>::from_slice(out.strides());
    let out_offset = out.offset();
    let out_data = out.host_storage_mut()?;
    // INVARIANT: the copied output layout describes this uniquely borrowed
    // host storage, already validated as disjoint from every input.
    let mut out_descriptor = erased_raw_strided_mut(
        dtype,
        typed_bytes_mut(out_data),
        &out_dims,
        &out_strides,
        out_offset,
    )
    .map_err(|error| Error::backend_source("elementwise_read_into", error))?;
    erased_zip_into(
        dtype,
        op,
        ctx,
        &mut out_descriptor,
        &lhs_descriptor,
        &rhs_descriptor,
    )
    .map_err(|error| Error::backend_source("elementwise_read_into", error))
}

fn execute_one_shot_elementwise(
    op: ElementwiseReadOp,
    inputs: &[TensorRead<'_>],
    out: TensorWrite<'_>,
    ctx: &ExecContext,
) -> crate::Result<()> {
    let out = tensor_write_view(out);
    macro_rules! dispatch_map {
        ($map_op:expr) => {{
            let input = inputs[0].clone().tensor_view();
            match (input, out) {
                (TensorView::F32(input), TensorViewMut::F32(out)) => {
                    execute_one_shot_map(KernelDType::F32, $map_op, ctx, input, out)
                }
                (TensorView::F64(input), TensorViewMut::F64(out)) => {
                    execute_one_shot_map(KernelDType::F64, $map_op, ctx, input, out)
                }
                (TensorView::I32(input), TensorViewMut::I32(out)) => {
                    execute_one_shot_map(KernelDType::I32, $map_op, ctx, input, out)
                }
                (TensorView::I64(input), TensorViewMut::I64(out)) => {
                    execute_one_shot_map(KernelDType::I64, $map_op, ctx, input, out)
                }
                (TensorView::Bool(input), TensorViewMut::Bool(out)) => {
                    execute_one_shot_map(KernelDType::Bool, $map_op, ctx, input, out)
                }
                (TensorView::C32(input), TensorViewMut::C32(out)) => {
                    execute_one_shot_map(KernelDType::C32, $map_op, ctx, input, out)
                }
                (TensorView::C64(input), TensorViewMut::C64(out)) => {
                    execute_one_shot_map(KernelDType::C64, $map_op, ctx, input, out)
                }
                _ => unreachable!("one-shot eligibility validates matching dtypes"),
            }
        }};
    }
    macro_rules! dispatch_zip {
        ($zip_op:expr) => {{
            let lhs = inputs[0].clone().tensor_view();
            let rhs = inputs[1].clone().tensor_view();
            match (lhs, rhs, out) {
                (TensorView::F32(lhs), TensorView::F32(rhs), TensorViewMut::F32(out)) => {
                    execute_one_shot_zip(KernelDType::F32, $zip_op, ctx, lhs, rhs, out)
                }
                (TensorView::F64(lhs), TensorView::F64(rhs), TensorViewMut::F64(out)) => {
                    execute_one_shot_zip(KernelDType::F64, $zip_op, ctx, lhs, rhs, out)
                }
                (TensorView::I32(lhs), TensorView::I32(rhs), TensorViewMut::I32(out)) => {
                    execute_one_shot_zip(KernelDType::I32, $zip_op, ctx, lhs, rhs, out)
                }
                (TensorView::I64(lhs), TensorView::I64(rhs), TensorViewMut::I64(out)) => {
                    execute_one_shot_zip(KernelDType::I64, $zip_op, ctx, lhs, rhs, out)
                }
                (TensorView::C32(lhs), TensorView::C32(rhs), TensorViewMut::C32(out)) => {
                    execute_one_shot_zip(KernelDType::C32, $zip_op, ctx, lhs, rhs, out)
                }
                (TensorView::C64(lhs), TensorView::C64(rhs), TensorViewMut::C64(out)) => {
                    execute_one_shot_zip(KernelDType::C64, $zip_op, ctx, lhs, rhs, out)
                }
                _ => unreachable!("one-shot eligibility validates matching dtypes"),
            }
        }};
    }

    match op {
        ElementwiseReadOp::Add => dispatch_zip!(ErasedZipOp::Add),
        ElementwiseReadOp::Subtract => dispatch_zip!(ErasedZipOp::Subtract),
        ElementwiseReadOp::Multiply => dispatch_zip!(ErasedZipOp::Multiply),
        ElementwiseReadOp::Divide => dispatch_zip!(ErasedZipOp::Divide),
        ElementwiseReadOp::Negate => dispatch_map!(ErasedMapOp::Negate),
        ElementwiseReadOp::Conj => dispatch_map!(ErasedMapOp::Conj),
    }
}

/// Execute the shared elementwise-into path with an explicit replay context.
///
/// This is backend glue for implementations that own an execution context.
///
/// # Errors
///
/// Returns [`crate::Error::Validation`] when the input arity or tensor
/// metadata is invalid, or when the destination overlaps an input. Returns
/// [`crate::Error::BackendSource`] when an eligible strided replay fails.
/// Errors returned by `fallback` are preserved unchanged.
#[doc(hidden)]
pub fn elementwise_read_into_with_context(
    op: ElementwiseReadOp,
    inputs: &[TensorRead<'_>],
    out: TensorWrite<'_>,
    ctx: &ExecContext,
    fallback: impl FnOnce(&[TensorRead<'_>], TensorWrite<'_>) -> crate::Result<()>,
) -> crate::Result<()> {
    if inputs.len() != op.arity() {
        return Err(Error::invalid_argument(
            op.label(),
            "inputs",
            format!("expected {} inputs, got {}", op.arity(), inputs.len()),
        ));
    }
    validate_elementwise_output_disjoint(op, inputs, &out)?;
    if one_shot_eligible(op, inputs, &out) {
        execute_one_shot_elementwise(op, inputs, out, ctx)
    } else {
        fallback(inputs, out)
    }
}

/// Elementwise tensor operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::TensorElementwise;
///
/// fn accepts_elementwise<B: TensorElementwise>(_backend: &mut B) {}
/// ```
pub trait TensorElementwise: TensorStructural {
    /// Execute an elementwise operation into caller-owned storage.
    ///
    /// Backend implementations normally override this hook only to inject
    /// their explicit execution context and buffer policy. The default uses a
    /// serial host one-shot kernel and preserves the allocating fallback for
    /// device storage, dtype promotion, and broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] when `inputs` has the wrong arity,
    /// tensor metadata is invalid, or the destination overlaps an input.
    /// Returns [`crate::Error::BackendSource`] when the strided kernel rejects
    /// an eligible host operation. Errors from the allocating backend fallback
    /// are preserved unchanged.
    fn elementwise_read_into(
        &mut self,
        op: ElementwiseReadOp,
        inputs: &[TensorRead<'_>],
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        let ctx = ExecContext::serial();
        elementwise_read_into_with_context(op, inputs, out, &ctx, |inputs, out| {
            let result = match op {
                ElementwiseReadOp::Add => self.add_read(inputs[0].clone(), inputs[1].clone())?,
                ElementwiseReadOp::Subtract => {
                    self.sub_read(inputs[0].clone(), inputs[1].clone())?
                }
                ElementwiseReadOp::Multiply => {
                    self.mul_read(inputs[0].clone(), inputs[1].clone())?
                }
                ElementwiseReadOp::Negate => self.neg_read(inputs[0].clone())?,
                ElementwiseReadOp::Conj => self.conj_read(inputs[0].clone())?,
                ElementwiseReadOp::Divide => self.div_read(inputs[0].clone(), inputs[1].clone())?,
            };
            self.copy_read_into(TensorRead::from_tensor(&result), out)
        })
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn add(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;

    /// Elementwise addition accepting either owned tensors or borrowed views.
    ///
    /// Backends that implement this method must not silently move data across
    /// devices. A backend that cannot consume views should return an explicit
    /// backend error rather than materializing or transferring implicitly.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Tensor, TensorElementwise, TensorRead};
    ///
    /// fn add_owned<B: TensorElementwise>(
    ///     backend: &mut B,
    ///     lhs: &Tensor,
    ///     rhs: &Tensor,
    /// ) -> tenferro_tensor::Result<Tensor> {
    ///     backend.add_read(TensorRead::from_tensor(lhs), TensorRead::from_tensor(rhs))
    /// }
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn add_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.add(read_tensor("add", lhs)?, read_tensor("add", rhs)?)
    }

    /// Overwrite caller-provided output with elementwise addition.
    ///
    /// `_into` methods never accumulate into the previous output value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Tensor, TensorElementwise, TensorWrite};
    ///
    /// fn add_into<B: TensorElementwise>(
    ///     backend: &mut B,
    ///     lhs: &Tensor,
    ///     rhs: &Tensor,
    ///     mut out: Tensor,
    /// ) -> tenferro_tensor::Result<Tensor> {
    ///     backend.add_into(lhs, rhs, TensorWrite::from_tensor(&mut out))?;
    ///     Ok(out)
    /// }
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn add_into(&mut self, lhs: &Tensor, rhs: &Tensor, out: TensorWrite<'_>) -> crate::Result<()> {
        self.add_read_into(
            TensorRead::from_tensor(lhs),
            TensorRead::from_tensor(rhs),
            out,
        )
    }

    /// Overwrite caller-provided output with elementwise addition from reads.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{TensorElementwise, TensorRead, TensorWrite};
    ///
    /// fn add_read_into<B: TensorElementwise>(
    ///     backend: &mut B,
    ///     lhs: TensorRead<'_>,
    ///     rhs: TensorRead<'_>,
    ///     out: TensorWrite<'_>,
    /// ) -> tenferro_tensor::Result<()> {
    ///     backend.add_read_into(lhs, rhs, out)
    /// }
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn add_read_into(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        self.elementwise_read_into(ElementwiseReadOp::Add, &[lhs, rhs], out)
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn sub(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;

    /// Elementwise subtraction accepting either owned tensors or borrowed views.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Tensor, TensorElementwise, TensorRead};
    ///
    /// fn sub_owned<B: TensorElementwise>(
    ///     backend: &mut B,
    ///     lhs: &Tensor,
    ///     rhs: &Tensor,
    /// ) -> tenferro_tensor::Result<Tensor> {
    ///     backend.sub_read(TensorRead::from_tensor(lhs), TensorRead::from_tensor(rhs))
    /// }
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn sub_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.sub(read_tensor("sub", lhs)?, read_tensor("sub", rhs)?)
    }

    /// Overwrite caller-provided output with elementwise subtraction.
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn sub_into(&mut self, lhs: &Tensor, rhs: &Tensor, out: TensorWrite<'_>) -> crate::Result<()> {
        self.sub_read_into(
            TensorRead::from_tensor(lhs),
            TensorRead::from_tensor(rhs),
            out,
        )
    }

    /// Overwrite caller-provided output with elementwise subtraction from reads.
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn sub_read_into(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        self.elementwise_read_into(ElementwiseReadOp::Subtract, &[lhs, rhs], out)
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn mul(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn mul_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.mul(read_tensor("mul", lhs)?, read_tensor("mul", rhs)?)
    }

    /// Overwrite caller-provided output with elementwise multiplication.
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn mul_into(&mut self, lhs: &Tensor, rhs: &Tensor, out: TensorWrite<'_>) -> crate::Result<()> {
        self.mul_read_into(
            TensorRead::from_tensor(lhs),
            TensorRead::from_tensor(rhs),
            out,
        )
    }

    /// Overwrite caller-provided output with elementwise multiplication from reads.
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn mul_read_into(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        self.elementwise_read_into(ElementwiseReadOp::Multiply, &[lhs, rhs], out)
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn neg(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn neg_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.neg(read_tensor("neg", input)?)
    }

    /// Overwrite caller-provided output with elementwise negation.
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn neg_into(&mut self, input: &Tensor, out: TensorWrite<'_>) -> crate::Result<()> {
        self.neg_read_into(TensorRead::from_tensor(input), out)
    }

    /// Overwrite caller-provided output with elementwise negation from a read.
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn neg_read_into(&mut self, input: TensorRead<'_>, out: TensorWrite<'_>) -> crate::Result<()> {
        self.elementwise_read_into(ElementwiseReadOp::Negate, &[input], out)
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn conj(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn conj_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.conj(read_tensor("conj", input)?)
    }

    /// Overwrite caller-provided output with elementwise conjugation.
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn conj_into(&mut self, input: &Tensor, out: TensorWrite<'_>) -> crate::Result<()> {
        self.conj_read_into(TensorRead::from_tensor(input), out)
    }

    /// Overwrite caller-provided output with elementwise conjugation from a read.
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn conj_read_into(&mut self, input: TensorRead<'_>, out: TensorWrite<'_>) -> crate::Result<()> {
        self.elementwise_read_into(ElementwiseReadOp::Conj, &[input], out)
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn div(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn div_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.div(read_tensor("div", lhs)?, read_tensor("div", rhs)?)
    }

    /// Overwrite caller-provided output with elementwise division.
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn div_into(&mut self, lhs: &Tensor, rhs: &Tensor, out: TensorWrite<'_>) -> crate::Result<()> {
        self.div_read_into(
            TensorRead::from_tensor(lhs),
            TensorRead::from_tensor(rhs),
            out,
        )
    }

    /// Overwrite caller-provided output with elementwise division from reads.
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn div_read_into(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        self.elementwise_read_into(ElementwiseReadOp::Divide, &[lhs, rhs], out)
    }

    /// Elementwise remainder.
    ///
    /// The default is an explicit unsupported error so backend implementors can
    /// opt in without silent fallback.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Tensor, TensorElementwise};
    ///
    /// fn rem_owned<B: TensorElementwise>(
    ///     backend: &mut B,
    ///     lhs: &Tensor,
    ///     rhs: &Tensor,
    /// ) -> tenferro_tensor::Result<Tensor> {
    ///     backend.rem(lhs, rhs)
    /// }
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn rem(&mut self, lhs: &Tensor, _rhs: &Tensor) -> crate::Result<Tensor> {
        Err(crate::Error::unsupported(
            "rem",
            format!("backend does not implement rem for dtype {:?}", lhs.dtype()),
        ))
    }

    /// Elementwise remainder accepting owned tensors or borrowed views.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Tensor, TensorElementwise, TensorRead};
    ///
    /// fn rem_read<B: TensorElementwise>(
    ///     backend: &mut B,
    ///     lhs: &Tensor,
    ///     rhs: &Tensor,
    /// ) -> tenferro_tensor::Result<Tensor> {
    ///     backend.rem_read(TensorRead::from_tensor(lhs), TensorRead::from_tensor(rhs))
    /// }
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn rem_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.rem(read_tensor("rem", lhs)?, read_tensor("rem", rhs)?)
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn abs(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn abs_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.abs(read_tensor("abs", input)?)
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn sign(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn sign_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.sign(read_tensor("sign", input)?)
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn maximum(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn maximum_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.maximum(read_tensor("maximum", lhs)?, read_tensor("maximum", rhs)?)
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn minimum(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn minimum_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.minimum(read_tensor("minimum", lhs)?, read_tensor("minimum", rhs)?)
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn compare(&mut self, lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn compare_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        dir: &CompareDir,
    ) -> crate::Result<Tensor> {
        self.compare(
            read_tensor("compare", lhs)?,
            read_tensor("compare", rhs)?,
            dir,
        )
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn select(
        &mut self,
        pred: &Tensor,
        on_true: &Tensor,
        on_false: &Tensor,
    ) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn select_read(
        &mut self,
        pred: TensorRead<'_>,
        on_true: TensorRead<'_>,
        on_false: TensorRead<'_>,
    ) -> crate::Result<Tensor> {
        self.select(
            read_tensor("select", pred)?,
            read_tensor("select", on_true)?,
            read_tensor("select", on_false)?,
        )
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn clamp(&mut self, input: &Tensor, lower: &Tensor, upper: &Tensor) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn clamp_read(
        &mut self,
        input: TensorRead<'_>,
        lower: TensorRead<'_>,
        upper: TensorRead<'_>,
    ) -> crate::Result<Tensor> {
        self.clamp(
            read_tensor("clamp", input)?,
            read_tensor("clamp", lower)?,
            read_tensor("clamp", upper)?,
        )
    }
}

/// Analytic unary and binary tensor operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::TensorAnalytic;
///
/// fn accepts_analytic<B: TensorAnalytic>(_backend: &mut B) {}
/// ```
pub trait TensorAnalytic {
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn exp(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn exp_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.exp(read_tensor("exp", input)?)
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn log(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn log_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.log(read_tensor("log", input)?)
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn sin(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn sin_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.sin(read_tensor("sin", input)?)
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn cos(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn cos_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.cos(read_tensor("cos", input)?)
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn tanh(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn tanh_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.tanh(read_tensor("tanh", input)?)
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn sqrt(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn sqrt_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.sqrt(read_tensor("sqrt", input)?)
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn rsqrt(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn rsqrt_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.rsqrt(read_tensor("rsqrt", input)?)
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn pow(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn pow_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.pow(read_tensor("pow", lhs)?, read_tensor("pow", rhs)?)
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn expm1(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn expm1_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.expm1(read_tensor("expm1", input)?)
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn log1p(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn log1p_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.log1p(read_tensor("log1p", input)?)
    }
}

/// Shape, layout, and dtype transformation operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::TensorStructural;
///
/// fn accepts_structural<B: TensorStructural>(_backend: &mut B) {}
/// ```
pub trait TensorStructural {
    /// Materialize an owned tensor or borrowed view into fresh compact storage.
    ///
    /// The result has the input's shape and dtype, uses compact column-major
    /// layout, and remains in the input's placement. This operation is a
    /// same-placement canonicalization boundary, never an implicit host/device
    /// transfer. The conservative default accepts only compact host-owned
    /// tensors and clones them; it rejects views, backend buffers, and device
    /// placement because only an owning backend can materialize those safely.
    ///
    /// Backend overrides may accept strided views. CUDA accepts numeric and
    /// complex views on its active device, including arbitrary valid strides,
    /// but currently reports an explicit unsupported-dtype error for `Bool`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DType, Tensor, TensorRead, TensorStructural};
    ///
    /// struct HostDefaults;
    /// impl TensorStructural for HostDefaults {
    ///     fn transpose(&mut self, _: &Tensor, _: &[usize]) -> tenferro_tensor::Result<Tensor> { unimplemented!() }
    ///     fn reshape(&mut self, _: &Tensor, _: &[usize]) -> tenferro_tensor::Result<Tensor> { unimplemented!() }
    ///     fn broadcast_in_dim(&mut self, _: &Tensor, _: &[usize], _: &[usize]) -> tenferro_tensor::Result<Tensor> { unimplemented!() }
    ///     fn cast(&mut self, _: &Tensor, _: DType) -> tenferro_tensor::Result<Tensor> { unimplemented!() }
    ///     fn extract_diagonal(&mut self, _: &Tensor, _: usize, _: usize) -> tenferro_tensor::Result<Tensor> { unimplemented!() }
    ///     fn embed_diagonal(&mut self, _: &Tensor, _: usize, _: usize) -> tenferro_tensor::Result<Tensor> { unimplemented!() }
    ///     fn tril(&mut self, _: &Tensor, _: i64) -> tenferro_tensor::Result<Tensor> { unimplemented!() }
    ///     fn triu(&mut self, _: &Tensor, _: i64) -> tenferro_tensor::Result<Tensor> { unimplemented!() }
    /// }
    ///
    /// let input = Tensor::from_vec_col_major(vec![2], vec![1_i32, 2])?;
    /// let mut backend = HostDefaults;
    /// let structural: &mut dyn TensorStructural = &mut backend;
    /// let output = structural.to_contiguous_read(TensorRead::from_tensor(&input))?;
    /// assert_eq!(output.shape(), &[2]);
    /// assert_eq!(output.as_slice::<i32>()?, &[1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn to_contiguous_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        match input {
            TensorRead::Tensor(input) => {
                if input.is_backend_buffer()
                    || !matches!(
                        input.placement().memory_kind,
                        crate::MemoryKind::PinnedHost | crate::MemoryKind::UnpinnedHost
                    )
                {
                    return Err(crate::Error::runtime_state(
                        "to_contiguous_read",
                        "default materialization accepts only host-owned tensors; use the storage's owning backend",
                    ));
                }
                input.duplicate()
            }
            TensorRead::View(view) => {
                if view.backend_family().is_some()
                    || !matches!(
                        view.placement().memory_kind,
                        crate::MemoryKind::PinnedHost | crate::MemoryKind::UnpinnedHost
                    )
                {
                    return Err(crate::Error::runtime_state(
                        "to_contiguous_read",
                        "default materialization accepts only host-owned tensors; use the storage's owning backend",
                    ));
                }
                view.duplicate()
            }
        }
    }

    /// Overwrite caller-provided storage from a readable tensor or view.
    ///
    /// Source and destination must have identical dtype and shape and belong to
    /// the executing backend's placement. The destination is not resized, and
    /// every logical destination element is overwritten without reading its old
    /// value. Source and destination allocations must not alias. Implementations
    /// must not materialize through host memory or perform an implicit transfer.
    ///
    /// CPU accepts arbitrary valid source and destination strides and performs
    /// no tensor allocation. CUDA currently accepts only a compact column-major
    /// source with offset zero covering its full allocation; CUDA destinations
    /// may be arbitrary valid non-overlapping views. CUDA rejects aliased
    /// allocations and currently reports an explicit unsupported-dtype error
    /// for `Bool`. The conservative default is explicitly unsupported.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DType, Tensor, TensorRead, TensorStructural, TensorWrite};
    ///
    /// struct ConservativeDefaults;
    /// impl TensorStructural for ConservativeDefaults {
    ///     fn transpose(&mut self, _: &Tensor, _: &[usize]) -> tenferro_tensor::Result<Tensor> { unimplemented!() }
    ///     fn reshape(&mut self, _: &Tensor, _: &[usize]) -> tenferro_tensor::Result<Tensor> { unimplemented!() }
    ///     fn broadcast_in_dim(&mut self, _: &Tensor, _: &[usize], _: &[usize]) -> tenferro_tensor::Result<Tensor> { unimplemented!() }
    ///     fn cast(&mut self, _: &Tensor, _: DType) -> tenferro_tensor::Result<Tensor> { unimplemented!() }
    ///     fn extract_diagonal(&mut self, _: &Tensor, _: usize, _: usize) -> tenferro_tensor::Result<Tensor> { unimplemented!() }
    ///     fn embed_diagonal(&mut self, _: &Tensor, _: usize, _: usize) -> tenferro_tensor::Result<Tensor> { unimplemented!() }
    ///     fn tril(&mut self, _: &Tensor, _: i64) -> tenferro_tensor::Result<Tensor> { unimplemented!() }
    ///     fn triu(&mut self, _: &Tensor, _: i64) -> tenferro_tensor::Result<Tensor> { unimplemented!() }
    /// }
    ///
    /// let src = Tensor::from_vec_col_major(vec![2], vec![1_i32, 2])?;
    /// let mut dst = Tensor::from_vec_col_major(vec![2], vec![0_i32, 0])?;
    /// let mut backend = ConservativeDefaults;
    /// let structural: &mut dyn TensorStructural = &mut backend;
    /// let error = structural.copy_read_into(
    ///     TensorRead::from_tensor(&src),
    ///     TensorWrite::from_tensor(&mut dst),
    /// ).unwrap_err();
    /// assert!(error.to_string().contains("unsupported"));
    /// assert_eq!(dst.as_slice::<i32>()?, &[0, 0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn copy_read_into(&mut self, _src: TensorRead<'_>, _dst: TensorWrite<'_>) -> crate::Result<()> {
        Err(crate::Error::unsupported(
            "copy_read_into",
            "backend-owned runtime copy is unsupported by this backend",
        ))
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn transpose(&mut self, input: &Tensor, perm: &[usize]) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn transpose_read(&mut self, input: TensorRead<'_>, perm: &[usize]) -> crate::Result<Tensor> {
        self.transpose(read_tensor("transpose", input)?, perm)
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn reshape(&mut self, input: &Tensor, shape: &[usize]) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn reshape_read(&mut self, input: TensorRead<'_>, shape: &[usize]) -> crate::Result<Tensor> {
        self.reshape(read_tensor("reshape", input)?, shape)
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn broadcast_in_dim(
        &mut self,
        input: &Tensor,
        shape: &[usize],
        dims: &[usize],
    ) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn broadcast_in_dim_read(
        &mut self,
        input: TensorRead<'_>,
        shape: &[usize],
        dims: &[usize],
    ) -> crate::Result<Tensor> {
        self.broadcast_in_dim(read_tensor("broadcast_in_dim", input)?, shape, dims)
    }

    /// Cast a tensor to another dtype using explicit dtype projection.
    ///
    /// Backends may truncate, narrow precision, project complex values, or use
    /// boolean truthiness according to their documented cast support.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DType, Tensor, TensorStructural};
    ///
    /// fn cast_to_i32<B: TensorStructural>(
    ///     backend: &mut B,
    ///     input: &Tensor,
    /// ) -> tenferro_tensor::Result<Tensor> {
    ///     backend.cast(input, DType::I32)
    /// }
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn cast(&mut self, input: &Tensor, to: crate::DType) -> crate::Result<Tensor>;

    /// Convert a tensor to another dtype using checked dtype conversion.
    ///
    /// `convert` accepts only conversions allowed by tenferro's dtype-promotion
    /// lattice. Use [`TensorStructural::cast`] for explicit lossy projection.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DType, Tensor, TensorStructural};
    ///
    /// fn convert_to_f64<B: TensorStructural>(
    ///     backend: &mut B,
    ///     input: &Tensor,
    /// ) -> tenferro_tensor::Result<Tensor> {
    ///     backend.convert(input, DType::F64)
    /// }
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn convert(&mut self, input: &Tensor, to: crate::DType) -> crate::Result<Tensor> {
        validate_convert_dtype("convert", input.dtype(), to)?;
        self.cast(input, to)
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn extract_diagonal(
        &mut self,
        input: &Tensor,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn embed_diagonal(
        &mut self,
        input: &Tensor,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn tril(&mut self, input: &Tensor, k: i64) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn triu(&mut self, input: &Tensor, k: i64) -> crate::Result<Tensor>;
}

/// Reduction operations.
///
/// Reducing over an axis whose extent is zero returns an error for every
/// reduction operation. Passing an empty `axes` slice is a no-op for the public
/// reductions and returns the input values unchanged. Internal mapped
/// reductions document their own empty-axis semantics.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::TensorReduction;
///
/// fn accepts_reduction<B: TensorReduction>(_backend: &mut B) {}
/// ```
pub trait TensorReduction {
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn reduce_sum(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;

    /// Sum elements across axes from an owned tensor or borrowed view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Tensor, TensorRead, TensorReduction};
    ///
    /// fn sum_owned<B: TensorReduction>(
    ///     backend: &mut B,
    ///     input: &Tensor,
    /// ) -> tenferro_tensor::Result<Tensor> {
    ///     backend.reduce_sum_read(TensorRead::from_tensor(input), &[0])
    /// }
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn reduce_sum_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        match input.as_tensor() {
            Some(input) => self.reduce_sum(input, axes),
            None => Err(crate::Error::unsupported(
                "reduce_sum",
                "backend does not accept borrowed tensor views at this execution boundary",
            )),
        }
    }

    /// Sum elementwise squares across axes.
    ///
    /// This execution hook is used by composite operations that avoid a
    /// materialized square. Empty axes produce an elementwise square. Backends
    /// that support this optimized path must override the hook directly.
    ///
    /// # Errors
    ///
    /// Returns the typed validation, unsupported, runtime-state, or backend
    /// error produced by multiplication or reduction.
    #[doc(hidden)]
    fn reduce_sum_squares_read(
        &mut self,
        _input: TensorRead<'_>,
        _axes: &[usize],
    ) -> crate::Result<Tensor> {
        Err(crate::Error::unsupported(
            "reduce_sum_squares",
            "backend does not implement fused sum-of-squares reduction",
        ))
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn reduce_prod(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;

    /// Multiply elements across axes from an owned tensor or borrowed view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Tensor, TensorRead, TensorReduction};
    ///
    /// fn prod_owned<B: TensorReduction>(
    ///     backend: &mut B,
    ///     input: &Tensor,
    /// ) -> tenferro_tensor::Result<Tensor> {
    ///     backend.reduce_prod_read(TensorRead::from_tensor(input), &[0])
    /// }
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn reduce_prod_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        match input.as_tensor() {
            Some(input) => self.reduce_prod(input, axes),
            None => Err(crate::Error::unsupported(
                "reduce_prod",
                "backend does not accept borrowed tensor views at this execution boundary",
            )),
        }
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn reduce_max(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;

    /// Take maximum values across axes from an owned tensor or borrowed view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Tensor, TensorRead, TensorReduction};
    ///
    /// fn max_owned<B: TensorReduction>(
    ///     backend: &mut B,
    ///     input: &Tensor,
    /// ) -> tenferro_tensor::Result<Tensor> {
    ///     backend.reduce_max_read(TensorRead::from_tensor(input), &[0])
    /// }
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn reduce_max_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        match input.as_tensor() {
            Some(input) => self.reduce_max(input, axes),
            None => Err(crate::Error::unsupported(
                "reduce_max",
                "backend does not accept borrowed tensor views at this execution boundary",
            )),
        }
    }

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn reduce_min(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;

    /// Take minimum values across axes from an owned tensor or borrowed view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Tensor, TensorRead, TensorReduction};
    ///
    /// fn min_owned<B: TensorReduction>(
    ///     backend: &mut B,
    ///     input: &Tensor,
    /// ) -> tenferro_tensor::Result<Tensor> {
    ///     backend.reduce_min_read(TensorRead::from_tensor(input), &[0])
    /// }
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn reduce_min_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        match input.as_tensor() {
            Some(input) => self.reduce_min(input, axes),
            None => Err(crate::Error::unsupported(
                "reduce_min",
                "backend does not accept borrowed tensor views at this execution boundary",
            )),
        }
    }
}

/// Dot-general operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::TensorDot;
///
/// fn accepts_dot<B: TensorDot>(_backend: &mut B) {}
/// ```
pub trait TensorDot: TensorElementwise {
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn dot_general(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor>;

    #[doc(hidden)]
    fn dot_general_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        match (lhs.as_tensor(), rhs.as_tensor()) {
            (Some(lhs), Some(rhs)) => self.dot_general(lhs, rhs, config),
            _ => {
                let lhs = self.to_contiguous_read(lhs)?;
                let rhs = self.to_contiguous_read(rhs)?;
                self.dot_general(&lhs, &rhs, config)
            }
        }
    }

    /// Overwrite caller-provided output with dot-general from read inputs.
    ///
    /// This is the dot/GEMM spelling of `_into`: the previous output value is
    /// not read. Use [`TensorDot::dot_general_read_into_accum`] for explicit
    /// read-modify-write accumulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DotGeneralConfig, TensorDot, TensorRead, TensorWrite};
    ///
    /// fn dot_into<B: TensorDot>(
    ///     backend: &mut B,
    ///     lhs: TensorRead<'_>,
    ///     rhs: TensorRead<'_>,
    ///     config: &DotGeneralConfig,
    ///     out: TensorWrite<'_>,
    /// ) -> tenferro_tensor::Result<()> {
    ///     backend.dot_general_read_into(lhs, rhs, config, out)
    /// }
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn dot_general_read_into(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        let accumulation = DotGeneralAccumulation::overwrite(lhs.dtype())?;
        self.dot_general_read_into_accum(lhs, rhs, config, accumulation, out)
    }

    #[doc(hidden)]
    fn dot_general_with_conj(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        if !lhs_conj && !rhs_conj {
            return self.dot_general(lhs, rhs, config);
        }

        let lhs_tmp;
        let lhs_ref = if lhs_conj {
            lhs_tmp = self.conj(lhs)?;
            &lhs_tmp
        } else {
            lhs
        };
        let rhs_tmp;
        let rhs_ref = if rhs_conj {
            rhs_tmp = self.conj(rhs)?;
            &rhs_tmp
        } else {
            rhs
        };
        self.dot_general(lhs_ref, rhs_ref, config)
    }

    #[allow(clippy::too_many_arguments)]
    #[doc(hidden)]
    fn dot_general_with_conj_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        if !lhs_conj && !rhs_conj {
            return self.dot_general_read(lhs, rhs, config);
        }

        let lhs_tmp;
        let lhs_ref = if let Some(tensor) = lhs.as_tensor() {
            tensor
        } else {
            lhs_tmp = self.to_contiguous_read(lhs)?;
            &lhs_tmp
        };
        let rhs_tmp;
        let rhs_ref = if let Some(tensor) = rhs.as_tensor() {
            tensor
        } else {
            rhs_tmp = self.to_contiguous_read(rhs)?;
            &rhs_tmp
        };
        self.dot_general_with_conj(lhs_ref, rhs_ref, config, lhs_conj, rhs_conj)
    }

    /// Apply scaled dot-general accumulation into caller-provided output.
    ///
    /// This is explicitly read-modify-write when `accumulation.beta` is nonzero:
    /// `out = alpha * dot_general(lhs, rhs) + beta * out`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{
    ///     DotGeneralAccumulation, DotGeneralConfig, TensorDot, TensorRead, TensorWrite,
    /// };
    ///
    /// fn dot_add_to<B: TensorDot>(
    ///     backend: &mut B,
    ///     lhs: TensorRead<'_>,
    ///     rhs: TensorRead<'_>,
    ///     config: &DotGeneralConfig,
    ///     out: TensorWrite<'_>,
    /// ) -> tenferro_tensor::Result<()> {
    ///     let accumulation = DotGeneralAccumulation::add_to(lhs.dtype())?;
    ///     backend.dot_general_read_into_accum(lhs, rhs, config, accumulation, out)
    /// }
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn dot_general_read_into_accum(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        accumulation: DotGeneralAccumulation,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        dot_general_accum_via_temp(self, lhs, rhs, config, accumulation, out)
    }
}

/// Session-scoped cached dot-general operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::BackendSession;
///
/// fn accepts_session_dot<S: BackendSession + ?Sized>(_session: &mut S) {}
/// ```
pub trait SessionCachedDot: TensorDot {
    #[doc(hidden)]
    fn dot_general_cached(
        &mut self,
        _cache_slot: Option<usize>,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        self.dot_general(lhs, rhs, config)
    }

    #[doc(hidden)]
    fn dot_general_read_cached(
        &mut self,
        cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        match (lhs.as_tensor(), rhs.as_tensor()) {
            (Some(lhs), Some(rhs)) => self.dot_general_cached(cache_slot, lhs, rhs, config),
            _ => {
                let lhs = self.to_contiguous_read(lhs)?;
                let rhs = self.to_contiguous_read(rhs)?;
                self.dot_general_cached(cache_slot, &lhs, &rhs, config)
            }
        }
    }

    // Mirrors the dot-general signature plus runtime-cache metadata.
    #[allow(clippy::too_many_arguments)]
    #[doc(hidden)]
    fn dot_general_with_conj_cached(
        &mut self,
        _cache_slot: Option<usize>,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        self.dot_general_with_conj(lhs, rhs, config, lhs_conj, rhs_conj)
    }

    // Mirrors the dot-general read signature plus runtime-cache metadata.
    #[allow(clippy::too_many_arguments)]
    #[doc(hidden)]
    fn dot_general_with_conj_read_cached(
        &mut self,
        cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        if !lhs_conj && !rhs_conj {
            return self.dot_general_read_cached(cache_slot, lhs, rhs, config);
        }

        let lhs_tmp;
        let lhs_ref = if let Some(tensor) = lhs.as_tensor() {
            tensor
        } else {
            lhs_tmp = self.to_contiguous_read(lhs)?;
            &lhs_tmp
        };
        let rhs_tmp;
        let rhs_ref = if let Some(tensor) = rhs.as_tensor() {
            tensor
        } else {
            rhs_tmp = self.to_contiguous_read(rhs)?;
            &rhs_tmp
        };
        self.dot_general_with_conj_cached(cache_slot, lhs_ref, rhs_ref, config, lhs_conj, rhs_conj)
    }

    /// Apply session-cached scaled dot-general accumulation into output.
    ///
    /// The cache slot is session-local metadata; `accumulation` still controls
    /// overwrite versus read-modify-write semantics.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{
    ///     DotGeneralAccumulation, DotGeneralConfig, SessionCachedDot, TensorRead, TensorWrite,
    /// };
    ///
    /// fn session_cached_dot_add_to<S: SessionCachedDot + ?Sized>(
    ///     session: &mut S,
    ///     lhs: TensorRead<'_>,
    ///     rhs: TensorRead<'_>,
    ///     config: &DotGeneralConfig,
    ///     out: TensorWrite<'_>,
    /// ) -> tenferro_tensor::Result<()> {
    ///     let accumulation = DotGeneralAccumulation::add_to(lhs.dtype())?;
    ///     session.dot_general_read_into_accum_cached(
    ///         Some(0),
    ///         lhs,
    ///         rhs,
    ///         config,
    ///         accumulation,
    ///         out,
    ///     )
    /// }
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn dot_general_read_into_accum_cached(
        &mut self,
        _cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        accumulation: DotGeneralAccumulation,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        self.dot_general_read_into_accum(lhs, rhs, config, accumulation, out)
    }

    #[doc(hidden)]
    fn grouped_gemm_cached(
        &mut self,
        _cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &GroupedGemmConfig<'_>,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        grouped_gemm_default(self, lhs, rhs, config, out)
    }
}

/// Indexing, slicing, and padding operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::TensorIndexing;
///
/// fn accepts_indexing<B: TensorIndexing>(_backend: &mut B) {}
/// ```
pub trait TensorIndexing {
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn gather(
        &mut self,
        operand: &Tensor,
        start_indices: &Tensor,
        config: &GatherConfig,
    ) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn scatter(
        &mut self,
        operand: &Tensor,
        scatter_indices: &Tensor,
        updates: &Tensor,
        config: &ScatterConfig,
    ) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn slice(&mut self, input: &Tensor, config: &SliceConfig) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn dynamic_slice(
        &mut self,
        input: &Tensor,
        starts: &Tensor,
        slice_sizes: &[usize],
    ) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn dynamic_update_slice(
        &mut self,
        operand: &Tensor,
        update: &Tensor,
        starts: &Tensor,
    ) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn pad(&mut self, input: &Tensor, config: &PadConfig) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn concatenate(&mut self, inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor>;
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn reverse(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
}

/// Backend-owned canonicalization for typed tensor views.
///
/// Implementations must preserve the input placement family. CPU backends
/// canonicalize host views through explicit host copies and reject backend
/// buffers with a diagnostic that asks the caller to download first. GPU
/// backends canonicalize GPU-resident views on the same device and reject host
/// buffers with an upload hint.
///
/// [`TensorViewCanonicalization::copy_into`] requires source and destination
/// shapes, scalar dtypes, and placement families to match. The destination
/// view must be internally non-overlapping, and source and destination backing
/// allocations must not alias unless an implementation explicitly documents
/// and supports that case. Implementations may reject layouts their native
/// kernels cannot consume.
///
/// CUDA currently accepts only a compact column-major source view with offset
/// zero that covers its full allocation; arbitrary-stride destinations remain
/// supported. Canonicalization and copying are same-placement operations: they
/// must not perform hidden host/device transfers or silently materialize an
/// unsupported source layout.
///
/// This trait is intentionally separate from [`BackendSession`] so generic
/// typed methods do not change the object-safety contract of `dyn BackendSession`.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{DynRank, TensorViewCanonicalization, TypedTensor};
///
/// fn compact_i32<B: TensorViewCanonicalization<i32, DynRank>>(
///     backend: &mut B,
///     tensor: &TypedTensor<i32>,
/// ) -> tenferro_tensor::Result<TypedTensor<i32>> {
///     backend.to_contiguous(&tensor.as_view())
/// }
///
/// fn copy_i32<B: TensorViewCanonicalization<i32, DynRank>>(
///     backend: &mut B,
///     src: &TypedTensor<i32>,
///     dst: &mut TypedTensor<i32>,
/// ) -> tenferro_tensor::Result<()> {
///     backend.copy_into(&src.as_view(), &mut dst.as_view_mut())
/// }
/// ```
pub trait TensorViewCanonicalization<T: TensorScalar, R: TensorRank> {
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn to_contiguous(
        &mut self,
        view: &TypedTensorView<'_, T, R>,
    ) -> crate::Result<TypedTensor<T, R>>;

    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn copy_into(
        &mut self,
        src: &TypedTensorView<'_, T, R>,
        dst: &mut TypedTensorViewMut<'_, T, R>,
    ) -> crate::Result<()>;
}

/// Optional elementwise fusion execution.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::TensorFusion;
///
/// fn accepts_fusion<B: TensorFusion>(_backend: &mut B) {}
/// ```
pub trait TensorFusion {
    #[doc(hidden)]
    fn execute_elementwise_fusion(
        &mut self,
        _inputs: &[&Tensor],
        _plan: &ElementwiseFusionPlan,
    ) -> crate::Result<Option<Vec<Tensor>>> {
        Ok(None)
    }

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    fn execute_broadcast_multiply(
        &mut self,
        _lhs: TensorRead<'_>,
        _lhs_shape: &[usize],
        _lhs_dims: &[usize],
        _rhs: TensorRead<'_>,
        _rhs_shape: &[usize],
        _rhs_dims: &[usize],
    ) -> crate::Result<Option<Tensor>> {
        Ok(None)
    }

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    fn execute_broadcast_multiply_value(
        &mut self,
        lhs: TensorRead<'_>,
        lhs_shape: &[usize],
        lhs_dims: &[usize],
        rhs: TensorRead<'_>,
        rhs_shape: &[usize],
        rhs_dims: &[usize],
    ) -> crate::Result<Option<TensorValue>> {
        self.execute_broadcast_multiply(lhs, lhs_shape, lhs_dims, rhs, rhs_shape, rhs_dims)
            .map(|tensor| tensor.map(TensorValue::from_tensor))
    }
}

/// Backend buffer lifecycle operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::TensorBuffer;
///
/// fn accepts_buffer<B: TensorBuffer>(_backend: &mut B) {}
/// ```
pub trait TensorBuffer {
    fn reclaim_buffer(&mut self, _tensor: Tensor) {}
}

/// Device transfer operations on backend boundaries.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::TensorDeviceTransfer;
///
/// fn accepts_transfer<B: TensorDeviceTransfer>(_backend: &mut B) {}
/// ```
pub trait TensorDeviceTransfer {
    /// Explicitly copy a provider-owned read target into host storage.
    ///
    /// Implementations must not return the input unchanged or stage through an
    /// unrelated provider. A backend that cannot transfer the requested read
    /// target returns a typed unsupported error.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Unsupported`] when the implementation cannot
    /// perform the requested transfer, or a typed validation/backend error when
    /// the source cannot be read.
    fn download_to_host(&mut self, tensor: TensorRead<'_>) -> crate::Result<Tensor>;

    /// Explicitly copy a host read target into provider storage.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Unsupported`] when the implementation cannot
    /// perform the requested transfer, or a typed validation/backend error when
    /// the source cannot be read.
    fn upload_host_tensor(&mut self, tensor: TensorRead<'_>) -> crate::Result<Tensor>;
}

/// Runtime cache associated with a backend.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::BackendRuntimeCache;
///
/// fn accepts_runtime_cache<B: BackendRuntimeCache>(_backend: &B) {}
/// ```
pub trait BackendRuntimeCache {
    #[doc(hidden)]
    type RuntimeCache: RuntimeCacheControl + Send + Sync + 'static;
}

/// Backend-owned cached dot-general operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::BackendCachedDot;
///
/// fn accepts_backend_cached_dot<B: BackendCachedDot>(_backend: &mut B) {}
/// ```
pub trait BackendCachedDot: BackendRuntimeCache + TensorDot {
    #[doc(hidden)]
    fn dot_general_cached(
        &mut self,
        _cache: &mut Self::RuntimeCache,
        _cache_slot: Option<usize>,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        self.dot_general(lhs, rhs, config)
    }

    #[doc(hidden)]
    fn dot_general_read_cached(
        &mut self,
        cache: &mut Self::RuntimeCache,
        cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        match (lhs.as_tensor(), rhs.as_tensor()) {
            (Some(lhs), Some(rhs)) => self.dot_general_cached(cache, cache_slot, lhs, rhs, config),
            _ => {
                let lhs = self.to_contiguous_read(lhs)?;
                let rhs = self.to_contiguous_read(rhs)?;
                self.dot_general_cached(cache, cache_slot, &lhs, &rhs, config)
            }
        }
    }

    // Mirrors the dot-general signature plus runtime-cache metadata.
    #[allow(clippy::too_many_arguments)]
    #[doc(hidden)]
    fn dot_general_with_conj_cached(
        &mut self,
        _cache: &mut Self::RuntimeCache,
        _cache_slot: Option<usize>,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        self.dot_general_with_conj(lhs, rhs, config, lhs_conj, rhs_conj)
    }

    // Mirrors the dot-general read signature plus runtime-cache metadata.
    #[allow(clippy::too_many_arguments)]
    #[doc(hidden)]
    fn dot_general_with_conj_read_cached(
        &mut self,
        cache: &mut Self::RuntimeCache,
        cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        if !lhs_conj && !rhs_conj {
            return self.dot_general_read_cached(cache, cache_slot, lhs, rhs, config);
        }

        let lhs_tmp;
        let lhs_ref = if let Some(tensor) = lhs.as_tensor() {
            tensor
        } else {
            lhs_tmp = self.to_contiguous_read(lhs)?;
            &lhs_tmp
        };
        let rhs_tmp;
        let rhs_ref = if let Some(tensor) = rhs.as_tensor() {
            tensor
        } else {
            rhs_tmp = self.to_contiguous_read(rhs)?;
            &rhs_tmp
        };
        self.dot_general_with_conj_cached(
            cache, cache_slot, lhs_ref, rhs_ref, config, lhs_conj, rhs_conj,
        )
    }

    /// Apply cached scaled dot-general accumulation into caller-provided output.
    ///
    /// The cache slot identifies backend-local analysis metadata only; output
    /// semantics are still fully described by `accumulation`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{
    ///     BackendCachedDot, BackendRuntimeCache, DotGeneralAccumulation, DotGeneralConfig,
    ///     TensorRead, TensorWrite,
    /// };
    ///
    /// fn cached_dot_add_to<B: BackendCachedDot>(
    ///     backend: &mut B,
    ///     cache: &mut B::RuntimeCache,
    ///     lhs: TensorRead<'_>,
    ///     rhs: TensorRead<'_>,
    ///     config: &DotGeneralConfig,
    ///     out: TensorWrite<'_>,
    /// ) -> tenferro_tensor::Result<()>
    /// where
    ///     B: BackendRuntimeCache,
    /// {
    ///     let accumulation = DotGeneralAccumulation::add_to(lhs.dtype())?;
    ///     backend.dot_general_read_into_accum_cached(
    ///         cache,
    ///         Some(0),
    ///         lhs,
    ///         rhs,
    ///         config,
    ///         accumulation,
    ///         out,
    ///     )
    /// }
    /// ```
    #[allow(clippy::too_many_arguments)]
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a typed `ValidationError` source
    /// for invalid shapes, ranks, axes, dtypes, or output metadata. It returns
    /// [`crate::Error::BackendFailure`] or [`crate::Error::BackendSource`] when
    /// backend execution or storage access cannot provide the requested result.
    fn dot_general_read_into_accum_cached(
        &mut self,
        _cache: &mut Self::RuntimeCache,
        _cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        accumulation: DotGeneralAccumulation,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        self.dot_general_read_into_accum(lhs, rhs, config, accumulation, out)
    }

    #[doc(hidden)]
    fn grouped_gemm_cached(
        &mut self,
        _cache: &mut Self::RuntimeCache,
        _cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &GroupedGemmConfig<'_>,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        grouped_gemm_default(self, lhs, rhs, config, out)
    }
}

/// Backend execution-session entry points.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::BackendSessionHost;
///
/// fn accepts_session_host<B: BackendSessionHost>(_backend: &mut B) {}
/// ```
pub trait BackendSessionHost: BackendRuntimeCache {
    fn with_backend_session<R: Send>(
        &mut self,
        f: impl FnOnce(&mut dyn BackendSession) -> R + Send,
    ) -> R
    where
        Self: TensorBackend + BackendSessionIdentity + Sized,
    {
        default_backend_session(self, f)
    }

    #[doc(hidden)]
    fn with_backend_session_cached<R: Send>(
        &mut self,
        _cache: &mut Self::RuntimeCache,
        f: impl FnOnce(&mut dyn BackendSession) -> R + Send,
    ) -> R
    where
        Self: TensorBackend + BackendSessionIdentity + Sized,
    {
        self.with_backend_session(f)
    }
}

/// Operation capabilities shared by backends and backend sessions.
#[doc(hidden)]
pub trait TensorBackendOps:
    TensorElementwise
    + TensorAnalytic
    + TensorStructural
    + TensorReduction
    + TensorIndexing
    + TensorDot
    + TensorFusion
    + TensorBuffer
{
}

impl<T> TensorBackendOps for T where
    T: TensorElementwise
        + TensorAnalytic
        + TensorStructural
        + TensorReduction
        + TensorIndexing
        + TensorDot
        + TensorFusion
        + TensorBuffer
        + ?Sized
{
}

/// Execution session surface for dense tensor backends.
///
/// All operations run within a backend-owned execution scope such as a CPU
/// thread policy or a GPU stream. Individual ops must not try to re-enter that
/// scope.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{BackendSessionHost, Tensor, TypedTensor};
///
/// fn add_in_session<B: BackendSessionHost>(
///     backend: &mut B,
///     a: &Tensor,
///     b: &Tensor,
/// ) -> tenferro_tensor::Result<Tensor>
/// where
///     B: tenferro_tensor::TensorBackend,
/// {
///     backend.with_backend_session(|exec| exec.add(a, b))
/// }
/// ```
/// Identity contract for a concrete erased backend-session target.
///
/// Implementors must use a distinct zero-sized `'static` marker for each
/// concrete target that a backend leaf may reconstruct from an erased session.
#[doc(hidden)]
pub trait BackendSessionIdentity {
    type Marker: 'static;
}

pub trait BackendSession: TensorBackendOps + SessionCachedDot + TensorDeviceTransfer {
    /// Build-local identity for backend-extension session capability dispatch.
    #[doc(hidden)]
    fn session_type_id(&self) -> TypeId;

    /// Erased pointer used only by backend leaf crates for a checked session
    /// capability bridge. The pointer is borrowed for the lifetime of `self`.
    ///
    /// # Safety
    ///
    /// The implementation must return a pointer to the same value represented
    /// by `self`, and that pointer must remain valid and uniquely borrowed for
    /// the duration of the `&mut self` borrow. Backend leaf crates may use this
    /// contract to recover a concrete session capability after checking
    /// [`Self::session_type_id`].
    #[doc(hidden)]
    unsafe fn session_data_mut(&mut self) -> *mut ();
}

impl<T> BackendSession for T
where
    T: TensorBackendOps + SessionCachedDot + TensorDeviceTransfer + BackendSessionIdentity + Sized,
{
    fn session_type_id(&self) -> TypeId {
        TypeId::of::<T::Marker>()
    }

    unsafe fn session_data_mut(&mut self) -> *mut () {
        self as *mut T as *mut ()
    }
}

/// Standard runtime backend over dynamic [`Tensor`] values.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::TensorBackend;
///
/// fn accepts_backend<B: TensorBackend>(_backend: &mut B) {}
/// ```
pub trait TensorBackend:
    BackendRuntimeCache
    + BackendSessionIdentity
    + TensorBackendOps
    + BackendCachedDot
    + TensorDeviceTransfer
    + BackendSessionHost
{
}

impl<T> SessionCachedDot for T where T: TensorBackend + ?Sized {}

/// Run a closure using the backend itself as a default execution session.
///
/// This is suitable for backends whose individual ops already manage their own
/// execution context.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{default_backend_session, TensorBackend};
///
/// fn run_with_default_session<B: TensorBackend>(backend: &mut B) -> usize {
///     default_backend_session(backend, |_exec| 1usize)
/// }
/// ```
pub fn default_backend_session<B: TensorBackend + BackendSessionIdentity, R: Send>(
    backend: &mut B,
    f: impl FnOnce(&mut dyn BackendSession) -> R + Send,
) -> R {
    f(backend)
}
