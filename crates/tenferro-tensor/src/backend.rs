use crate::config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use crate::types::{
    Buffer, TensorRank, TensorScalar, TensorView, TensorViewMut, TypedTensor, TypedTensorView,
    TypedTensorViewMut,
};
use crate::validate::validate_convert_dtype;
use crate::{DType, RuntimeCacheControl, Tensor, TensorRead, TensorValue, TensorWrite};
use num_complex::{Complex32, Complex64};

fn read_boundary_error(op: &'static str) -> crate::Error {
    crate::Error::backend_failure(
        op,
        "backend does not accept borrowed tensor views at this execution boundary",
    )
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
            return Err(crate::Error::AxisOutOfBounds { op, axis, rank });
        }
        if seen[axis] {
            return Err(crate::Error::DuplicateAxis { op, axis, role });
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
            return Err(crate::Error::AxisRoleConflict {
                op,
                axis,
                first_role,
                second_role,
            });
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
        return Err(crate::Error::InvalidConfig {
            op,
            message: "lhs/rhs contracting dim counts differ".into(),
        });
    }
    if config.lhs_batch_dims.len() != config.rhs_batch_dims.len() {
        return Err(crate::Error::InvalidConfig {
            op,
            message: "lhs/rhs batch dim counts differ".into(),
        });
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
            return Err(crate::Error::ShapeMismatch {
                op,
                lhs: lhs_shape.to_vec(),
                rhs: rhs_shape.to_vec(),
            });
        }
    }
    for (&lhs_axis, &rhs_axis) in config.lhs_batch_dims.iter().zip(&config.rhs_batch_dims) {
        if lhs_shape[lhs_axis] != rhs_shape[rhs_axis] {
            return Err(crate::Error::ShapeMismatch {
                op,
                lhs: lhs_shape.to_vec(),
                rhs: rhs_shape.to_vec(),
            });
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
        return Err(crate::Error::DTypeMismatch {
            op,
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
        });
    }
    if out.dtype() != lhs.dtype() {
        return Err(crate::Error::DTypeMismatch {
            op,
            lhs: out.dtype(),
            rhs: lhs.dtype(),
        });
    }
    let expected = dot_general_output_shape(lhs.shape(), rhs.shape(), config, op)?;
    if out.shape() != expected.as_slice() {
        return Err(crate::Error::ShapeMismatch {
            op,
            lhs: out.shape().to_vec(),
            rhs: expected.clone(),
        });
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
    pub fn one(dtype: DType) -> crate::Result<Self> {
        match dtype {
            DType::F32 => Ok(Self::F32(1.0)),
            DType::F64 => Ok(Self::F64(1.0)),
            DType::C32 => Ok(Self::C32(Complex32::new(1.0, 0.0))),
            DType::C64 => Ok(Self::C64(Complex64::new(1.0, 0.0))),
            DType::I32 | DType::I64 | DType::Bool => Err(crate::Error::DTypeMismatch {
                op: "dot_general",
                lhs: dtype,
                rhs: DType::F32,
            }),
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
    pub fn zero(dtype: DType) -> crate::Result<Self> {
        match dtype {
            DType::F32 => Ok(Self::F32(0.0)),
            DType::F64 => Ok(Self::F64(0.0)),
            DType::C32 => Ok(Self::C32(Complex32::new(0.0, 0.0))),
            DType::C64 => Ok(Self::C64(Complex64::new(0.0, 0.0))),
            DType::I32 | DType::I64 | DType::Bool => Err(crate::Error::DTypeMismatch {
                op: "dot_general",
                lhs: dtype,
                rhs: DType::F32,
            }),
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
#[doc(hidden)]
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

    pub fn out_offset(&self) -> usize {
        self.out_offset
    }

    pub fn lhs_offset(&self) -> usize {
        self.lhs_offset
    }

    pub fn rhs_offset(&self) -> usize {
        self.rhs_offset
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn contracted(&self) -> usize {
        self.contracted
    }

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
    pub fn scaled(alpha: ContractionScalar, beta: ContractionScalar) -> crate::Result<Self> {
        if alpha.dtype() != beta.dtype() {
            return Err(crate::Error::DTypeMismatch {
                op: "dot_general",
                lhs: alpha.dtype(),
                rhs: beta.dtype(),
            });
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
                return Err(crate::Error::DTypeMismatch {
                    op: "dot_general",
                    lhs: scalar.dtype(),
                    rhs: dtype,
                });
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
        acc.checked_mul(dim)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op,
                message: format!("{role} logical element count overflows usize for shape {dims:?}"),
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
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| crate::Error::InvalidConfig {
            op,
            message: format!(
                "{role} matrix element count overflows usize: rows={rows} cols={cols}"
            ),
        })?;
    if len == 0 {
        return Ok(None);
    }
    let end = offset
        .checked_add(len)
        .ok_or_else(|| crate::Error::InvalidConfig {
            op,
            message: format!("{role} matrix range overflows usize: offset={offset} len={len}"),
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
        return Err(crate::Error::InvalidConfig {
            op,
            message: format!(
                "{role} matrix range {}..{} exceeds shared buffer logical length {len}",
                range.start, range.end
            ),
        });
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
        return Err(crate::Error::DTypeMismatch {
            op,
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
        });
    }
    if lhs.dtype() != out.dtype() {
        return Err(crate::Error::DTypeMismatch {
            op,
            lhs: lhs.dtype(),
            rhs: out.dtype(),
        });
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
            return Err(crate::Error::InvalidConfig {
                op,
                message: format!(
                    "grouped GEMM output range for job {idx} overlaps job {prev_idx} range {}..{}",
                    previous.start, previous.end
                ),
            });
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
    let offset = isize::try_from(offset).map_err(|_| crate::Error::InvalidConfig {
        op,
        message: format!("{role} offset {offset} does not fit in isize"),
    })?;
    base.checked_add(offset)
        .ok_or_else(|| crate::Error::InvalidConfig {
            op,
            message: format!("{role} offset overflows isize: base={base} offset={offset}"),
        })
}

fn dim_stride(op: &'static str, dim: usize, role: &'static str) -> crate::Result<isize> {
    isize::try_from(dim).map_err(|_| crate::Error::InvalidConfig {
        op,
        message: format!("{role} leading dimension {dim} does not fit in isize"),
    })
}

fn typed_read_storage<'a, T>(
    tensor: &'a TypedTensor<T>,
    op: &'static str,
) -> crate::Result<(&'a [T], isize)> {
    match tensor.buffer() {
        Buffer::Host(data) => Ok((data, 0)),
        Buffer::Backend(_) => Err(crate::Error::backend_failure(
            op,
            "grouped GEMM default path requires host-backed tensor storage",
        )),
    }
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
    Err(crate::Error::DTypeMismatch {
        op: "grouped_gemm",
        lhs: lhs.dtype(),
        rhs: out.dtype(),
    })
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

    Err(crate::Error::DTypeMismatch {
        op: "dot_general",
        lhs: accumulation.alpha.dtype(),
        rhs: dot.dtype(),
    })
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
    for (linear, dot_value) in dot.iter().copied().enumerate() {
        let indices = flat_to_multi_for_shape(out.shape(), linear);
        let output = out
            .get_mut(&indices)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op: "dot_general",
                message: format!("output index {indices:?} is outside accumulation target"),
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
    fn add_read_into(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        let result = self.add_read(lhs, rhs)?;
        self.copy_read_into(TensorRead::from_tensor(&result), out)
    }

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
    fn sub_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.sub(read_tensor("sub", lhs)?, read_tensor("sub", rhs)?)
    }

    /// Overwrite caller-provided output with elementwise subtraction.
    fn sub_into(&mut self, lhs: &Tensor, rhs: &Tensor, out: TensorWrite<'_>) -> crate::Result<()> {
        self.sub_read_into(
            TensorRead::from_tensor(lhs),
            TensorRead::from_tensor(rhs),
            out,
        )
    }

    /// Overwrite caller-provided output with elementwise subtraction from reads.
    fn sub_read_into(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        let result = self.sub_read(lhs, rhs)?;
        self.copy_read_into(TensorRead::from_tensor(&result), out)
    }

    fn mul(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn mul_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.mul(read_tensor("mul", lhs)?, read_tensor("mul", rhs)?)
    }

    /// Overwrite caller-provided output with elementwise multiplication.
    fn mul_into(&mut self, lhs: &Tensor, rhs: &Tensor, out: TensorWrite<'_>) -> crate::Result<()> {
        self.mul_read_into(
            TensorRead::from_tensor(lhs),
            TensorRead::from_tensor(rhs),
            out,
        )
    }

    /// Overwrite caller-provided output with elementwise multiplication from reads.
    fn mul_read_into(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        let result = self.mul_read(lhs, rhs)?;
        self.copy_read_into(TensorRead::from_tensor(&result), out)
    }

    fn neg(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn neg_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.neg(read_tensor("neg", input)?)
    }

    /// Overwrite caller-provided output with elementwise negation.
    fn neg_into(&mut self, input: &Tensor, out: TensorWrite<'_>) -> crate::Result<()> {
        self.neg_read_into(TensorRead::from_tensor(input), out)
    }

    /// Overwrite caller-provided output with elementwise negation from a read.
    fn neg_read_into(&mut self, input: TensorRead<'_>, out: TensorWrite<'_>) -> crate::Result<()> {
        let result = self.neg_read(input)?;
        self.copy_read_into(TensorRead::from_tensor(&result), out)
    }

    fn conj(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn conj_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.conj(read_tensor("conj", input)?)
    }

    /// Overwrite caller-provided output with elementwise conjugation.
    fn conj_into(&mut self, input: &Tensor, out: TensorWrite<'_>) -> crate::Result<()> {
        self.conj_read_into(TensorRead::from_tensor(input), out)
    }

    /// Overwrite caller-provided output with elementwise conjugation from a read.
    fn conj_read_into(&mut self, input: TensorRead<'_>, out: TensorWrite<'_>) -> crate::Result<()> {
        let result = self.conj_read(input)?;
        self.copy_read_into(TensorRead::from_tensor(&result), out)
    }

    fn div(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn div_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.div(read_tensor("div", lhs)?, read_tensor("div", rhs)?)
    }

    /// Overwrite caller-provided output with elementwise division.
    fn div_into(&mut self, lhs: &Tensor, rhs: &Tensor, out: TensorWrite<'_>) -> crate::Result<()> {
        self.div_read_into(
            TensorRead::from_tensor(lhs),
            TensorRead::from_tensor(rhs),
            out,
        )
    }

    /// Overwrite caller-provided output with elementwise division from reads.
    fn div_read_into(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        let result = self.div_read(lhs, rhs)?;
        self.copy_read_into(TensorRead::from_tensor(&result), out)
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
    fn rem(&mut self, lhs: &Tensor, _rhs: &Tensor) -> crate::Result<Tensor> {
        Err(crate::Error::backend_failure(
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
    fn rem_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.rem(read_tensor("rem", lhs)?, read_tensor("rem", rhs)?)
    }

    fn abs(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn abs_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.abs(read_tensor("abs", input)?)
    }

    fn sign(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn sign_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.sign(read_tensor("sign", input)?)
    }

    fn maximum(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn maximum_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.maximum(read_tensor("maximum", lhs)?, read_tensor("maximum", rhs)?)
    }

    fn minimum(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn minimum_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.minimum(read_tensor("minimum", lhs)?, read_tensor("minimum", rhs)?)
    }

    fn compare(&mut self, lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> crate::Result<Tensor>;
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

    fn select(
        &mut self,
        pred: &Tensor,
        on_true: &Tensor,
        on_false: &Tensor,
    ) -> crate::Result<Tensor>;
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

    fn clamp(&mut self, input: &Tensor, lower: &Tensor, upper: &Tensor) -> crate::Result<Tensor>;
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
    fn exp(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn exp_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.exp(read_tensor("exp", input)?)
    }

    fn log(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn log_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.log(read_tensor("log", input)?)
    }

    fn sin(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn sin_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.sin(read_tensor("sin", input)?)
    }

    fn cos(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn cos_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.cos(read_tensor("cos", input)?)
    }

    fn tanh(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn tanh_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.tanh(read_tensor("tanh", input)?)
    }

    fn sqrt(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn sqrt_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.sqrt(read_tensor("sqrt", input)?)
    }

    fn rsqrt(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn rsqrt_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.rsqrt(read_tensor("rsqrt", input)?)
    }

    fn pow(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn pow_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.pow(read_tensor("pow", lhs)?, read_tensor("pow", rhs)?)
    }

    fn expm1(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn expm1_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.expm1(read_tensor("expm1", input)?)
    }

    fn log1p(&mut self, input: &Tensor) -> crate::Result<Tensor>;
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
    fn to_contiguous_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        let input = read_tensor("to_contiguous_read", input)?;
        if input.is_backend_buffer()
            || !matches!(
                input.placement().memory_kind,
                crate::MemoryKind::PinnedHost | crate::MemoryKind::UnpinnedHost
            )
        {
            return Err(crate::Error::backend_failure(
                "to_contiguous_read",
                "default materialization accepts only host-owned tensors; use the storage's owning backend",
            ));
        }
        Ok(input.clone())
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
    fn copy_read_into(&mut self, _src: TensorRead<'_>, _dst: TensorWrite<'_>) -> crate::Result<()> {
        Err(crate::Error::backend_failure(
            "copy_read_into",
            "backend-owned runtime copy is unsupported by this backend",
        ))
    }

    fn transpose(&mut self, input: &Tensor, perm: &[usize]) -> crate::Result<Tensor>;
    fn transpose_read(&mut self, input: TensorRead<'_>, perm: &[usize]) -> crate::Result<Tensor> {
        self.transpose(read_tensor("transpose", input)?, perm)
    }

    fn reshape(&mut self, input: &Tensor, shape: &[usize]) -> crate::Result<Tensor>;
    fn reshape_read(&mut self, input: TensorRead<'_>, shape: &[usize]) -> crate::Result<Tensor> {
        self.reshape(read_tensor("reshape", input)?, shape)
    }

    fn broadcast_in_dim(
        &mut self,
        input: &Tensor,
        shape: &[usize],
        dims: &[usize],
    ) -> crate::Result<Tensor>;
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
    fn convert(&mut self, input: &Tensor, to: crate::DType) -> crate::Result<Tensor> {
        validate_convert_dtype("convert", input.dtype(), to)?;
        self.cast(input, to)
    }

    fn extract_diagonal(
        &mut self,
        input: &Tensor,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<Tensor>;
    fn embed_diagonal(
        &mut self,
        input: &Tensor,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<Tensor>;
    fn tril(&mut self, input: &Tensor, k: i64) -> crate::Result<Tensor>;
    fn triu(&mut self, input: &Tensor, k: i64) -> crate::Result<Tensor>;
}

/// Reduction operations.
///
/// Reducing over an axis whose extent is zero returns an error for every
/// reduction operation. Passing an empty `axes` slice is a no-op and returns the
/// input values unchanged.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::TensorReduction;
///
/// fn accepts_reduction<B: TensorReduction>(_backend: &mut B) {}
/// ```
pub trait TensorReduction {
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
    fn reduce_sum_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        match input.as_tensor() {
            Some(input) => self.reduce_sum(input, axes),
            None => Err(crate::Error::backend_failure(
                "reduce_sum",
                "backend does not accept borrowed tensor views at this execution boundary",
            )),
        }
    }

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
    fn reduce_prod_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        match input.as_tensor() {
            Some(input) => self.reduce_prod(input, axes),
            None => Err(crate::Error::backend_failure(
                "reduce_prod",
                "backend does not accept borrowed tensor views at this execution boundary",
            )),
        }
    }

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
    fn reduce_max_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        match input.as_tensor() {
            Some(input) => self.reduce_max(input, axes),
            None => Err(crate::Error::backend_failure(
                "reduce_max",
                "backend does not accept borrowed tensor views at this execution boundary",
            )),
        }
    }

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
    fn reduce_min_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        match input.as_tensor() {
            Some(input) => self.reduce_min(input, axes),
            None => Err(crate::Error::backend_failure(
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
    fn gather(
        &mut self,
        operand: &Tensor,
        start_indices: &Tensor,
        config: &GatherConfig,
    ) -> crate::Result<Tensor>;
    fn scatter(
        &mut self,
        operand: &Tensor,
        scatter_indices: &Tensor,
        updates: &Tensor,
        config: &ScatterConfig,
    ) -> crate::Result<Tensor>;
    fn slice(&mut self, input: &Tensor, config: &SliceConfig) -> crate::Result<Tensor>;
    fn dynamic_slice(
        &mut self,
        input: &Tensor,
        starts: &Tensor,
        slice_sizes: &[usize],
    ) -> crate::Result<Tensor>;
    fn dynamic_update_slice(
        &mut self,
        operand: &Tensor,
        update: &Tensor,
        starts: &Tensor,
    ) -> crate::Result<Tensor>;
    fn pad(&mut self, input: &Tensor, config: &PadConfig) -> crate::Result<Tensor>;
    fn concatenate(&mut self, inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor>;
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
    fn to_contiguous(
        &mut self,
        view: &TypedTensorView<'_, T, R>,
    ) -> crate::Result<TypedTensor<T, R>>;

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
    fn download_to_host(&mut self, tensor: &Tensor) -> crate::Result<Tensor> {
        Ok(tensor.clone())
    }

    fn upload_host_tensor(&mut self, tensor: &Tensor) -> crate::Result<Tensor> {
        Ok(tensor.clone())
    }
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
        Self: TensorBackend + Sized,
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
        Self: TensorBackend + Sized,
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
pub trait BackendSession: TensorBackendOps + SessionCachedDot + TensorDeviceTransfer {}

impl<T> BackendSession for T where
    T: TensorBackendOps + SessionCachedDot + TensorDeviceTransfer + ?Sized
{
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
pub fn default_backend_session<B: TensorBackend, R: Send>(
    backend: &mut B,
    f: impl FnOnce(&mut dyn BackendSession) -> R + Send,
) -> R {
    f(backend)
}
