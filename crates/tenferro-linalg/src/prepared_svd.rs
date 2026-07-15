use std::fmt;
use std::mem::size_of;
use std::sync::Arc;

use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::{diag::DiagMut, MatMut, MatRef};
use num_complex::{Complex32, Complex64};
use tenferro_cpu::{CpuBackend, CpuBackendKind, CpuLinalgBinding};
use tenferro_tensor::{
    validate::checked_shape_product, BackendId, DType, Error, MemoryKind, Placement, Tensor,
    TensorRead, TensorView, TensorViewMut, TensorWrite, TypedTensor, TypedTensorView,
    TypedTensorViewMut,
};

use crate::{
    backend::LinalgBackend,
    extension::{
        canonicalize_svd_gauge_c32, canonicalize_svd_gauge_c64, canonicalize_svd_gauge_f32,
        canonicalize_svd_gauge_f64, validate_derivative_eps,
    },
    SvdGauge, SvdOptions,
};

const PREPARE_OP: &str = "prepare_svd";
const EXECUTE_OP: &str = "PreparedSvd::execute_into";
const PREPARED_CAPABILITY: &str = "prepared compact SVD";
const BINDING_CAPABILITY: &str = "prepared SVD backend/context binding";
const DESTINATION_CAPABILITY: &str = "compact column-major prepared SVD destination";

/// Exact metadata for one prepared SVD output.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::{CpuBackend, CpuBackendKind};
/// use tenferro_linalg::{PreparedSvdBackendExt, SvdOptions};
/// use tenferro_tensor::DType;
///
/// let mut backend = CpuBackend::with_kind(CpuBackendKind::Faer)?;
/// let plan = backend.prepare_svd([3, 2], DType::F64, SvdOptions::default())?;
/// assert_eq!(plan.output_specs().u().shape(), &[3, 2]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SvdOutputSpec {
    shape: Vec<usize>,
    dtype: DType,
    placement: Placement,
}

impl SvdOutputSpec {
    /// Return the exact output shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return the exact output dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Return the default host placement expected for a newly allocated output.
    ///
    /// Pinned host destinations are also accepted by execution.
    pub fn placement(&self) -> &Placement {
        &self.placement
    }
}

/// Output metadata for prepared compact SVD.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::{CpuBackend, CpuBackendKind};
/// use tenferro_linalg::{PreparedSvdBackendExt, SvdOptions};
/// use tenferro_tensor::DType;
///
/// let mut backend = CpuBackend::with_kind(CpuBackendKind::Faer)?;
/// let plan = backend.prepare_svd([2, 4], DType::C64, SvdOptions::default())?;
/// let specs = plan.output_specs();
/// assert_eq!(specs.s().dtype(), DType::F64);
/// assert_eq!(specs.vt().shape(), &[2, 4]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SvdOutputSpecs {
    u: SvdOutputSpec,
    s: SvdOutputSpec,
    vt: SvdOutputSpec,
}

impl SvdOutputSpecs {
    /// Return the left-singular-vector output specification.
    pub fn u(&self) -> &SvdOutputSpec {
        &self.u
    }

    /// Return the singular-value output specification.
    pub fn s(&self) -> &SvdOutputSpec {
        &self.s
    }

    /// Return the conjugate-transposed right-singular-vector output specification.
    pub fn vt(&self) -> &SvdOutputSpec {
        &self.vt
    }
}

/// Caller-provided writable outputs for compact SVD.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::SvdOutputWrites;
/// use tenferro_tensor::{Tensor, TensorWrite};
///
/// let mut u = Tensor::from_vec_col_major(vec![1, 1], vec![0.0_f64])?;
/// let mut s = Tensor::from_vec_col_major(vec![1], vec![0.0_f64])?;
/// let mut vt = Tensor::from_vec_col_major(vec![1, 1], vec![0.0_f64])?;
/// let writes = SvdOutputWrites::new(
///     TensorWrite::from_tensor(&mut u),
///     TensorWrite::from_tensor(&mut s),
///     TensorWrite::from_tensor(&mut vt),
/// );
/// assert_eq!(writes.u().shape(), &[1, 1]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Debug)]
pub struct SvdOutputWrites<'a> {
    pub(crate) u: TensorWrite<'a>,
    pub(crate) s: TensorWrite<'a>,
    pub(crate) vt: TensorWrite<'a>,
}

impl<'a> SvdOutputWrites<'a> {
    /// Bundle caller-owned `U`, `S`, and `Vt` destinations.
    pub fn new(u: TensorWrite<'a>, s: TensorWrite<'a>, vt: TensorWrite<'a>) -> Self {
        Self { u, s, vt }
    }

    /// Inspect the `U` destination without mutating it.
    pub fn u(&self) -> &TensorWrite<'a> {
        &self.u
    }

    /// Inspect the `S` destination without mutating it.
    pub fn s(&self) -> &TensorWrite<'a> {
        &self.s
    }

    /// Inspect the `Vt` destination without mutating it.
    pub fn vt(&self) -> &TensorWrite<'a> {
        &self.vt
    }
}

/// Immutable prepared compact-SVD plan.
///
/// A plan binds shape, dtype, provider, execution context, and options. Allocate
/// one workspace per concurrent execution lane.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::{CpuBackend, CpuBackendKind};
/// use tenferro_linalg::{PreparedSvdBackendExt, SvdOptions, SvdOutputWrites};
/// use tenferro_tensor::{DType, Tensor, TensorRead, TensorWrite};
///
/// let mut backend = CpuBackend::with_kind(CpuBackendKind::Faer)?;
/// let plan = backend.prepare_svd([2, 2], DType::F64, SvdOptions::default())?;
/// let mut workspace = plan.allocate_workspace(&mut backend)?;
/// let input = Tensor::from_vec_col_major(
///     vec![2, 2],
///     vec![3.0_f64, 0.0, 0.0, 2.0],
/// )?;
/// let mut u = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64; 4])?;
/// let mut s = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2])?;
/// let mut vt = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64; 4])?;
/// for _ in 0..2 {
///     plan.execute_into(
///         &mut backend,
///         &mut workspace,
///         TensorRead::from_tensor(&input),
///         SvdOutputWrites::new(
///             TensorWrite::from_tensor(&mut u),
///             TensorWrite::from_tensor(&mut s),
///             TensorWrite::from_tensor(&mut vt),
///         ),
///     )?;
/// }
/// assert_eq!(s.as_slice::<f64>()?, &[3.0, 2.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct PreparedSvd {
    shape: [usize; 2],
    dtype: DType,
    options: SvdOptions,
    specs: SvdOutputSpecs,
    binding: CpuFaerBinding,
    plan_token: Arc<()>,
    provider: FaerPlan,
}

impl fmt::Debug for PreparedSvd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedSvd")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("options", &self.options)
            .field("provider", &"faer")
            .field("scratch_bytes", &self.provider.scratch_bytes())
            .finish_non_exhaustive()
    }
}

impl PreparedSvd {
    /// Return exact `U`, `S`, and `Vt` output metadata.
    pub fn output_specs(&self) -> &SvdOutputSpecs {
        &self.specs
    }

    /// Allocate one opaque workspace bound to this plan and backend context.
    pub fn allocate_workspace<B: PreparedSvdBackendExt>(
        &self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<SvdWorkspace> {
        private::PreparedSvdDispatch::allocate_svd_workspace_impl(backend, self)
    }

    /// Execute into caller-owned outputs, reusing the supplied workspace.
    ///
    /// All metadata and overlap checks complete before the first output write.
    /// Once the provider call starts, a numerical provider failure may leave
    /// destinations partially overwritten.
    pub fn execute_into<B: PreparedSvdBackendExt>(
        &self,
        backend: &mut B,
        workspace: &mut SvdWorkspace,
        input: TensorRead<'_>,
        outputs: SvdOutputWrites<'_>,
    ) -> tenferro_tensor::Result<()> {
        private::PreparedSvdDispatch::execute_prepared_svd_into_impl(
            backend, self, workspace, input, outputs,
        )
    }
}

/// Opaque caller-owned mutable workspace for a [`PreparedSvd`].
///
/// The workspace is intentionally not `Clone`; pass it as `&mut` to enforce
/// exclusive use and allocate another workspace for another concurrency lane.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::{CpuBackend, CpuBackendKind};
/// use tenferro_linalg::{PreparedSvdBackendExt, SvdOptions};
/// use tenferro_tensor::DType;
///
/// let mut backend = CpuBackend::with_kind(CpuBackendKind::Faer)?;
/// let plan = backend.prepare_svd([2, 2], DType::F64, SvdOptions::default())?;
/// let workspace = plan.allocate_workspace(&mut backend)?;
/// assert!(format!("{workspace:?}").contains("retained_bytes"));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct SvdWorkspace {
    binding: CpuFaerBinding,
    shape: [usize; 2],
    dtype: DType,
    plan_token: Arc<()>,
    inner: FaerWorkspace,
}

impl fmt::Debug for SvdWorkspace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SvdWorkspace")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("provider", &"faer")
            .field("retained_bytes", &self.inner.retained_bytes())
            .finish_non_exhaustive()
    }
}

/// Backend capability for explicit prepared compact SVD execution.
///
/// The default hooks return an explicit unsupported error. Implementations must
/// not call an owned SVD or allocate replacement outputs as a fallback.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::{CpuBackend, CpuBackendKind};
/// use tenferro_linalg::{PreparedSvdBackendExt, SvdOptions};
/// use tenferro_tensor::DType;
///
/// let mut backend = CpuBackend::with_kind(CpuBackendKind::Faer)?;
/// let plan = backend.prepare_svd([3, 2], DType::F64, SvdOptions::default())?;
/// assert_eq!(plan.output_specs().s().shape(), &[2]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait PreparedSvdBackendExt: private::PreparedSvdDispatch {
    /// Prepare compact SVD for one fixed rank-2 shape and dtype.
    fn prepare_svd(
        &mut self,
        shape: [usize; 2],
        dtype: DType,
        options: SvdOptions,
    ) -> tenferro_tensor::Result<PreparedSvd> {
        private::PreparedSvdDispatch::prepare_svd_impl(self, shape, dtype, options)
    }
}

impl<T: private::PreparedSvdDispatch> PreparedSvdBackendExt for T {}

mod private {
    use super::*;

    pub trait PreparedSvdDispatch: LinalgBackend {
        fn prepare_svd_impl(
            &mut self,
            shape: [usize; 2],
            dtype: DType,
            options: SvdOptions,
        ) -> tenferro_tensor::Result<PreparedSvd>;

        fn allocate_svd_workspace_impl(
            &mut self,
            plan: &PreparedSvd,
        ) -> tenferro_tensor::Result<SvdWorkspace>;

        fn execute_prepared_svd_into_impl(
            &mut self,
            plan: &PreparedSvd,
            workspace: &mut SvdWorkspace,
            input: TensorRead<'_>,
            outputs: SvdOutputWrites<'_>,
        ) -> tenferro_tensor::Result<()>;
    }
}

struct CpuFaerBinding {
    resources: CpuLinalgBinding,
}

impl Clone for CpuFaerBinding {
    fn clone(&self) -> Self {
        Self {
            resources: self.resources.clone(),
        }
    }
}

impl fmt::Debug for CpuFaerBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.resources.fmt(f)
    }
}

impl CpuFaerBinding {
    fn capture(backend: &CpuBackend) -> Self {
        Self {
            resources: backend.linalg_binding(),
        }
    }

    fn validate(&self, backend: &CpuBackend, dtype: DType) -> tenferro_tensor::Result<()> {
        if !self.resources.matches(backend) || backend.kind() != CpuBackendKind::Faer {
            return Err(Error::unsupported_capability(
                EXECUTE_OP,
                BackendId::Cpu,
                cpu_provider_name(backend.kind()),
                dtype,
                BINDING_CAPABILITY,
            ));
        }
        Ok(())
    }
}

#[cfg(feature = "cpu-faer")]
impl private::PreparedSvdDispatch for CpuBackend {
    fn prepare_svd_impl(
        &mut self,
        shape: [usize; 2],
        dtype: DType,
        options: SvdOptions,
    ) -> tenferro_tensor::Result<PreparedSvd> {
        validate_derivative_eps(PREPARE_OP, options.derivative_eps)?;
        if self.kind() != CpuBackendKind::Faer {
            return Err(Error::unsupported_capability(
                PREPARE_OP,
                BackendId::Cpu,
                cpu_provider_name(self.kind()),
                dtype,
                PREPARED_CAPABILITY,
            ));
        }
        checked_shape_product(PREPARE_OP, "matrix shape", &shape)?;
        for &extent in &shape {
            isize::try_from(extent).map_err(|_| Error::InvalidConfig {
                op: PREPARE_OP,
                message: format!("matrix extent {extent} does not fit in isize"),
            })?;
        }
        let singular_dtype = real_dtype(dtype)?;
        let provider = FaerPlan::new(dtype, shape, self.linalg_context().faer_par())?;
        let k = shape[0].min(shape[1]);
        let placement = Placement {
            memory_kind: MemoryKind::UnpinnedHost,
            device: None,
        };
        let specs = SvdOutputSpecs {
            u: SvdOutputSpec {
                shape: vec![shape[0], k],
                dtype,
                placement: placement.clone(),
            },
            s: SvdOutputSpec {
                shape: vec![k],
                dtype: singular_dtype,
                placement: placement.clone(),
            },
            vt: SvdOutputSpec {
                shape: vec![k, shape[1]],
                dtype,
                placement,
            },
        };
        Ok(PreparedSvd {
            shape,
            dtype,
            options,
            specs,
            binding: CpuFaerBinding::capture(self),
            plan_token: Arc::new(()),
            provider,
        })
    }

    fn allocate_svd_workspace_impl(
        &mut self,
        plan: &PreparedSvd,
    ) -> tenferro_tensor::Result<SvdWorkspace> {
        plan.binding.validate(self, plan.dtype)?;
        let inner = plan.provider.allocate(plan.shape)?;
        Ok(SvdWorkspace {
            binding: plan.binding.clone(),
            shape: plan.shape,
            dtype: plan.dtype,
            plan_token: Arc::clone(&plan.plan_token),
            inner,
        })
    }

    fn execute_prepared_svd_into_impl(
        &mut self,
        plan: &PreparedSvd,
        workspace: &mut SvdWorkspace,
        input: TensorRead<'_>,
        outputs: SvdOutputWrites<'_>,
    ) -> tenferro_tensor::Result<()> {
        plan.binding.validate(self, plan.dtype)?;
        workspace.binding.validate(self, workspace.dtype)?;
        if !workspace.binding.resources.matches(self)
            || !plan.binding.resources.matches(self)
            || workspace.shape != plan.shape
            || workspace.dtype != plan.dtype
            || !Arc::ptr_eq(&workspace.plan_token, &plan.plan_token)
        {
            return Err(Error::unsupported_capability(
                EXECUTE_OP,
                BackendId::Cpu,
                cpu_provider_name(self.kind()),
                plan.dtype,
                BINDING_CAPABILITY,
            ));
        }
        validate_execution(plan, &input, &outputs)?;
        if plan.shape[0] == 0 || plan.shape[1] == 0 {
            return Ok(());
        }
        let par = self.linalg_context().faer_par();
        self.install(move || execute_faer(plan, &mut workspace.inner, par, input, outputs))
    }
}

#[cfg(not(feature = "cpu-faer"))]
impl private::PreparedSvdDispatch for CpuBackend {
    fn prepare_svd_impl(
        &mut self,
        _shape: [usize; 2],
        dtype: DType,
        _options: SvdOptions,
    ) -> tenferro_tensor::Result<PreparedSvd> {
        Err(Error::unsupported_capability(
            PREPARE_OP,
            BackendId::Cpu,
            cpu_provider_name(self.kind()),
            dtype,
            PREPARED_CAPABILITY,
        ))
    }

    fn allocate_svd_workspace_impl(
        &mut self,
        plan: &PreparedSvd,
    ) -> tenferro_tensor::Result<SvdWorkspace> {
        Err(Error::unsupported_capability(
            PREPARE_OP,
            BackendId::Cpu,
            cpu_provider_name(self.kind()),
            plan.dtype,
            PREPARED_CAPABILITY,
        ))
    }

    fn execute_prepared_svd_into_impl(
        &mut self,
        plan: &PreparedSvd,
        _workspace: &mut SvdWorkspace,
        _input: TensorRead<'_>,
        _outputs: SvdOutputWrites<'_>,
    ) -> tenferro_tensor::Result<()> {
        Err(Error::unsupported_capability(
            EXECUTE_OP,
            BackendId::Cpu,
            cpu_provider_name(self.kind()),
            plan.dtype,
            PREPARED_CAPABILITY,
        ))
    }
}

fn real_dtype(dtype: DType) -> tenferro_tensor::Result<DType> {
    match dtype {
        DType::F32 | DType::C32 => Ok(DType::F32),
        DType::F64 | DType::C64 => Ok(DType::F64),
        _ => Err(Error::unsupported_capability(
            PREPARE_OP,
            BackendId::Cpu,
            "faer",
            dtype,
            PREPARED_CAPABILITY,
        )),
    }
}

const fn cpu_provider_name(kind: CpuBackendKind) -> &'static str {
    match kind {
        CpuBackendKind::Faer => "faer",
        CpuBackendKind::Blas => "blas",
    }
}

fn validate_execution(
    plan: &PreparedSvd,
    input: &TensorRead<'_>,
    outputs: &SvdOutputWrites<'_>,
) -> tenferro_tensor::Result<()> {
    if input.shape() != plan.shape || input.dtype() != plan.dtype {
        return Err(Error::InvalidConfig {
            op: EXECUTE_OP,
            message: format!(
                "input must have shape {:?} and dtype {:?}, got {:?} and {:?}",
                plan.shape,
                plan.dtype,
                input.shape(),
                input.dtype()
            ),
        });
    }
    validate_host_read(input)?;
    validate_output(&outputs.u, &plan.specs.u, "U")?;
    validate_output(&outputs.s, &plan.specs.s, "S")?;
    validate_output(&outputs.vt, &plan.specs.vt, "Vt")?;

    let input_region = read_region(input)?;
    let u_region = write_region(&outputs.u)?;
    let s_region = write_region(&outputs.s)?;
    let vt_region = write_region(&outputs.vt)?;
    for (lhs_name, lhs, rhs_name, rhs) in [
        ("input", input_region, "U", u_region),
        ("input", input_region, "S", s_region),
        ("input", input_region, "Vt", vt_region),
        ("U", u_region, "S", s_region),
        ("U", u_region, "Vt", vt_region),
        ("S", s_region, "Vt", vt_region),
    ] {
        if regions_overlap(lhs, rhs) {
            return Err(Error::InvalidConfig {
                op: EXECUTE_OP,
                message: format!("{lhs_name} and {rhs_name} may overlap"),
            });
        }
    }
    Ok(())
}

fn validate_output(
    output: &TensorWrite<'_>,
    spec: &SvdOutputSpec,
    name: &str,
) -> tenferro_tensor::Result<()> {
    if output.shape() != spec.shape || output.dtype() != spec.dtype {
        return Err(Error::InvalidConfig {
            op: EXECUTE_OP,
            message: format!(
                "{name} must have shape {:?} and dtype {:?}, got {:?} and {:?}",
                spec.shape,
                spec.dtype,
                output.shape(),
                output.dtype()
            ),
        });
    }
    if !output.is_col_major_contiguous()? {
        let _ = name;
        return Err(Error::unsupported_capability(
            EXECUTE_OP,
            BackendId::Cpu,
            "faer",
            spec.dtype,
            DESTINATION_CAPABILITY,
        ));
    }
    validate_host_read(&output.as_read())
}

fn validate_host_read(read: &TensorRead<'_>) -> tenferro_tensor::Result<()> {
    let placement = read_placement(read);
    if placement.device.is_some()
        || !matches!(
            placement.memory_kind,
            MemoryKind::PinnedHost | MemoryKind::UnpinnedHost
        )
    {
        return Err(Error::unsupported_capability(
            EXECUTE_OP,
            BackendId::Cpu,
            "faer",
            read.dtype(),
            "host-resident prepared SVD storage",
        ));
    }
    Ok(())
}

fn read_placement<'a>(read: &'a TensorRead<'_>) -> &'a Placement {
    match read {
        TensorRead::Tensor(tensor) => tensor.placement(),
        TensorRead::View(view) => match view {
            TensorView::F32(v) => v.placement(),
            TensorView::F64(v) => v.placement(),
            TensorView::I32(v) => v.placement(),
            TensorView::I64(v) => v.placement(),
            TensorView::Bool(v) => v.placement(),
            TensorView::C32(v) => v.placement(),
            TensorView::C64(v) => v.placement(),
        },
    }
}

#[derive(Clone, Copy)]
struct ByteRegion {
    start: usize,
    end: usize,
}

fn regions_overlap(lhs: Option<ByteRegion>, rhs: Option<ByteRegion>) -> bool {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => lhs.start < rhs.end && rhs.start < lhs.end,
        _ => false,
    }
}

fn read_region(read: &TensorRead<'_>) -> tenferro_tensor::Result<Option<ByteRegion>> {
    macro_rules! region {
        ($value:expr, $ty:ty) => {
            typed_read_region::<$ty>($value)
        };
    }
    match read {
        TensorRead::Tensor(Tensor::F32(t)) => region!(TypedReadRef::Tensor(t), f32),
        TensorRead::Tensor(Tensor::F64(t)) => region!(TypedReadRef::Tensor(t), f64),
        TensorRead::Tensor(Tensor::I32(t)) => region!(TypedReadRef::Tensor(t), i32),
        TensorRead::Tensor(Tensor::I64(t)) => region!(TypedReadRef::Tensor(t), i64),
        TensorRead::Tensor(Tensor::Bool(t)) => region!(TypedReadRef::Tensor(t), bool),
        TensorRead::Tensor(Tensor::C32(t)) => region!(TypedReadRef::Tensor(t), Complex32),
        TensorRead::Tensor(Tensor::C64(t)) => region!(TypedReadRef::Tensor(t), Complex64),
        TensorRead::View(TensorView::F32(v)) => region!(TypedReadRef::View(v), f32),
        TensorRead::View(TensorView::F64(v)) => region!(TypedReadRef::View(v), f64),
        TensorRead::View(TensorView::I32(v)) => region!(TypedReadRef::View(v), i32),
        TensorRead::View(TensorView::I64(v)) => region!(TypedReadRef::View(v), i64),
        TensorRead::View(TensorView::Bool(v)) => region!(TypedReadRef::View(v), bool),
        TensorRead::View(TensorView::C32(v)) => region!(TypedReadRef::View(v), Complex32),
        TensorRead::View(TensorView::C64(v)) => region!(TypedReadRef::View(v), Complex64),
    }
}

fn write_region(write: &TensorWrite<'_>) -> tenferro_tensor::Result<Option<ByteRegion>> {
    read_region(&write.as_read())
}

enum TypedReadRef<'a, T> {
    Tensor(&'a TypedTensor<T>),
    View(&'a TypedTensorView<'a, T>),
}

fn typed_read_region<T: Clone + 'static>(
    read: TypedReadRef<'_, T>,
) -> tenferro_tensor::Result<Option<ByteRegion>> {
    let (base, shape, strides, offset) = match read {
        TypedReadRef::Tensor(tensor) => {
            let data = tensor.host_data()?;
            return byte_region(data.as_ptr(), 0, data.len(), size_of::<T>());
        }
        TypedReadRef::View(view) => (
            view.host_storage()?.as_ptr(),
            view.shape(),
            view.strides(),
            view.offset(),
        ),
    };
    if shape.contains(&0) {
        return Ok(None);
    }
    let mut min = offset;
    let mut max = offset;
    for (&extent, &stride) in shape.iter().zip(strides.iter()) {
        let span = isize::try_from(extent - 1)
            .ok()
            .and_then(|extent| extent.checked_mul(stride))
            .ok_or_else(|| Error::InvalidConfig {
                op: EXECUTE_OP,
                message: "input layout span overflows isize".to_owned(),
            })?;
        min = min
            .checked_add(span.min(0))
            .ok_or_else(|| Error::InvalidConfig {
                op: EXECUTE_OP,
                message: "input layout lower bound overflows isize".to_owned(),
            })?;
        max = max
            .checked_add(span.max(0))
            .ok_or_else(|| Error::InvalidConfig {
                op: EXECUTE_OP,
                message: "input layout upper bound overflows isize".to_owned(),
            })?;
    }
    let start = usize::try_from(min).map_err(|_| Error::InvalidConfig {
        op: EXECUTE_OP,
        message: "input layout lower bound is negative".to_owned(),
    })?;
    let span = max
        .checked_sub(min)
        .and_then(|span| span.checked_add(1))
        .ok_or_else(|| Error::InvalidConfig {
            op: EXECUTE_OP,
            message: "input layout range overflows isize".to_owned(),
        })?;
    let len = usize::try_from(span).map_err(|_| Error::InvalidConfig {
        op: EXECUTE_OP,
        message: "input layout range does not fit usize".to_owned(),
    })?;
    byte_region(base, start, len, size_of::<T>())
}

fn byte_region<T>(
    base: *const T,
    element_offset: usize,
    element_len: usize,
    element_size: usize,
) -> tenferro_tensor::Result<Option<ByteRegion>> {
    if element_len == 0 {
        return Ok(None);
    }
    let byte_offset =
        element_offset
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidConfig {
                op: EXECUTE_OP,
                message: "storage byte offset overflows usize".to_owned(),
            })?;
    let byte_len = element_len
        .checked_mul(element_size)
        .ok_or_else(|| Error::InvalidConfig {
            op: EXECUTE_OP,
            message: "storage byte length overflows usize".to_owned(),
        })?;
    let start = (base as usize)
        .checked_add(byte_offset)
        .ok_or_else(|| Error::InvalidConfig {
            op: EXECUTE_OP,
            message: "storage address range overflows usize".to_owned(),
        })?;
    let end = start
        .checked_add(byte_len)
        .ok_or_else(|| Error::InvalidConfig {
            op: EXECUTE_OP,
            message: "storage address range overflows usize".to_owned(),
        })?;
    Ok(Some(ByteRegion { start, end }))
}

mod faer_impl;
use faer_impl::{execute_faer, FaerPlan, FaerWorkspace};
