use super::*;

#[cfg(feature = "cpu-faer")]
pub(super) struct CpuFaerBinding {
    resources: CpuLinalgBinding,
}

#[cfg(feature = "cpu-faer")]
impl Clone for CpuFaerBinding {
    fn clone(&self) -> Self {
        Self {
            resources: self.resources.clone(),
        }
    }
}

#[cfg(feature = "cpu-faer")]
impl fmt::Debug for CpuFaerBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.resources.fmt(f)
    }
}

#[cfg(feature = "cpu-faer")]
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

#[cfg(feature = "cpu-faer")]
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

#[cfg(feature = "cpu-faer")]
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
    validate_non_aliasing(input_region, u_region, s_region, vt_region)
}

#[cfg(feature = "cpu-faer")]
fn validate_non_aliasing(
    input_region: Option<ByteRegion>,
    u_region: Option<ByteRegion>,
    s_region: Option<ByteRegion>,
    vt_region: Option<ByteRegion>,
) -> tenferro_tensor::Result<()> {
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

#[cfg(all(test, feature = "cpu-faer"))]
mod alias_tests {
    use super::*;

    #[test]
    fn numeric_regions_reject_all_six_alias_pairs_without_touching_sentinels() {
        let separate = [
            ByteRegion { start: 0, end: 16 },
            ByteRegion { start: 32, end: 48 },
            ByteRegion { start: 64, end: 72 },
            ByteRegion {
                start: 96,
                end: 104,
            },
        ];
        for (expected, lhs, rhs) in [
            ("input and U", 0, 1),
            ("input and S", 0, 2),
            ("input and Vt", 0, 3),
            ("U and S", 1, 2),
            ("U and Vt", 1, 3),
            ("S and Vt", 2, 3),
        ] {
            let mut regions = separate;
            regions[rhs] = regions[lhs];
            let sentinels = [11_u8, 22, 33];
            let error = validate_non_aliasing(
                Some(regions[0]),
                Some(regions[1]),
                Some(regions[2]),
                Some(regions[3]),
            )
            .unwrap_err();
            assert!(error.to_string().contains(expected));
            assert_eq!(sentinels, [11, 22, 33]);
        }
    }
}

#[cfg(feature = "cpu-faer")]
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

#[cfg(feature = "cpu-faer")]
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

#[cfg(feature = "cpu-faer")]
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

#[cfg(feature = "cpu-faer")]
#[derive(Clone, Copy)]
struct ByteRegion {
    start: usize,
    end: usize,
}

#[cfg(feature = "cpu-faer")]
fn regions_overlap(lhs: Option<ByteRegion>, rhs: Option<ByteRegion>) -> bool {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => lhs.start < rhs.end && rhs.start < lhs.end,
        _ => false,
    }
}

#[cfg(feature = "cpu-faer")]
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

#[cfg(feature = "cpu-faer")]
fn write_region(write: &TensorWrite<'_>) -> tenferro_tensor::Result<Option<ByteRegion>> {
    read_region(&write.as_read())
}

#[cfg(feature = "cpu-faer")]
enum TypedReadRef<'a, T> {
    Tensor(&'a TypedTensor<T>),
    View(&'a TypedTensorView<'a, T>),
}

#[cfg(feature = "cpu-faer")]
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

#[cfg(feature = "cpu-faer")]
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
