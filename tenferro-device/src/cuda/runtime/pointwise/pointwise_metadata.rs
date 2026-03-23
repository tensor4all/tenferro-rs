use super::*;
use crate::{Error, Result};
use cudarc::driver::{LaunchConfig, PushKernelArg};

impl CudaRuntime {
    pub(crate) fn metadata_generate(
        &self,
        op: MetadataGenerateOp,
        dst: MetadataTensorMut<'_>,
        spec: &MetadataGenerateSpec,
    ) -> Result<()> {
        match (op, dst) {
            (MetadataGenerateOp::IotaStartZero, MetadataTensorMut::I32(dst)) => {
                self.ensure_same_device(dst.device_id())?;
                let numel = checked_numel(&spec.dims)?;
                if numel > i32::MAX as usize {
                    return Err(Error::InvalidArgument(
                        "metadata iota currently requires len <= i32::MAX".into(),
                    ));
                }
                let required_len = required_storage_len(
                    &spec.dims,
                    spec.dst_strides(),
                    spec.dst_offset(),
                    "metadata iota destination",
                )?;
                if dst.len() < required_len {
                    return Err(Error::InvalidArgument(format!(
                        "metadata iota length mismatch: dst={} required={}",
                        dst.len(),
                        required_len
                    )));
                }
                unsafe { self.metadata_generate_iota_i32_raw(dst.device_ptr(), spec) }
            }
            (
                MetadataGenerateOp::Constant(MetadataConstantValue::I32(value)),
                MetadataTensorMut::I32(dst),
            ) => {
                self.ensure_same_device(dst.device_id())?;
                let required_len = required_storage_len(
                    &spec.dims,
                    spec.dst_strides(),
                    spec.dst_offset(),
                    "metadata constant destination",
                )?;
                if dst.len() < required_len {
                    return Err(Error::InvalidArgument(format!(
                        "metadata constant length mismatch: dst={} required={}",
                        dst.len(),
                        required_len
                    )));
                }
                unsafe { self.metadata_generate_constant_i32_raw(dst.device_ptr(), value, spec) }
            }
            (
                MetadataGenerateOp::Constant(MetadataConstantValue::Bool(value)),
                MetadataTensorMut::Bool(dst),
            ) => {
                self.ensure_same_device(dst.device_id())?;
                let required_len = required_storage_len(
                    &spec.dims,
                    spec.dst_strides(),
                    spec.dst_offset(),
                    "metadata constant destination",
                )?;
                if dst.len() < required_len {
                    return Err(Error::InvalidArgument(format!(
                        "metadata constant length mismatch: dst={} required={}",
                        dst.len(),
                        required_len
                    )));
                }
                unsafe {
                    self.metadata_generate_constant_bool_raw(dst.device_ptr(), value as u8, spec)
                }
            }
            (MetadataGenerateOp::IotaStartZero, MetadataTensorMut::Bool(_)) => Err(
                Error::InvalidArgument("metadata iota currently supports i32 output only".into()),
            ),
            (
                MetadataGenerateOp::Constant(MetadataConstantValue::I32(_)),
                MetadataTensorMut::Bool(_),
            )
            | (
                MetadataGenerateOp::Constant(MetadataConstantValue::Bool(_)),
                MetadataTensorMut::I32(_),
            ) => Err(Error::InvalidArgument(
                "metadata constant output dtype does not match payload".into(),
            )),
        }
    }

    pub(crate) fn metadata_binary(
        &self,
        op: MetadataBinaryOp,
        lhs: MetadataTensorRef<'_>,
        rhs: MetadataTensorRef<'_>,
        dst: MetadataTensorMut<'_>,
        spec: &MetadataBinarySpec,
    ) -> Result<()> {
        match (lhs, rhs, dst) {
            (
                MetadataTensorRef::I32(lhs),
                MetadataTensorRef::I32(rhs),
                MetadataTensorMut::Bool(dst),
            ) => {
                self.ensure_same_device(lhs.device_id())?;
                self.ensure_same_device(rhs.device_id())?;
                self.ensure_same_device(dst.device_id())?;
                let lhs_required = required_storage_len(
                    &spec.dims,
                    &spec.lhs_strides,
                    spec.lhs_offset,
                    "metadata binary lhs",
                )?;
                let rhs_required = required_storage_len(
                    &spec.dims,
                    &spec.rhs_strides,
                    spec.rhs_offset,
                    "metadata binary rhs",
                )?;
                let dst_required = required_storage_len(
                    &spec.dims,
                    &spec.dst_strides,
                    spec.dst_offset,
                    "metadata binary dst",
                )?;
                if lhs.len() < lhs_required || rhs.len() < rhs_required || dst.len() < dst_required
                {
                    return Err(Error::InvalidArgument(format!(
                        "metadata binary storage mismatch: lhs={} rhs={} dst={} required_lhs={} required_rhs={} required_dst={}",
                        lhs.len(),
                        rhs.len(),
                        dst.len(),
                        lhs_required,
                        rhs_required,
                        dst_required
                    )));
                }
                unsafe {
                    self.metadata_binary_i32_bool_raw(
                        metadata_binary_opcode(op),
                        lhs.device_ptr().cast_const(),
                        rhs.device_ptr().cast_const(),
                        dst.device_ptr(),
                        spec,
                    )
                }
            }
            (
                MetadataTensorRef::I32(lhs),
                MetadataTensorRef::I32(rhs),
                MetadataTensorMut::I32(dst),
            ) => {
                self.ensure_same_device(lhs.device_id())?;
                self.ensure_same_device(rhs.device_id())?;
                self.ensure_same_device(dst.device_id())?;
                let lhs_required = required_storage_len(
                    &spec.dims,
                    &spec.lhs_strides,
                    spec.lhs_offset,
                    "metadata binary lhs",
                )?;
                let rhs_required = required_storage_len(
                    &spec.dims,
                    &spec.rhs_strides,
                    spec.rhs_offset,
                    "metadata binary rhs",
                )?;
                let dst_required = required_storage_len(
                    &spec.dims,
                    &spec.dst_strides,
                    spec.dst_offset,
                    "metadata binary dst",
                )?;
                if lhs.len() < lhs_required || rhs.len() < rhs_required || dst.len() < dst_required
                {
                    return Err(Error::InvalidArgument(format!(
                        "metadata binary storage mismatch: lhs={} rhs={} dst={} required_lhs={} required_rhs={} required_dst={}",
                        lhs.len(),
                        rhs.len(),
                        dst.len(),
                        lhs_required,
                        rhs_required,
                        dst_required
                    )));
                }
                unsafe {
                    self.metadata_binary_i32_i32_raw(
                        metadata_binary_opcode(op),
                        lhs.device_ptr().cast_const(),
                        rhs.device_ptr().cast_const(),
                        dst.device_ptr(),
                        spec,
                    )
                }
            }
            (
                MetadataTensorRef::Bool(lhs),
                MetadataTensorRef::Bool(rhs),
                MetadataTensorMut::Bool(dst),
            ) => {
                self.ensure_same_device(lhs.device_id())?;
                self.ensure_same_device(rhs.device_id())?;
                self.ensure_same_device(dst.device_id())?;
                let lhs_required = required_storage_len(
                    &spec.dims,
                    &spec.lhs_strides,
                    spec.lhs_offset,
                    "metadata binary lhs",
                )?;
                let rhs_required = required_storage_len(
                    &spec.dims,
                    &spec.rhs_strides,
                    spec.rhs_offset,
                    "metadata binary rhs",
                )?;
                let dst_required = required_storage_len(
                    &spec.dims,
                    &spec.dst_strides,
                    spec.dst_offset,
                    "metadata binary dst",
                )?;
                if lhs.len() < lhs_required || rhs.len() < rhs_required || dst.len() < dst_required
                {
                    return Err(Error::InvalidArgument(format!(
                        "metadata binary storage mismatch: lhs={} rhs={} dst={} required_lhs={} required_rhs={} required_dst={}",
                        lhs.len(),
                        rhs.len(),
                        dst.len(),
                        lhs_required,
                        rhs_required,
                        dst_required
                    )));
                }
                unsafe {
                    self.metadata_binary_bool_bool_raw(
                        metadata_binary_opcode(op),
                        lhs.device_ptr().cast_const(),
                        rhs.device_ptr().cast_const(),
                        dst.device_ptr(),
                        spec,
                    )
                }
            }
            (lhs, rhs, dst) => Err(Error::InvalidArgument(format!(
                "unsupported metadata binary dtype combination: lhs={:?} rhs={:?} dst={:?}",
                lhs.dtype(),
                rhs.dtype(),
                dst.dtype()
            ))),
        }
    }

    pub(crate) fn metadata_ternary(
        &self,
        op: MetadataTernaryOp,
        cond: MetadataTensorRef<'_>,
        on_true: MetadataTensorRef<'_>,
        on_false: MetadataTensorRef<'_>,
        dst: MetadataTensorMut<'_>,
        spec: &MetadataTernarySpec,
    ) -> Result<()> {
        match (op, cond, on_true, on_false, dst) {
            (
                MetadataTernaryOp::Where,
                MetadataTensorRef::Bool(cond),
                MetadataTensorRef::I32(on_true),
                MetadataTensorRef::I32(on_false),
                MetadataTensorMut::I32(dst),
            ) => {
                self.ensure_same_device(cond.device_id())?;
                self.ensure_same_device(on_true.device_id())?;
                self.ensure_same_device(on_false.device_id())?;
                self.ensure_same_device(dst.device_id())?;
                let cond_required = required_storage_len(
                    &spec.dims,
                    &spec.cond_strides,
                    spec.cond_offset,
                    "metadata ternary cond",
                )?;
                let true_required = required_storage_len(
                    &spec.dims,
                    &spec.true_strides,
                    spec.true_offset,
                    "metadata ternary true",
                )?;
                let false_required = required_storage_len(
                    &spec.dims,
                    &spec.false_strides,
                    spec.false_offset,
                    "metadata ternary false",
                )?;
                let dst_required = required_storage_len(
                    &spec.dims,
                    &spec.dst_strides,
                    spec.dst_offset,
                    "metadata ternary dst",
                )?;
                if cond.len() < cond_required
                    || on_true.len() < true_required
                    || on_false.len() < false_required
                    || dst.len() < dst_required
                {
                    return Err(Error::InvalidArgument(format!(
                        "metadata ternary storage mismatch: cond={} true={} false={} dst={} required_cond={} required_true={} required_false={} required_dst={}",
                        cond.len(),
                        on_true.len(),
                        on_false.len(),
                        dst.len(),
                        cond_required,
                        true_required,
                        false_required,
                        dst_required
                    )));
                }
                unsafe {
                self.metadata_where_i32_raw(
                    cond.device_ptr().cast_const(),
                    on_true.device_ptr().cast_const(),
                    on_false.device_ptr().cast_const(),
                    dst.device_ptr(),
                    spec,
                )
                }
            }
            (
                MetadataTernaryOp::Where,
                MetadataTensorRef::Bool(cond),
                MetadataTensorRef::Bool(on_true),
                MetadataTensorRef::Bool(on_false),
                MetadataTensorMut::Bool(dst),
            ) => {
                self.ensure_same_device(cond.device_id())?;
                self.ensure_same_device(on_true.device_id())?;
                self.ensure_same_device(on_false.device_id())?;
                self.ensure_same_device(dst.device_id())?;
                let cond_required = required_storage_len(
                    &spec.dims,
                    &spec.cond_strides,
                    spec.cond_offset,
                    "metadata ternary cond",
                )?;
                let true_required = required_storage_len(
                    &spec.dims,
                    &spec.true_strides,
                    spec.true_offset,
                    "metadata ternary true",
                )?;
                let false_required = required_storage_len(
                    &spec.dims,
                    &spec.false_strides,
                    spec.false_offset,
                    "metadata ternary false",
                )?;
                let dst_required = required_storage_len(
                    &spec.dims,
                    &spec.dst_strides,
                    spec.dst_offset,
                    "metadata ternary dst",
                )?;
                if cond.len() < cond_required
                    || on_true.len() < true_required
                    || on_false.len() < false_required
                    || dst.len() < dst_required
                {
                    return Err(Error::InvalidArgument(format!(
                        "metadata ternary storage mismatch: cond={} true={} false={} dst={} required_cond={} required_true={} required_false={} required_dst={}",
                        cond.len(),
                        on_true.len(),
                        on_false.len(),
                        dst.len(),
                        cond_required,
                        true_required,
                        false_required,
                        dst_required
                    )));
                }
                unsafe {
                self.metadata_where_bool_raw(
                    cond.device_ptr().cast_const(),
                    on_true.device_ptr().cast_const(),
                    on_false.device_ptr().cast_const(),
                    dst.device_ptr(),
                    spec,
                )
                }
            }
            (op, cond, on_true, on_false, dst) => Err(Error::InvalidArgument(format!(
                "unsupported metadata ternary combination for {op:?}: cond={:?} true={:?} false={:?} dst={:?}",
                cond.dtype(),
                on_true.dtype(),
                on_false.dtype(),
                dst.dtype()
            ))),
        }
    }

    pub(crate) fn metadata_reduction(
        &self,
        op: MetadataReductionOp,
        input: MetadataTensorRef<'_>,
        dst: MetadataTensorMut<'_>,
        spec: &MetadataReductionSpec,
    ) -> Result<()> {
        match (op, input, dst) {
            (
                MetadataReductionOp::Sum,
                MetadataTensorRef::I32(input),
                MetadataTensorMut::I32(dst),
            ) => {
                self.ensure_same_device(input.device_id())?;
                self.ensure_same_device(dst.device_id())?;
                let input_required = required_storage_len(
                    &spec.input_dims,
                    &spec.input_strides,
                    spec.input_offset,
                    "metadata reduction input",
                )?;
                let output_required = required_storage_len(
                    &spec.output_dims,
                    &spec.output_strides,
                    spec.output_offset,
                    "metadata reduction output",
                )?;
                if input.len() < input_required || dst.len() < output_required {
                    return Err(Error::InvalidArgument(format!(
                        "metadata reduction storage mismatch: input={} dst={} required_input={} required_dst={}",
                        input.len(),
                        dst.len(),
                        input_required,
                        output_required
                    )));
                }
                unsafe {
                    self.metadata_reduce_sum_i32_raw(
                        input.device_ptr().cast_const(),
                        dst.device_ptr(),
                        spec,
                    )
                }
            }
            (
                MetadataReductionOp::Sum,
                MetadataTensorRef::Bool(input),
                MetadataTensorMut::I32(dst),
            ) => {
                self.ensure_same_device(input.device_id())?;
                self.ensure_same_device(dst.device_id())?;
                let input_required = required_storage_len(
                    &spec.input_dims,
                    &spec.input_strides,
                    spec.input_offset,
                    "metadata reduction input",
                )?;
                let output_required = required_storage_len(
                    &spec.output_dims,
                    &spec.output_strides,
                    spec.output_offset,
                    "metadata reduction output",
                )?;
                if input.len() < input_required || dst.len() < output_required {
                    return Err(Error::InvalidArgument(format!(
                        "metadata reduction storage mismatch: input={} dst={} required_input={} required_dst={}",
                        input.len(),
                        dst.len(),
                        input_required,
                        output_required
                    )));
                }
                unsafe {
                    self.metadata_reduce_sum_bool_raw(
                        input.device_ptr().cast_const(),
                        dst.device_ptr(),
                        spec,
                    )
                }
            }
            (
                MetadataReductionOp::All,
                MetadataTensorRef::Bool(input),
                MetadataTensorMut::Bool(dst),
            ) => {
                self.ensure_same_device(input.device_id())?;
                self.ensure_same_device(dst.device_id())?;
                let input_required = required_storage_len(
                    &spec.input_dims,
                    &spec.input_strides,
                    spec.input_offset,
                    "metadata reduction input",
                )?;
                let output_required = required_storage_len(
                    &spec.output_dims,
                    &spec.output_strides,
                    spec.output_offset,
                    "metadata reduction output",
                )?;
                if input.len() < input_required || dst.len() < output_required {
                    return Err(Error::InvalidArgument(format!(
                        "metadata reduction storage mismatch: input={} dst={} required_input={} required_dst={}",
                        input.len(),
                        dst.len(),
                        input_required,
                        output_required
                    )));
                }
                unsafe {
                    self.metadata_reduce_all_bool_raw(
                        input.device_ptr().cast_const(),
                        dst.device_ptr(),
                        spec,
                    )
                }
            }
            (
                MetadataReductionOp::Any,
                MetadataTensorRef::Bool(input),
                MetadataTensorMut::Bool(dst),
            ) => {
                self.ensure_same_device(input.device_id())?;
                self.ensure_same_device(dst.device_id())?;
                let input_required = required_storage_len(
                    &spec.input_dims,
                    &spec.input_strides,
                    spec.input_offset,
                    "metadata reduction input",
                )?;
                let output_required = required_storage_len(
                    &spec.output_dims,
                    &spec.output_strides,
                    spec.output_offset,
                    "metadata reduction output",
                )?;
                if input.len() < input_required || dst.len() < output_required {
                    return Err(Error::InvalidArgument(format!(
                        "metadata reduction storage mismatch: input={} dst={} required_input={} required_dst={}",
                        input.len(),
                        dst.len(),
                        input_required,
                        output_required
                    )));
                }
                unsafe {
                    self.metadata_reduce_any_bool_raw(
                        input.device_ptr().cast_const(),
                        dst.device_ptr(),
                        spec,
                    )
                }
            }
            (op, input, dst) => Err(Error::InvalidArgument(format!(
                "unsupported metadata reduction combination for {op:?}: input={:?} dst={:?}",
                input.dtype(),
                dst.dtype()
            ))),
        }
    }

    pub(crate) unsafe fn metadata_generate_iota_i32_raw(
        &self,
        dst: *mut i32,
        spec: &MetadataGenerateSpec,
    ) -> Result<()> {
        let numel = checked_numel(spec.dims())?;
        if numel == 0 {
            return Ok(());
        }

        let (kernel, stream) =
            load_metadata_scalar_kernel(self, METADATA_GENERATE_IOTA_I32_KERNEL_NAME)?;
        let dims_dev = stream
            .clone_htod(&dims_to_i64(spec.dims())?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD metadata dims", err))?;
        let dst_strides_dev = stream
            .clone_htod(&to_i64_vec(spec.dst_strides(), "metadata dst stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD metadata dst strides", err))?;
        let ndim = i32::try_from(spec.dims().len())
            .map_err(|_| Error::InvalidArgument("metadata rank exceeds i32 range".into()))?;
        let dst_offset = i64::try_from(spec.dst_offset())
            .map_err(|_| Error::InvalidArgument("metadata dst offset exceeds i64 range".into()))?;
        let numel_u64 = u64::try_from(numel)
            .map_err(|_| Error::InvalidArgument("metadata numel exceeds u64 range".into()))?;
        let numel_u32 = u32::try_from(numel).map_err(|_| {
            Error::InvalidArgument("metadata generate currently requires len <= i32::MAX".into())
        })?;
        let dst_ptr = dst as u64;
        let config = LaunchConfig {
            grid_dim: (numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&dst_ptr)
                .arg(&dims_dev)
                .arg(&dst_strides_dev)
                .arg(&dst_offset)
                .arg(&ndim)
                .arg(&numel_u64)
                .launch(config)
                .map_err(|err| cuda_error("CUDA metadata iota kernel launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    fn metadata_generate_constant_raw_impl<T: cudarc::driver::DeviceRepr + Copy>(
        &self,
        kernel_name: &'static str,
        dst: *mut T,
        value: T,
        spec: &MetadataGenerateSpec,
    ) -> Result<()> {
        let numel = checked_numel(spec.dims())?;
        if numel == 0 {
            return Ok(());
        }

        let (kernel, stream) = load_metadata_scalar_kernel(self, kernel_name)?;
        let dims_dev = stream
            .clone_htod(&dims_to_i64(spec.dims())?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD metadata dims", err))?;
        let dst_strides_dev = stream
            .clone_htod(&to_i64_vec(spec.dst_strides(), "metadata dst stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD metadata dst strides", err))?;
        let ndim = i32::try_from(spec.dims().len())
            .map_err(|_| Error::InvalidArgument("metadata rank exceeds i32 range".into()))?;
        let dst_offset = i64::try_from(spec.dst_offset())
            .map_err(|_| Error::InvalidArgument("metadata dst offset exceeds i64 range".into()))?;
        let numel_u64 = u64::try_from(numel)
            .map_err(|_| Error::InvalidArgument("metadata numel exceeds u64 range".into()))?;
        let numel_u32 = u32::try_from(numel).map_err(|_| {
            Error::InvalidArgument("metadata generate currently requires len <= i32::MAX".into())
        })?;
        let dst_ptr = dst as u64;
        let config = LaunchConfig {
            grid_dim: (numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&dst_ptr)
                .arg(&dims_dev)
                .arg(&dst_strides_dev)
                .arg(&dst_offset)
                .arg(&ndim)
                .arg(&numel_u64)
                .arg(&value)
                .launch(config)
                .map_err(|err| cuda_error("CUDA metadata constant kernel launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    pub(crate) unsafe fn metadata_generate_constant_i32_raw(
        &self,
        dst: *mut i32,
        value: i32,
        spec: &MetadataGenerateSpec,
    ) -> Result<()> {
        self.metadata_generate_constant_raw_impl(
            METADATA_GENERATE_CONSTANT_I32_KERNEL_NAME,
            dst,
            value,
            spec,
        )
    }

    pub(crate) unsafe fn metadata_generate_constant_bool_raw(
        &self,
        dst: *mut u8,
        value: u8,
        spec: &MetadataGenerateSpec,
    ) -> Result<()> {
        self.metadata_generate_constant_raw_impl(
            METADATA_GENERATE_CONSTANT_BOOL_KERNEL_NAME,
            dst,
            value,
            spec,
        )
    }

    pub(crate) unsafe fn metadata_binary_i32_bool_raw(
        &self,
        op_code: i32,
        lhs: *const i32,
        rhs: *const i32,
        dst: *mut u8,
        spec: &MetadataBinarySpec,
    ) -> Result<()> {
        self.metadata_binary_compare_raw(
            METADATA_BINARY_I32_BOOL_KERNEL_NAME,
            op_code,
            lhs,
            rhs,
            dst,
            spec,
        )
    }

    pub(crate) unsafe fn metadata_binary_i32_i32_raw(
        &self,
        op_code: i32,
        lhs: *const i32,
        rhs: *const i32,
        dst: *mut i32,
        spec: &MetadataBinarySpec,
    ) -> Result<()> {
        self.metadata_binary_arithmetic_raw(
            METADATA_BINARY_I32_I32_KERNEL_NAME,
            op_code,
            lhs,
            rhs,
            dst,
            spec,
        )
    }

    pub(crate) unsafe fn metadata_binary_bool_bool_raw(
        &self,
        op_code: i32,
        lhs: *const u8,
        rhs: *const u8,
        dst: *mut u8,
        spec: &MetadataBinarySpec,
    ) -> Result<()> {
        self.metadata_binary_compare_raw(
            METADATA_BINARY_BOOL_BOOL_KERNEL_NAME,
            op_code,
            lhs,
            rhs,
            dst,
            spec,
        )
    }

    unsafe fn metadata_binary_compare_raw<Lhs, Rhs>(
        &self,
        kernel_name: &str,
        op_code: i32,
        lhs: *const Lhs,
        rhs: *const Rhs,
        dst: *mut u8,
        spec: &MetadataBinarySpec,
    ) -> Result<()>
    where
        Lhs: cudarc::driver::DeviceRepr,
        Rhs: cudarc::driver::DeviceRepr,
    {
        let numel = checked_numel(&spec.dims)?;
        if numel == 0 {
            return Ok(());
        }

        let (kernel, stream) = load_metadata_scalar_kernel(self, kernel_name)?;
        let dims_dev = stream
            .clone_htod(&dims_to_i64(&spec.dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD metadata dims", err))?;
        let lhs_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.lhs_strides, "metadata lhs stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD metadata lhs strides", err))?;
        let rhs_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.rhs_strides, "metadata rhs stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD metadata rhs strides", err))?;
        let dst_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.dst_strides, "metadata dst stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD metadata dst strides", err))?;
        let ndim = i32::try_from(spec.dims.len())
            .map_err(|_| Error::InvalidArgument("metadata rank exceeds i32 range".into()))?;
        let lhs_offset = i64::try_from(spec.lhs_offset)
            .map_err(|_| Error::InvalidArgument("metadata lhs offset exceeds i64 range".into()))?;
        let rhs_offset = i64::try_from(spec.rhs_offset)
            .map_err(|_| Error::InvalidArgument("metadata rhs offset exceeds i64 range".into()))?;
        let dst_offset = i64::try_from(spec.dst_offset)
            .map_err(|_| Error::InvalidArgument("metadata dst offset exceeds i64 range".into()))?;
        let numel_u64 = u64::try_from(numel)
            .map_err(|_| Error::InvalidArgument("metadata numel exceeds u64 range".into()))?;
        let numel_u32 = u32::try_from(numel).map_err(|_| {
            Error::InvalidArgument("metadata binary currently requires len <= u32::MAX".into())
        })?;
        let lhs_ptr = lhs as u64;
        let rhs_ptr = rhs as u64;
        let dst_ptr = dst as u64;
        let config = LaunchConfig {
            grid_dim: (numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&lhs_ptr)
                .arg(&rhs_ptr)
                .arg(&dst_ptr)
                .arg(&dims_dev)
                .arg(&lhs_strides_dev)
                .arg(&lhs_offset)
                .arg(&rhs_strides_dev)
                .arg(&rhs_offset)
                .arg(&dst_strides_dev)
                .arg(&dst_offset)
                .arg(&ndim)
                .arg(&numel_u64)
                .arg(&op_code)
                .launch(config)
                .map_err(|err| cuda_error("CUDA metadata binary kernel launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    unsafe fn metadata_binary_arithmetic_raw<Lhs, Rhs>(
        &self,
        kernel_name: &str,
        op_code: i32,
        lhs: *const Lhs,
        rhs: *const Rhs,
        dst: *mut i32,
        spec: &MetadataBinarySpec,
    ) -> Result<()>
    where
        Lhs: cudarc::driver::DeviceRepr,
        Rhs: cudarc::driver::DeviceRepr,
    {
        let numel = checked_numel(&spec.dims)?;
        if numel == 0 {
            return Ok(());
        }

        let (kernel, stream) = load_metadata_scalar_kernel(self, kernel_name)?;
        let dims_dev = stream
            .clone_htod(&dims_to_i64(&spec.dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD metadata dims", err))?;
        let lhs_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.lhs_strides, "metadata lhs stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD metadata lhs strides", err))?;
        let rhs_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.rhs_strides, "metadata rhs stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD metadata rhs strides", err))?;
        let dst_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.dst_strides, "metadata dst stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD metadata dst strides", err))?;
        let ndim = i32::try_from(spec.dims.len())
            .map_err(|_| Error::InvalidArgument("metadata rank exceeds i32 range".into()))?;
        let lhs_offset = i64::try_from(spec.lhs_offset)
            .map_err(|_| Error::InvalidArgument("metadata lhs offset exceeds i64 range".into()))?;
        let rhs_offset = i64::try_from(spec.rhs_offset)
            .map_err(|_| Error::InvalidArgument("metadata rhs offset exceeds i64 range".into()))?;
        let dst_offset = i64::try_from(spec.dst_offset)
            .map_err(|_| Error::InvalidArgument("metadata dst offset exceeds i64 range".into()))?;
        let numel_u64 = u64::try_from(numel)
            .map_err(|_| Error::InvalidArgument("metadata numel exceeds u64 range".into()))?;
        let numel_u32 = u32::try_from(numel).map_err(|_| {
            Error::InvalidArgument("metadata binary currently requires len <= u32::MAX".into())
        })?;
        let lhs_ptr = lhs as u64;
        let rhs_ptr = rhs as u64;
        let dst_ptr = dst as u64;
        let config = LaunchConfig {
            grid_dim: (numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&lhs_ptr)
                .arg(&rhs_ptr)
                .arg(&dst_ptr)
                .arg(&dims_dev)
                .arg(&lhs_strides_dev)
                .arg(&lhs_offset)
                .arg(&rhs_strides_dev)
                .arg(&rhs_offset)
                .arg(&dst_strides_dev)
                .arg(&dst_offset)
                .arg(&ndim)
                .arg(&numel_u64)
                .arg(&op_code)
                .launch(config)
                .map_err(|err| cuda_error("CUDA metadata binary kernel launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    pub(crate) unsafe fn metadata_where_i32_raw(
        &self,
        cond: *const u8,
        on_true: *const i32,
        on_false: *const i32,
        dst: *mut i32,
        spec: &MetadataTernarySpec,
    ) -> Result<()> {
        self.metadata_where_raw(
            METADATA_TERNARY_I32_KERNEL_NAME,
            cond,
            on_true,
            on_false,
            dst,
            spec,
        )
    }

    pub(crate) unsafe fn metadata_where_bool_raw(
        &self,
        cond: *const u8,
        on_true: *const u8,
        on_false: *const u8,
        dst: *mut u8,
        spec: &MetadataTernarySpec,
    ) -> Result<()> {
        self.metadata_where_raw(
            METADATA_TERNARY_BOOL_KERNEL_NAME,
            cond,
            on_true,
            on_false,
            dst,
            spec,
        )
    }

    unsafe fn metadata_where_raw<InputT, OutputT>(
        &self,
        kernel_name: &str,
        cond: *const u8,
        on_true: *const InputT,
        on_false: *const InputT,
        dst: *mut OutputT,
        spec: &MetadataTernarySpec,
    ) -> Result<()>
    where
        InputT: cudarc::driver::DeviceRepr,
        OutputT: cudarc::driver::DeviceRepr,
    {
        let numel = checked_numel(&spec.dims)?;
        if numel == 0 {
            return Ok(());
        }

        let (kernel, stream) = load_metadata_scalar_kernel(self, kernel_name)?;
        let dims_dev = stream
            .clone_htod(&dims_to_i64(&spec.dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD metadata dims", err))?;
        let cond_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.cond_strides, "metadata cond stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD metadata cond strides", err))?;
        let true_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.true_strides, "metadata true stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD metadata true strides", err))?;
        let false_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.false_strides, "metadata false stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD metadata false strides", err))?;
        let dst_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.dst_strides, "metadata dst stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD metadata dst strides", err))?;
        let ndim = i32::try_from(spec.dims.len())
            .map_err(|_| Error::InvalidArgument("metadata rank exceeds i32 range".into()))?;
        let cond_offset = i64::try_from(spec.cond_offset)
            .map_err(|_| Error::InvalidArgument("metadata cond offset exceeds i64 range".into()))?;
        let true_offset = i64::try_from(spec.true_offset)
            .map_err(|_| Error::InvalidArgument("metadata true offset exceeds i64 range".into()))?;
        let false_offset = i64::try_from(spec.false_offset).map_err(|_| {
            Error::InvalidArgument("metadata false offset exceeds i64 range".into())
        })?;
        let dst_offset = i64::try_from(spec.dst_offset)
            .map_err(|_| Error::InvalidArgument("metadata dst offset exceeds i64 range".into()))?;
        let numel_u64 = u64::try_from(numel)
            .map_err(|_| Error::InvalidArgument("metadata numel exceeds u64 range".into()))?;
        let numel_u32 = u32::try_from(numel).map_err(|_| {
            Error::InvalidArgument("metadata where currently requires len <= u32::MAX".into())
        })?;
        let cond_ptr = cond as u64;
        let true_ptr = on_true as u64;
        let false_ptr = on_false as u64;
        let dst_ptr = dst as u64;
        let config = LaunchConfig {
            grid_dim: (numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&cond_ptr)
                .arg(&true_ptr)
                .arg(&false_ptr)
                .arg(&dst_ptr)
                .arg(&dims_dev)
                .arg(&cond_strides_dev)
                .arg(&cond_offset)
                .arg(&true_strides_dev)
                .arg(&true_offset)
                .arg(&false_strides_dev)
                .arg(&false_offset)
                .arg(&dst_strides_dev)
                .arg(&dst_offset)
                .arg(&ndim)
                .arg(&numel_u64)
                .launch(config)
                .map_err(|err| cuda_error("CUDA metadata where kernel launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    pub(crate) unsafe fn metadata_reduce_sum_i32_raw(
        &self,
        input: *const i32,
        dst: *mut i32,
        spec: &MetadataReductionSpec,
    ) -> Result<()> {
        self.metadata_reduce_raw(METADATA_REDUCE_SUM_I32_KERNEL_NAME, input, dst, spec)
    }

    pub(crate) unsafe fn metadata_reduce_sum_bool_raw(
        &self,
        input: *const u8,
        dst: *mut i32,
        spec: &MetadataReductionSpec,
    ) -> Result<()> {
        self.metadata_reduce_raw(METADATA_REDUCE_SUM_BOOL_KERNEL_NAME, input, dst, spec)
    }

    pub(crate) unsafe fn metadata_reduce_all_bool_raw(
        &self,
        input: *const u8,
        dst: *mut u8,
        spec: &MetadataReductionSpec,
    ) -> Result<()> {
        self.metadata_reduce_raw(METADATA_REDUCE_ALL_BOOL_KERNEL_NAME, input, dst, spec)
    }

    pub(crate) unsafe fn metadata_reduce_any_bool_raw(
        &self,
        input: *const u8,
        dst: *mut u8,
        spec: &MetadataReductionSpec,
    ) -> Result<()> {
        self.metadata_reduce_raw(METADATA_REDUCE_ANY_BOOL_KERNEL_NAME, input, dst, spec)
    }

    unsafe fn metadata_reduce_raw<InputT, OutputT>(
        &self,
        kernel_name: &str,
        input: *const InputT,
        dst: *mut OutputT,
        spec: &MetadataReductionSpec,
    ) -> Result<()>
    where
        InputT: cudarc::driver::DeviceRepr,
        OutputT: cudarc::driver::DeviceRepr,
    {
        let output_numel = checked_numel(&spec.output_dims)?;
        if output_numel == 0 {
            return Ok(());
        }
        let reduced_dims: Vec<usize> = spec
            .reduced_axes
            .iter()
            .map(|&axis| {
                spec.input_dims.get(axis).copied().ok_or_else(|| {
                    Error::InvalidArgument(format!("metadata reduction axis {axis} out of bounds"))
                })
            })
            .collect::<Result<_>>()?;
        let reduced_total = checked_numel(&reduced_dims)?;
        if (kernel_name == METADATA_REDUCE_SUM_I32_KERNEL_NAME
            || kernel_name == METADATA_REDUCE_SUM_BOOL_KERNEL_NAME)
            && reduced_total > i32::MAX as usize
        {
            return Err(Error::InvalidArgument(
                "metadata sum currently requires reduction volume <= i32::MAX".into(),
            ));
        }
        let (kernel, stream) = load_metadata_scalar_kernel(self, kernel_name)?;
        let input_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.input_strides, "metadata input stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD metadata input strides", err))?;
        let output_dims_dev = stream
            .clone_htod(&dims_to_i64(&spec.output_dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD metadata output dims", err))?;
        let output_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.output_strides, "metadata output stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD metadata output strides", err))?;
        let kept_axes_dev = stream
            .clone_htod(&axes_to_i32(&spec.kept_axes, "metadata kept")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD metadata kept axes", err))?;
        let reduced_axes_dev = stream
            .clone_htod(&axes_to_i32(&spec.reduced_axes, "metadata reduced")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD metadata reduced axes", err))?;
        let reduced_dims_dev = stream
            .clone_htod(&dims_to_i64(&reduced_dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD metadata reduced dims", err))?;
        let kept_rank = i32::try_from(spec.kept_axes.len()).map_err(|_| {
            Error::InvalidArgument("metadata reduction kept rank exceeds i32 range".into())
        })?;
        let reduced_rank = i32::try_from(spec.reduced_axes.len()).map_err(|_| {
            Error::InvalidArgument("metadata reduction reduced rank exceeds i32 range".into())
        })?;
        let input_offset = i64::try_from(spec.input_offset).map_err(|_| {
            Error::InvalidArgument("metadata input offset exceeds i64 range".into())
        })?;
        let output_offset = i64::try_from(spec.output_offset).map_err(|_| {
            Error::InvalidArgument("metadata output offset exceeds i64 range".into())
        })?;
        let output_numel_u64 = u64::try_from(output_numel).map_err(|_| {
            Error::InvalidArgument("metadata reduction output numel exceeds u64 range".into())
        })?;
        let output_numel_u32 = u32::try_from(output_numel).map_err(|_| {
            Error::InvalidArgument("metadata reduction currently requires len <= u32::MAX".into())
        })?;
        let reduced_total_u64 = u64::try_from(reduced_total).map_err(|_| {
            Error::InvalidArgument("metadata reduction total exceeds u64 range".into())
        })?;
        let input_ptr = input as u64;
        let dst_ptr = dst as u64;
        let config = LaunchConfig {
            grid_dim: (output_numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&input_ptr)
                .arg(&dst_ptr)
                .arg(&input_strides_dev)
                .arg(&input_offset)
                .arg(&output_dims_dev)
                .arg(&output_strides_dev)
                .arg(&output_offset)
                .arg(&kept_axes_dev)
                .arg(&kept_rank)
                .arg(&reduced_axes_dev)
                .arg(&reduced_dims_dev)
                .arg(&reduced_rank)
                .arg(&output_numel_u64)
                .arg(&reduced_total_u64)
                .launch(config)
                .map_err(|err| cuda_error("CUDA metadata reduction kernel launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }
}

fn metadata_binary_opcode(op: MetadataBinaryOp) -> i32 {
    match op {
        MetadataBinaryOp::Equal => 0,
        MetadataBinaryOp::NotEqual => 1,
        MetadataBinaryOp::Add => 2,
        MetadataBinaryOp::Sub => 3,
        MetadataBinaryOp::Mul => 4,
        MetadataBinaryOp::BitAnd => 2,
    }
}
