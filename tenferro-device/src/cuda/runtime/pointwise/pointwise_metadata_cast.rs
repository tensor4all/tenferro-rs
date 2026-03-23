use super::*;
use crate::{Error, Result};
use cudarc::driver::{LaunchConfig, PushKernelArg};

impl CudaRuntime {
    pub(crate) unsafe fn metadata_cast_bool_f32_raw(
        &self,
        input: *const u8,
        dst: *mut f32,
        spec: &MetadataCastSpec,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        self.metadata_cast_raw(
            METADATA_CAST_BOOL_F32_KERNEL_NAME,
            input,
            dst,
            spec,
            alpha,
            beta,
        )
    }

    pub(crate) unsafe fn metadata_cast_i32_f32_raw(
        &self,
        input: *const i32,
        dst: *mut f32,
        spec: &MetadataCastSpec,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        self.metadata_cast_raw(
            METADATA_CAST_I32_F32_KERNEL_NAME,
            input,
            dst,
            spec,
            alpha,
            beta,
        )
    }

    pub(crate) unsafe fn metadata_cast_bool_f64_raw(
        &self,
        input: *const u8,
        dst: *mut f64,
        spec: &MetadataCastSpec,
        alpha: f64,
        beta: f64,
    ) -> Result<()> {
        self.metadata_cast_raw(
            METADATA_CAST_BOOL_F64_KERNEL_NAME,
            input,
            dst,
            spec,
            alpha,
            beta,
        )
    }

    pub(crate) unsafe fn metadata_cast_i32_f64_raw(
        &self,
        input: *const i32,
        dst: *mut f64,
        spec: &MetadataCastSpec,
        alpha: f64,
        beta: f64,
    ) -> Result<()> {
        self.metadata_cast_raw(
            METADATA_CAST_I32_F64_KERNEL_NAME,
            input,
            dst,
            spec,
            alpha,
            beta,
        )
    }

    unsafe fn metadata_cast_raw<Src, Dst>(
        &self,
        kernel_name: &str,
        input: *const Src,
        dst: *mut Dst,
        spec: &MetadataCastSpec,
        alpha: Dst,
        beta: Dst,
    ) -> Result<()>
    where
        Src: cudarc::driver::DeviceRepr,
        Dst: cudarc::driver::DeviceRepr + Copy,
    {
        let numel = checked_numel(&spec.dims)?;
        if numel == 0 {
            return Ok(());
        }

        let (kernel, stream) = load_metadata_scalar_kernel(self, kernel_name)?;
        let dims_dev = stream
            .clone_htod(&dims_to_i64(&spec.dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD metadata cast dims", err))?;
        let input_strides_dev = stream
            .clone_htod(&to_i64_vec(
                &spec.input_strides,
                "metadata cast input stride",
            )?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD metadata cast input strides", err))?;
        let dst_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.dst_strides, "metadata cast dst stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD metadata cast dst strides", err))?;
        let ndim = i32::try_from(spec.dims.len())
            .map_err(|_| Error::InvalidArgument("metadata cast rank exceeds i32 range".into()))?;
        let input_offset = i64::try_from(spec.input_offset).map_err(|_| {
            Error::InvalidArgument("metadata cast input offset exceeds i64 range".into())
        })?;
        let dst_offset = i64::try_from(spec.dst_offset).map_err(|_| {
            Error::InvalidArgument("metadata cast dst offset exceeds i64 range".into())
        })?;
        let numel_u64 = u64::try_from(numel)
            .map_err(|_| Error::InvalidArgument("metadata cast numel exceeds u64 range".into()))?;
        let numel_u32 = u32::try_from(numel).map_err(|_| {
            Error::InvalidArgument("metadata cast currently requires len <= u32::MAX".into())
        })?;
        let input_ptr = input as u64;
        let dst_ptr = dst as u64;
        let config = LaunchConfig {
            grid_dim: (numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&input_ptr)
                .arg(&dst_ptr)
                .arg(&dims_dev)
                .arg(&input_strides_dev)
                .arg(&input_offset)
                .arg(&dst_strides_dev)
                .arg(&dst_offset)
                .arg(&ndim)
                .arg(&numel_u64)
                .arg(&alpha)
                .arg(&beta)
                .launch(config)
                .map_err(|err| cuda_error("CUDA metadata cast kernel launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }
}
