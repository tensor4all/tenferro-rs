use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::*;
use crate::{Error, Generator, Result};

fn validate_rng_output_len(
    actual_len: usize,
    dims: &[usize],
    strides: &[isize],
    offset: isize,
    label: &str,
) -> Result<()> {
    let required = required_storage_len(dims, strides, offset, label)?;
    if actual_len < required {
        return Err(Error::InvalidArgument(format!(
            "{label} length mismatch: actual={actual_len} required={required}"
        )));
    }
    Ok(())
}

impl CudaRuntime {
    /// Fill a raw CUDA buffer with uniform `f64` samples in `[0, 1)`.
    ///
    /// # Safety
    ///
    /// `dst` must point to a live CUDA allocation on this runtime's device and
    /// be compatible with the provided layout metadata.
    pub unsafe fn rng_fill_uniform_f64_raw(
        &self,
        generator: &mut Generator,
        dst: *mut f64,
        dst_len: usize,
        dims: &[usize],
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        validate_rng_output_len(dst_len, dims, dst_strides, dst_offset, "rng uniform dst")?;
        let numel = checked_numel(dims)?;
        if numel == 0 {
            return Ok(());
        }
        let (seed, offset_counter) = generator.cuda_seed_and_offset(self.device_id())?;
        let (kernel, stream) = load_rng_kernel(self, RNG_UNIFORM_F64_KERNEL_NAME)?;
        let dims_dev = stream
            .clone_htod(&dims_to_i64(dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dims", err))?;
        let dst_strides_dev = stream
            .clone_htod(&to_i64_vec(dst_strides, "dst stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dst strides", err))?;
        let ndim = i32::try_from(dims.len())
            .map_err(|_| Error::InvalidArgument("rng uniform rank exceeds i32 range".into()))?;
        let dst_offset = i64::try_from(dst_offset).map_err(|_| {
            Error::InvalidArgument("rng uniform dst offset exceeds i64 range".into())
        })?;
        let numel_u64 = u64::try_from(numel)
            .map_err(|_| Error::InvalidArgument("rng uniform numel exceeds u64 range".into()))?;
        let numel_u32 = u32::try_from(numel)
            .map_err(|_| Error::InvalidArgument("rng uniform requires len <= u32::MAX".into()))?;
        let dst_ptr = dst as u64;
        let config = LaunchConfig {
            grid_dim: (numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&seed)
                .arg(&offset_counter)
                .arg(&dst_ptr)
                .arg(&dims_dev)
                .arg(&dst_strides_dev)
                .arg(&dst_offset)
                .arg(&ndim)
                .arg(&numel_u64)
                .launch(config)
                .map_err(|err| cuda_error("CUDA rng uniform kernel launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))?;
        generator.advance_cuda_offset(self.device_id(), numel_u64)
    }

    /// Fill a raw CUDA buffer with standard-normal `f64` samples.
    ///
    /// # Safety
    ///
    /// `dst` must point to a live CUDA allocation on this runtime's device and
    /// be compatible with the provided layout metadata.
    pub unsafe fn rng_fill_normal_f64_raw(
        &self,
        generator: &mut Generator,
        dst: *mut f64,
        dst_len: usize,
        dims: &[usize],
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        validate_rng_output_len(dst_len, dims, dst_strides, dst_offset, "rng normal dst")?;
        let numel = checked_numel(dims)?;
        if numel == 0 {
            return Ok(());
        }
        let (seed, offset_counter) = generator.cuda_seed_and_offset(self.device_id())?;
        let (kernel, stream) = load_rng_kernel(self, RNG_NORMAL_F64_KERNEL_NAME)?;
        let dims_dev = stream
            .clone_htod(&dims_to_i64(dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dims", err))?;
        let dst_strides_dev = stream
            .clone_htod(&to_i64_vec(dst_strides, "dst stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dst strides", err))?;
        let ndim = i32::try_from(dims.len())
            .map_err(|_| Error::InvalidArgument("rng normal rank exceeds i32 range".into()))?;
        let dst_offset = i64::try_from(dst_offset).map_err(|_| {
            Error::InvalidArgument("rng normal dst offset exceeds i64 range".into())
        })?;
        let numel_u64 = u64::try_from(numel)
            .map_err(|_| Error::InvalidArgument("rng normal numel exceeds u64 range".into()))?;
        let numel_u32 = u32::try_from(numel)
            .map_err(|_| Error::InvalidArgument("rng normal requires len <= u32::MAX".into()))?;
        let dst_ptr = dst as u64;
        let config = LaunchConfig {
            grid_dim: (numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&seed)
                .arg(&offset_counter)
                .arg(&dst_ptr)
                .arg(&dims_dev)
                .arg(&dst_strides_dev)
                .arg(&dst_offset)
                .arg(&ndim)
                .arg(&numel_u64)
                .launch(config)
                .map_err(|err| cuda_error("CUDA rng normal kernel launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))?;
        generator.advance_cuda_offset(self.device_id(), numel_u64)
    }

    /// Fill a raw CUDA buffer with integer `i32` samples in `[low, high)`.
    ///
    /// # Safety
    ///
    /// `dst` must point to a live CUDA allocation on this runtime's device and
    /// be compatible with the provided layout metadata.
    pub unsafe fn rng_fill_i32_raw(
        &self,
        generator: &mut Generator,
        low: i32,
        high: i32,
        dst: *mut i32,
        dst_len: usize,
        dims: &[usize],
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        if low >= high {
            return Err(Error::InvalidArgument(format!(
                "rng integer requires low < high (got low={low}, high={high})"
            )));
        }
        validate_rng_output_len(dst_len, dims, dst_strides, dst_offset, "rng integer dst")?;
        let numel = checked_numel(dims)?;
        if numel == 0 {
            return Ok(());
        }
        let (seed, offset_counter) = generator.cuda_seed_and_offset(self.device_id())?;
        let (kernel, stream) = load_rng_kernel(self, RNG_INT_I32_KERNEL_NAME)?;
        let dims_dev = stream
            .clone_htod(&dims_to_i64(dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dims", err))?;
        let dst_strides_dev = stream
            .clone_htod(&to_i64_vec(dst_strides, "dst stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dst strides", err))?;
        let ndim = i32::try_from(dims.len())
            .map_err(|_| Error::InvalidArgument("rng integer rank exceeds i32 range".into()))?;
        let dst_offset = i64::try_from(dst_offset).map_err(|_| {
            Error::InvalidArgument("rng integer dst offset exceeds i64 range".into())
        })?;
        let numel_u64 = u64::try_from(numel)
            .map_err(|_| Error::InvalidArgument("rng integer numel exceeds u64 range".into()))?;
        let numel_u32 = u32::try_from(numel)
            .map_err(|_| Error::InvalidArgument("rng integer requires len <= u32::MAX".into()))?;
        let dst_ptr = dst as u64;
        let config = LaunchConfig {
            grid_dim: (numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&seed)
                .arg(&offset_counter)
                .arg(&low)
                .arg(&high)
                .arg(&dst_ptr)
                .arg(&dims_dev)
                .arg(&dst_strides_dev)
                .arg(&dst_offset)
                .arg(&ndim)
                .arg(&numel_u64)
                .launch(config)
                .map_err(|err| cuda_error("CUDA rng integer kernel launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))?;
        generator.advance_cuda_offset(self.device_id(), numel_u64)
    }
}
