use std::{
    any::TypeId,
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use cudarc::{
    driver::{CudaFunction, CudaStream},
    nvrtc::{compile_ptx, Ptx},
};
use num_complex::{Complex32, Complex64};

use super::super::{shared::ContiguousOrder, state::CudaRuntime};
use crate::{Error, Result};

pub(crate) fn cuda_error(operation: &str, err: impl std::fmt::Debug) -> Error {
    Error::DeviceError(format!("{operation} failed: {err:?}"))
}

pub(crate) fn runtime_cache() -> &'static Mutex<HashMap<usize, Arc<CudaRuntime>>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<CudaRuntime>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn compile_ptx_once(
    cache: &'static OnceLock<std::result::Result<Ptx, String>>,
    src: &'static str,
    label: &str,
) -> Result<Ptx> {
    cache
        .get_or_init(|| {
            compile_ptx(src).map_err(|err| format!("NVRTC compile failed for {label}: {err:?}"))
        })
        .clone()
        .map_err(Error::DeviceError)
}

pub fn load_kernel_from_ptx(
    runtime: &CudaRuntime,
    ptx: Ptx,
    kernel_name: &str,
) -> Result<(CudaFunction, Arc<CudaStream>)> {
    runtime.bind_context()?;
    let ctx = runtime.context();
    let module = ctx
        .load_module(ptx)
        .map_err(|err| cuda_error("CUDA module load", err))?;
    let kernel = module
        .load_function(kernel_name)
        .map_err(|err| cuda_error("CUDA load function", err))?;
    Ok((kernel, ctx.default_stream()))
}

pub fn checked_num_bytes<T>(len: usize) -> Result<usize> {
    len.checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| Error::DeviceError("CUDA allocation size overflow".into()))
}

pub fn checked_numel(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| Error::InvalidArgument("strided copy numel overflow".into()))
    })
}

pub fn contiguous_strides(dims: &[usize], order: ContiguousOrder) -> Result<Vec<isize>> {
    let mut strides = vec![0isize; dims.len()];
    let mut stride = 1isize;

    match order {
        ContiguousOrder::ColumnMajor => {
            for (axis, &dim) in dims.iter().enumerate() {
                strides[axis] = stride;
                let dim = isize::try_from(dim).map_err(|_| {
                    Error::InvalidArgument(format!("dimension {dim} exceeds isize range"))
                })?;
                stride = stride
                    .checked_mul(dim)
                    .ok_or_else(|| Error::InvalidArgument("contiguous stride overflow".into()))?;
            }
        }
        ContiguousOrder::RowMajor => {
            for axis in (0..dims.len()).rev() {
                strides[axis] = stride;
                let dim = isize::try_from(dims[axis]).map_err(|_| {
                    Error::InvalidArgument(format!("dimension {} exceeds isize range", dims[axis]))
                })?;
                stride = stride
                    .checked_mul(dim)
                    .ok_or_else(|| Error::InvalidArgument("contiguous stride overflow".into()))?;
            }
        }
    }

    Ok(strides)
}

pub fn to_i64_vec(values: &[isize], label: &str) -> Result<Vec<i64>> {
    values
        .iter()
        .map(|&value| {
            i64::try_from(value).map_err(|_| {
                Error::InvalidArgument(format!("{label} value {value} exceeds i64 range"))
            })
        })
        .collect()
}

pub fn dims_to_i64(dims: &[usize]) -> Result<Vec<i64>> {
    dims.iter()
        .map(|&dim| {
            i64::try_from(dim)
                .map_err(|_| Error::InvalidArgument(format!("dimension {dim} exceeds i64 range")))
        })
        .collect()
}

pub fn axes_to_i32(axes: &[usize], label: &str) -> Result<Vec<i32>> {
    axes.iter()
        .map(|&axis| {
            i32::try_from(axis).map_err(|_| {
                Error::InvalidArgument(format!("{label} axis {axis} exceeds i32 range"))
            })
        })
        .collect()
}

pub fn supports_conj_strided_copy<T: 'static>() -> bool {
    TypeId::of::<T>() == TypeId::of::<Complex32>() || TypeId::of::<T>() == TypeId::of::<Complex64>()
}

pub fn validate_pointwise_rank(
    dims: &[usize],
    lhs_or_src_strides: &[isize],
    rhs_strides: Option<&[isize]>,
    dst_strides: &[isize],
) -> Result<()> {
    if dims.len() != lhs_or_src_strides.len() || dims.len() != dst_strides.len() {
        return Err(Error::InvalidArgument(format!(
            "pointwise rank mismatch: dims={} src={} dst={}",
            dims.len(),
            lhs_or_src_strides.len(),
            dst_strides.len()
        )));
    }
    if let Some(rhs_strides) = rhs_strides {
        if dims.len() != rhs_strides.len() {
            return Err(Error::InvalidArgument(format!(
                "pointwise rank mismatch: dims={} rhs={}",
                dims.len(),
                rhs_strides.len()
            )));
        }
    }
    Ok(())
}

pub fn validate_ternary_pointwise_rank(
    dims: &[usize],
    cond_strides: &[isize],
    true_strides: &[isize],
    false_strides: &[isize],
    dst_strides: &[isize],
) -> Result<()> {
    if dims.len() != cond_strides.len()
        || dims.len() != true_strides.len()
        || dims.len() != false_strides.len()
        || dims.len() != dst_strides.len()
    {
        return Err(Error::InvalidArgument(format!(
            "pointwise ternary rank mismatch: dims={} cond={} true={} false={} dst={}",
            dims.len(),
            cond_strides.len(),
            true_strides.len(),
            false_strides.len(),
            dst_strides.len()
        )));
    }
    Ok(())
}

pub fn as_byte_slice<T>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), std::mem::size_of_val(slice)) }
}
