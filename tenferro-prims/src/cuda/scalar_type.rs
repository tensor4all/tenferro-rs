use num_complex::Complex32;
use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};

use crate::cuda_ffi::{
    cutensorComputeDescriptor_t, CutensorDataType, CutensorVtable, CUTENSOR_C_32F, CUTENSOR_C_64F,
    CUTENSOR_R_32F, CUTENSOR_R_64F,
};
use crate::infra::typed_dispatch::{dispatch_complex_scalar_type, dispatch_real_scalar_type};

pub(super) fn scalar_data_type<T: Scalar>() -> Result<CutensorDataType> {
    dispatch_real_scalar_type!(T, Concrete, {
        let _ = std::marker::PhantomData::<Concrete>;
        return if std::mem::size_of::<Concrete>() == std::mem::size_of::<f32>() {
            Ok(CUTENSOR_R_32F)
        } else {
            Ok(CUTENSOR_R_64F)
        };
    });
    dispatch_complex_scalar_type!(T, Concrete, {
        let _ = std::marker::PhantomData::<Concrete>;
        return if std::mem::size_of::<Concrete>() == std::mem::size_of::<Complex32>() {
            Ok(CUTENSOR_C_32F)
        } else {
            Ok(CUTENSOR_C_64F)
        };
    });

    Err(Error::DeviceError(
        "Unsupported scalar type for CUDA backend".into(),
    ))
}

pub(super) fn scalar_compute_descriptor<T: Scalar>(
    vtable: &CutensorVtable,
) -> Result<cutensorComputeDescriptor_t> {
    dispatch_real_scalar_type!(T, Concrete, {
        let _ = std::marker::PhantomData::<Concrete>;
        return if std::mem::size_of::<Concrete>() == std::mem::size_of::<f32>() {
            Ok(vtable.compute_desc_32f)
        } else {
            Ok(vtable.compute_desc_64f)
        };
    });
    dispatch_complex_scalar_type!(T, Concrete, {
        let _ = std::marker::PhantomData::<Concrete>;
        return if std::mem::size_of::<Concrete>() == std::mem::size_of::<Complex32>() {
            Ok(vtable.compute_desc_32f)
        } else {
            Ok(vtable.compute_desc_64f)
        };
    });

    Err(Error::DeviceError(
        "Unsupported scalar type for CUDA backend".into(),
    ))
}
