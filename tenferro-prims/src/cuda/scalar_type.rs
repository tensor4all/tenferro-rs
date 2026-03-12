use num_complex::{Complex32, Complex64};
use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};

use crate::cuda_ffi::{
    cutensorComputeDescriptor_t, CutensorDataType, CutensorVtable, CUTENSOR_C_32F, CUTENSOR_C_64F,
    CUTENSOR_R_32F, CUTENSOR_R_64F,
};
use crate::typed_dispatch::{
    dispatch_complex_scalar_type, dispatch_real_scalar_type, dispatch_standard_scalar_type,
};

pub(super) fn scalar_data_type<T: Scalar>() -> Result<CutensorDataType> {
    macro_rules! cutensor_data_type_for {
        (f32) => {
            CUTENSOR_R_32F
        };
        (f64) => {
            CUTENSOR_R_64F
        };
        (Complex32) => {
            CUTENSOR_C_32F
        };
        (Complex64) => {
            CUTENSOR_C_64F
        };
    }

    dispatch_standard_scalar_type!(T, Concrete, {
        return Ok(cutensor_data_type_for!(Concrete));
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
