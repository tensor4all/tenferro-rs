#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CudaDataType {
    F32,
    F64,
    Complex32,
    Complex64,
}

pub(crate) trait CudaLinalgScalar: crate::KernelLinalgScalar {
    fn cuda_data_type() -> CudaDataType;
}

macro_rules! impl_cuda_linalg_scalar {
    ($ty:ty, $dtype:expr) => {
        impl CudaLinalgScalar for $ty {
            fn cuda_data_type() -> CudaDataType {
                $dtype
            }
        }
    };
}

impl_cuda_linalg_scalar!(f32, CudaDataType::F32);
impl_cuda_linalg_scalar!(f64, CudaDataType::F64);
impl_cuda_linalg_scalar!(num_complex::Complex32, CudaDataType::Complex32);
impl_cuda_linalg_scalar!(num_complex::Complex64, CudaDataType::Complex64);
