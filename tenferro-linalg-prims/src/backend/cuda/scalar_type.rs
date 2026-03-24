use tenferro_algebra::Conjugate;

/// Hidden CUDA dtype inventory for linalg backend dispatch.
///
/// # Examples
///
/// ```ignore
/// let _ = tenferro_linalg_prims::backend::CudaDataType::F64;
/// ```
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaDataType {
    F32,
    F64,
    Complex32,
    Complex64,
}

/// Hidden marker for scalars that can bind to the CUDA linalg backend.
///
/// High-level crates use this only as a compile-time contract so their generic
/// runtime dispatch does not advertise CUDA linalg for unsupported dtypes.
///
/// # Examples
///
/// ```ignore
/// fn require_cuda_linalg<T: tenferro_linalg_prims::backend::CudaLinalgScalar>() {}
/// require_cuda_linalg::<f64>();
/// ```
#[doc(hidden)]
pub trait CudaLinalgScalar: crate::KernelLinalgScalar + Conjugate {
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
