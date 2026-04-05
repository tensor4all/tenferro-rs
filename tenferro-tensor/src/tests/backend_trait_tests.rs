use crate::backend::{SemiringBackend, TensorBackend};
use crate::cpu::CpuBackend;
use tenferro_algebra::Standard;

fn assert_backend_traits<B: TensorBackend + SemiringBackend<Standard<f64>>>() {}

#[test]
fn cpu_backend_implements_tensor_and_semiring_backends() {
    assert_backend_traits::<CpuBackend>();
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_backend_implements_tensor_and_semiring_backends() {
    assert_backend_traits::<crate::cuda::CudaBackend>();
}

#[cfg(feature = "rocm")]
#[test]
fn rocm_backend_implements_tensor_and_semiring_backends() {
    assert_backend_traits::<crate::rocm::RocmBackend>();
}
