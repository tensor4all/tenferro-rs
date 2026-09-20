//! CUDA-only session-admission contract for the linalg extension family.
//!
//! The main `extension::tests` module is compiled without the `cuda` feature,
//! so the CUDA admission table needs its own module rather than a `cfg` arm
//! inside a module that is itself cfg'd out.

use tenferro_tensor::DType;

use super::{LinalgExtensionOp, LinalgOp};

#[test]
fn session_support_admits_cuda_full_svd_and_still_rejects_the_missing_kernels() {
    use tenferro_gpu::cuda::CudaBackend;

    // cuSOLVER now executes full-matrices SVD in-session (gesvdj with
    // `econ = 0`, or gesvd with `jobu = jobvt = 'A'`), so admission must not
    // keep routing it to the compiled fallback.
    assert!(
        super::linalg_session_supported::<CudaBackend>(&LinalgExtensionOp::new(LinalgOp::SvdFull)),
        "CUDA must admit SvdFull now that it has a kernel"
    );
    // The ops cuSOLVER genuinely lacks must stay rejected: admission may not
    // over-claim just because its neighbour gained support.
    for op in [
        LinalgOp::FullPivLu,
        LinalgOp::Eig {
            input_dtype: DType::F64,
        },
    ] {
        assert!(
            !super::linalg_session_supported::<CudaBackend>(&LinalgExtensionOp::new(op)),
            "CUDA must keep rejecting {op:?}"
        );
    }
}
