use super::*;
use tenferro_tensor::MemoryOrder;

fn dummy_tensor() -> Tensor<f64> {
    Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap()
}

#[cfg(feature = "cuda")]
fn with_cuda_ctx<T>(f: impl FnOnce(&mut tenferro_prims::CudaContext) -> T) -> Option<T> {
    let path = [
        "/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so",
        "/usr/lib/x86_64-linux-gnu/libcutensor.so",
        "/usr/lib/libcutensor.so",
    ]
    .into_iter()
    .find(|path| std::path::Path::new(path).exists())?;
    let (_backend, mut ctx) = tenferro_prims::CudaBackend::load(path).ok()?;
    Some(f(&mut ctx))
}

#[cfg(not(feature = "cuda"))]
fn with_cuda_ctx<T>(f: impl FnOnce(&mut tenferro_prims::CudaContext) -> T) -> Option<T> {
    let mut ctx = tenferro_prims::CudaContext::new();
    Some(f(&mut ctx))
}

#[test]
fn cuda_stubs_return_device_error() {
    with_cuda_ctx(|ctx| {
        let a = dummy_tensor();
        let b = dummy_tensor();
        let pivots = Tensor::from_slice(&[1_i32, 2], &[2], MemoryOrder::ColumnMajor).unwrap();

        assert!(
            !<CudaTensorLinalgBackend as TensorLinalgBackend<f64>>::has_linalg_support(
                crate::backend::tensor_api::LinalgCapabilityOp::SolveEx
            )
        );
        assert!(
            !<CudaTensorLinalgBackend as TensorLinalgBackend<f64>>::has_linalg_support(
                crate::backend::tensor_api::LinalgCapabilityOp::LuFactorEx
            )
        );
        assert!(
            !<CudaTensorLinalgBackend as TensorLinalgBackend<f64>>::has_linalg_support(
                crate::backend::tensor_api::LinalgCapabilityOp::CholeskyEx
            )
        );

        assert!(
            <CudaTensorLinalgBackend as TensorLinalgBackend<f64>>::solve_ex(ctx, &a, &b).is_err()
        );
        assert!(<CudaTensorLinalgBackend as TensorLinalgBackend<f64>>::solve(ctx, &a, &b).is_err());
        assert!(
            <CudaTensorLinalgBackend as TensorLinalgBackend<f64>>::lu_solve(ctx, &a, &pivots, &b)
                .is_err()
        );
        assert!(
            <CudaTensorLinalgBackend as TensorLinalgBackend<f64>>::solve_triangular(
                ctx, &a, &b, true,
            )
            .is_err()
        );
        assert!(<CudaTensorLinalgBackend as TensorLinalgBackend<f64>>::qr(ctx, &a).is_err());
        assert!(<CudaTensorLinalgBackend as TensorLinalgBackend<f64>>::thin_svd(ctx, &a).is_err());
        assert!(<CudaTensorLinalgBackend as TensorLinalgBackend<f64>>::svdvals(ctx, &a).is_err());
        assert!(
            <CudaTensorLinalgBackend as TensorLinalgBackend<f64>>::lu_factor_ex(ctx, &a).is_err()
        );
        assert!(<CudaTensorLinalgBackend as TensorLinalgBackend<f64>>::lu_factor(ctx, &a).is_err());
        assert!(
            <CudaTensorLinalgBackend as TensorLinalgBackend<f64>>::lu_factor_no_pivot(ctx, &a)
                .is_err()
        );
        assert!(
            <CudaTensorLinalgBackend as TensorLinalgBackend<f64>>::cholesky_ex(ctx, &a).is_err()
        );
        assert!(<CudaTensorLinalgBackend as TensorLinalgBackend<f64>>::cholesky(ctx, &a).is_err());
        assert!(<CudaTensorLinalgBackend as TensorLinalgBackend<f64>>::eigen_sym(ctx, &a).is_err());
        assert!(<CudaTensorLinalgBackend as TensorLinalgBackend<f64>>::eig(ctx, &a).is_err());
    });
}
