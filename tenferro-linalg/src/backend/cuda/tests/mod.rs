use super::*;
use tenferro_tensor::MemoryOrder;

fn dummy_tensor() -> Tensor<f64> {
    Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn cuda_stubs_return_device_error() {
    let mut ctx = tenferro_prims::CudaContext::new();
    let a = dummy_tensor();
    let b = dummy_tensor();
    assert!(CudaTensorLinalgBackend::solve(&mut ctx, &a, &b).is_err());
    assert!(CudaTensorLinalgBackend::solve_triangular(&mut ctx, &a, &b, true).is_err());
    assert!(CudaTensorLinalgBackend::qr(&mut ctx, &a).is_err());
    assert!(CudaTensorLinalgBackend::thin_svd(&mut ctx, &a).is_err());
    assert!(CudaTensorLinalgBackend::lu_factor(&mut ctx, &a).is_err());
    assert!(CudaTensorLinalgBackend::cholesky(&mut ctx, &a).is_err());
    assert!(CudaTensorLinalgBackend::eigen_sym(&mut ctx, &a).is_err());
    assert!(CudaTensorLinalgBackend::eig(&mut ctx, &a).is_err());
}
