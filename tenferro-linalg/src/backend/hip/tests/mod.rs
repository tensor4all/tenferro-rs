use super::*;
use tenferro_tensor::MemoryOrder;

fn dummy_tensor() -> Tensor<f64> {
    Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn hip_stubs_return_device_error() {
    let mut ctx = tenferro_prims::RocmContext::new();
    let a = dummy_tensor();
    let b = dummy_tensor();
    assert!(HipTensorLinalgBackend::solve(&mut ctx, &a, &b).is_err());
    assert!(HipTensorLinalgBackend::solve_triangular(&mut ctx, &a, &b, true).is_err());
    assert!(HipTensorLinalgBackend::qr(&mut ctx, &a).is_err());
    assert!(HipTensorLinalgBackend::thin_svd(&mut ctx, &a).is_err());
    assert!(HipTensorLinalgBackend::lu_factor(&mut ctx, &a).is_err());
    assert!(HipTensorLinalgBackend::cholesky(&mut ctx, &a).is_err());
    assert!(HipTensorLinalgBackend::eigen_sym(&mut ctx, &a).is_err());
    assert!(HipTensorLinalgBackend::eig(&mut ctx, &a).is_err());
}
