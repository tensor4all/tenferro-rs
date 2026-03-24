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
    let pivots = Tensor::from_slice(&[1_i32, 2], &[2], MemoryOrder::ColumnMajor).unwrap();

    assert!(
        !<HipTensorLinalgBackend as TensorLinalgBackend<f64>>::has_linalg_support(
            crate::backend::tensor_api::LinalgCapabilityOp::SolveEx
        )
    );
    assert!(
        !<HipTensorLinalgBackend as TensorLinalgBackend<f64>>::has_linalg_support(
            crate::backend::tensor_api::LinalgCapabilityOp::LuFactorEx
        )
    );
    assert!(
        !<HipTensorLinalgBackend as TensorLinalgBackend<f64>>::has_linalg_support(
            crate::backend::tensor_api::LinalgCapabilityOp::CholeskyEx
        )
    );

    assert!(
        <HipTensorLinalgBackend as TensorLinalgBackend<f64>>::solve_ex(&mut ctx, &a, &b).is_err()
    );
    assert!(<HipTensorLinalgBackend as TensorLinalgBackend<f64>>::solve(&mut ctx, &a, &b).is_err());
    assert!(
        <HipTensorLinalgBackend as TensorLinalgBackend<f64>>::lu_solve(&mut ctx, &a, &pivots, &b)
            .is_err()
    );
    assert!(
        <HipTensorLinalgBackend as TensorLinalgBackend<f64>>::solve_triangular(
            &mut ctx, &a, &b, true,
        )
        .is_err()
    );
    assert!(<HipTensorLinalgBackend as TensorLinalgBackend<f64>>::qr(&mut ctx, &a).is_err());
    assert!(<HipTensorLinalgBackend as TensorLinalgBackend<f64>>::thin_svd(&mut ctx, &a).is_err());
    assert!(<HipTensorLinalgBackend as TensorLinalgBackend<f64>>::svdvals(&mut ctx, &a).is_err());
    assert!(
        <HipTensorLinalgBackend as TensorLinalgBackend<f64>>::lu_factor_ex(&mut ctx, &a).is_err()
    );
    assert!(<HipTensorLinalgBackend as TensorLinalgBackend<f64>>::lu_factor(&mut ctx, &a).is_err());
    assert!(
        <HipTensorLinalgBackend as TensorLinalgBackend<f64>>::lu_factor_no_pivot(&mut ctx, &a)
            .is_err()
    );
    assert!(
        <HipTensorLinalgBackend as TensorLinalgBackend<f64>>::cholesky_ex(&mut ctx, &a).is_err()
    );
    assert!(<HipTensorLinalgBackend as TensorLinalgBackend<f64>>::cholesky(&mut ctx, &a).is_err());
    assert!(<HipTensorLinalgBackend as TensorLinalgBackend<f64>>::eigen_sym(&mut ctx, &a).is_err());
    assert!(<HipTensorLinalgBackend as TensorLinalgBackend<f64>>::eig(&mut ctx, &a).is_err());
}
