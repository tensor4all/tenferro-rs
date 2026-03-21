use crate::backend::TensorLinalgContextFor;

fn assert_ctx<T, C>()
where
    T: crate::KernelLinalgScalar,
    C: TensorLinalgContextFor<T>,
{
}

#[test]
fn cpu_context_is_bound_from_linalg_prims() {
    assert_ctx::<f64, tenferro_prims::CpuContext>();
}

#[test]
fn cuda_context_is_bound_from_linalg_prims() {
    assert_ctx::<f64, tenferro_prims::CudaContext>();
}

#[test]
fn cpu_backend_still_solves_after_move() {
    let mut ctx = tenferro_prims::CpuContext::new(1);
    let a = tenferro_tensor::Tensor::from_slice(
        &[2.0_f64, 0.0, 0.0, 4.0],
        &[2, 2],
        tenferro_tensor::MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b = tenferro_tensor::Tensor::from_slice(
        &[4.0_f64, 8.0],
        &[2],
        tenferro_tensor::MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let x = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::solve(
        &mut ctx, &a, &b,
    )
    .unwrap();
    assert_eq!(x.dims(), &[2]);
    assert_eq!(x.buffer().as_slice().unwrap(), &[2.0, 2.0]);
}

#[test]
fn cpu_backend_supports_core_factorizations_after_move() {
    let mut ctx = tenferro_prims::CpuContext::new(1);
    let spd = tenferro_tensor::Tensor::from_slice(
        &[4.0_f64, 1.0, 1.0, 3.0],
        &[2, 2],
        tenferro_tensor::MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let general = tenferro_tensor::Tensor::from_slice(
        &[1.0_f64, 2.0, 0.0, 3.0],
        &[2, 2],
        tenferro_tensor::MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let qr = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::qr(
        &mut ctx, &spd,
    )
    .unwrap();
    assert_eq!(qr.q.dims(), &[2, 2]);
    assert_eq!(qr.r.dims(), &[2, 2]);

    let svd = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::thin_svd(
        &mut ctx, &spd,
    )
    .unwrap();
    assert_eq!(svd.u.dims(), &[2, 2]);
    assert_eq!(svd.s.dims(), &[2]);
    assert_eq!(svd.vt.dims(), &[2, 2]);

    let lu = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::lu_factor(
        &mut ctx, &general,
    )
    .unwrap();
    assert_eq!(lu.l.dims(), &[2, 2]);
    assert_eq!(lu.u.dims(), &[2, 2]);
    assert_eq!(lu.pivots.len(), 2);

    let chol = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::cholesky(
        &mut ctx, &spd,
    )
    .unwrap();
    assert_eq!(chol.dims(), &[2, 2]);

    let eigen =
        <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::eigen_sym(
            &mut ctx, &spd,
        )
        .unwrap();
    assert_eq!(eigen.values.dims(), &[2]);
    assert_eq!(eigen.vectors.dims(), &[2, 2]);

    let eig = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::eig(
        &mut ctx, &general,
    )
    .unwrap();
    assert_eq!(eig.values.dims(), &[2]);
    assert_eq!(eig.vectors.dims(), &[2, 2]);
}

#[test]
fn cpu_backend_svdvals_matches_thin_svd_after_move() {
    let mut ctx = tenferro_prims::CpuContext::new(1);
    let a = tenferro_tensor::Tensor::from_slice(
        &[1.0_f64, 0.0, 0.0, 2.0],
        &[2, 2],
        tenferro_tensor::MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let svd = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::thin_svd(
        &mut ctx, &a,
    )
    .unwrap();
    let s = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::svdvals(
        &mut ctx, &a,
    )
    .unwrap();

    assert_eq!(s.dims(), &[2]);
    assert_eq!(
        s.buffer().as_slice().unwrap(),
        svd.s.buffer().as_slice().unwrap()
    );
}

#[test]
fn ex_capabilities_track_cpu_ex_implementation_state() {
    use crate::LinalgCapabilityOp::{CholeskyEx, LuFactorEx, SolveEx};

    assert!(
        <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(SolveEx),
        "CPU should report SolveEx once the corresponding EX semantics exist",
    );
    assert!(
        !<crate::backend::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(SolveEx),
        "CUDA should not report SolveEx before support is wired",
    );
    assert!(
        !<crate::backend::HipTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(SolveEx),
        "HIP should not report SolveEx before support is wired",
    );

    assert!(
        <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(LuFactorEx),
        "CPU should report LuFactorEx once the corresponding EX semantics exist",
    );
    assert!(
        <crate::backend::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(LuFactorEx),
        "CUDA should report LuFactorEx once native kernels are wired",
    );
    assert!(
        !<crate::backend::HipTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(LuFactorEx),
        "HIP should not report LuFactorEx before support is wired",
    );

    assert!(
        <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(CholeskyEx),
        "CPU should report CholeskyEx once the corresponding EX semantics exist",
    );
    assert!(
        !<crate::backend::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(CholeskyEx),
        "CUDA should not report CholeskyEx before support is wired",
    );
    assert!(
        !<crate::backend::HipTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(CholeskyEx),
        "HIP should not report CholeskyEx before support is wired",
    );
}

#[test]
fn cpu_backend_solve_ex_preserves_successful_batches_and_reports_zero_pivot() {
    let mut ctx = tenferro_prims::CpuContext::new(1);
    let a = tenferro_tensor::Tensor::from_slice(
        &[
            1.0_f64, 0.0, 0.0, 1.0, //
            1.0, 2.0, 2.0, 4.0,
        ],
        &[2, 2, 2],
        tenferro_tensor::MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b = tenferro_tensor::Tensor::from_slice(
        &[3.0_f64, -1.0, 1.0, 1.0],
        &[2, 2],
        tenferro_tensor::MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let result =
        <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::solve_ex(
            &mut ctx, &a, &b,
        )
        .unwrap();
    assert_eq!(result.info, vec![0, 2]);
    assert_eq!(tensor_data(&result.solution), vec![3.0, -1.0, 0.0, 0.0]);
}

#[test]
fn cpu_backend_lu_factor_ex_preserves_successful_batches_and_reports_zero_pivot() {
    let mut ctx = tenferro_prims::CpuContext::new(1);
    let a = tenferro_tensor::Tensor::from_slice(
        &[
            1.0_f64, 0.0, 0.0, 1.0, //
            1.0, 2.0, 2.0, 4.0,
        ],
        &[2, 2, 2],
        tenferro_tensor::MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let plain =
        <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::lu_factor(
            &mut ctx, &a,
        )
        .unwrap();
    let result =
        <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::lu_factor_ex(
            &mut ctx, &a,
        )
        .unwrap();

    assert_eq!(result.info, vec![0, 2]);
    assert_eq!(tensor_data(&result.l), tensor_data(&plain.l));
    assert_eq!(tensor_data(&result.u), tensor_data(&plain.u));
    assert_eq!(result.pivots, plain.pivots);
}

#[test]
fn cpu_backend_cholesky_ex_preserves_successful_batches_and_reports_minor() {
    let mut ctx = tenferro_prims::CpuContext::new(1);
    let good = tenferro_tensor::Tensor::from_slice(
        &[4.0_f64, 2.0, 2.0, 3.0],
        &[2, 2],
        tenferro_tensor::MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let a = tenferro_tensor::Tensor::from_slice(
        &[
            4.0_f64, 2.0, 2.0, 3.0, //
            1.0, 2.0, 2.0, 1.0,
        ],
        &[2, 2, 2],
        tenferro_tensor::MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let result =
        <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::cholesky_ex(
            &mut ctx, &a,
        )
        .unwrap();
    let plain =
        <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::cholesky(
            &mut ctx, &good,
        )
        .unwrap();

    assert_eq!(result.info, vec![0, 2]);
    assert_eq!(&tensor_data(&result.l)[..4], tensor_data(&plain).as_slice());
    assert_eq!(&tensor_data(&result.l)[4..], &[0.0, 0.0, 0.0, 0.0]);
}

fn tensor_data(tensor: &tenferro_tensor::Tensor<f64>) -> Vec<f64> {
    let contiguous = tensor.contiguous(tenferro_tensor::MemoryOrder::ColumnMajor);
    let offset = contiguous.offset() as usize;
    let len = contiguous.len();
    contiguous.buffer().as_slice().unwrap()[offset..offset + len].to_vec()
}

#[test]
fn linalg_utils_matrix_stride_uses_leading_matrix_dims() {
    assert_eq!(
        crate::backend::linalg_utils::matrix_stride(&[2, 3]).unwrap(),
        6
    );
    assert_eq!(
        crate::backend::linalg_utils::matrix_stride(&[2, 3, 4, 5]).unwrap(),
        6
    );
}

#[test]
fn linalg_utils_clone_batched_column_major_repackages_permuted_batches() {
    let mut ctx = tenferro_prims::CpuContext::new(1);
    let base = tenferro_tensor::Tensor::from_slice(
        &[
            1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
        ],
        &[2, 3, 2],
        tenferro_tensor::MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let permuted = base.permute(&[1, 0, 2]).unwrap();

    let cloned =
        crate::backend::linalg_utils::clone_batched_column_major(&mut ctx, &permuted).unwrap();

    assert_eq!(cloned.dims(), permuted.dims());
    assert!(cloned.is_col_major_contiguous());
    assert_eq!(
        cloned.logical_memory_space(),
        permuted.logical_memory_space()
    );
    assert_eq!(tensor_data(&cloned), tensor_data(&permuted));
}
