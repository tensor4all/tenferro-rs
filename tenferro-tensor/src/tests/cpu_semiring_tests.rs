use crate::backend::SemiringBackend;
use crate::config::DotGeneralConfig;
use crate::cpu::CpuBackend;
use crate::types::TypedTensor;
use tenferro_algebra::Standard;

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> TypedTensor<f64> {
    TypedTensor::from_vec(shape, data)
}

#[test]
fn semiring_backend_default_add_mul_and_reduce_sum_work_for_standard_f64() {
    let mut backend = CpuBackend::new();
    let lhs = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let rhs = f64_tensor(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0]);

    let sum =
        <CpuBackend as SemiringBackend<Standard<f64>>>::add(&mut backend, &lhs, &rhs).unwrap();
    let prod =
        <CpuBackend as SemiringBackend<Standard<f64>>>::mul(&mut backend, &lhs, &rhs).unwrap();
    let reduced =
        <CpuBackend as SemiringBackend<Standard<f64>>>::reduce_sum(&mut backend, &lhs, &[1])
            .unwrap();
    let scalar =
        <CpuBackend as SemiringBackend<Standard<f64>>>::reduce_sum(&mut backend, &lhs, &[0, 1])
            .unwrap();
    let unchanged =
        <CpuBackend as SemiringBackend<Standard<f64>>>::reduce_sum(&mut backend, &lhs, &[])
            .unwrap();

    assert_eq!(sum.host_data(), &[11.0, 22.0, 33.0, 44.0]);
    assert_eq!(prod.host_data(), &[10.0, 40.0, 90.0, 160.0]);
    assert_eq!(reduced.shape, vec![2]);
    assert_eq!(reduced.host_data(), &[4.0, 6.0]);
    assert_eq!(scalar.shape, Vec::<usize>::new());
    assert_eq!(scalar.host_data(), &[10.0]);
    assert_eq!(unchanged.host_data(), lhs.host_data());
}

#[test]
fn semiring_backend_batched_gemm_handles_batch_and_contract_dimensions() {
    let mut backend = CpuBackend::new();
    let lhs = f64_tensor(
        vec![2, 2, 2],
        vec![1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0],
    );
    let rhs = f64_tensor(
        vec![2, 2, 2],
        vec![5.0, 6.0, 7.0, 8.0, 13.0, 14.0, 15.0, 16.0],
    );

    let out = <CpuBackend as SemiringBackend<Standard<f64>>>::batched_gemm(
        &mut backend,
        &lhs,
        &rhs,
        &DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![2],
            rhs_batch_dims: vec![2],
            lhs_rank: 3,
            rhs_rank: 3,
        },
    )
    .unwrap();

    assert_eq!(out.shape, vec![2, 2, 2]);
    assert_eq!(
        out.host_data(),
        &[23.0, 34.0, 31.0, 46.0, 271.0, 298.0, 311.0, 342.0]
    );
}

#[test]
fn semiring_backend_batched_gemm_inner_product_returns_rank0_scalar() {
    let mut backend = CpuBackend::new();
    let lhs = f64_tensor(vec![3], vec![1.0, 2.0, 3.0]);
    let rhs = f64_tensor(vec![3], vec![4.0, 5.0, 6.0]);

    let out = <CpuBackend as SemiringBackend<Standard<f64>>>::batched_gemm(
        &mut backend,
        &lhs,
        &rhs,
        &DotGeneralConfig {
            lhs_contracting_dims: vec![0],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
            lhs_rank: 1,
            rhs_rank: 1,
        },
    )
    .unwrap();

    assert_eq!(out.shape, Vec::<usize>::new());
    assert_eq!(out.host_data(), &[32.0]);
}
