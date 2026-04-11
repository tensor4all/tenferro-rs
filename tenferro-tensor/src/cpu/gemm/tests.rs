use super::try_fuse_dims;

#[cfg(feature = "cpu-blas")]
use super::dot_general;
#[cfg(feature = "cpu-faer")]
use super::faer_gemm::FaerGemm;
#[cfg(feature = "cpu-blas")]
use crate::buffer_pool::BufferPool;
#[cfg(feature = "cpu-blas")]
use crate::config::DotGeneralConfig;
#[cfg(feature = "cpu-faer")]
use crate::cpu::CpuContext;
#[cfg(feature = "cpu-blas")]
use crate::types::TypedTensor;

#[test]
fn try_fuse_dims_reversed_strides() {
    assert_eq!(try_fuse_dims(&[3, 4], &[4, 1]), Some((12, 1)));
}

#[test]
fn try_fuse_dims_sorted_strides_unchanged() {
    assert_eq!(try_fuse_dims(&[3, 4], &[1, 3]), Some((12, 1)));
}

#[test]
fn try_fuse_dims_non_adjacent_fails() {
    assert_eq!(try_fuse_dims(&[3, 2], &[1, 6]), None);
}

#[test]
fn try_fuse_dims_single_dim() {
    assert_eq!(try_fuse_dims(&[5], &[3]), Some((5, 3)));
}

#[test]
fn try_fuse_dims_empty() {
    assert_eq!(try_fuse_dims(&[], &[]), Some((1, 0)));
}

#[cfg(feature = "cpu-blas")]
#[test]
fn blas_dot_general_contract_trailing_rhs_dim() {
    let lhs = TypedTensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs = TypedTensor::from_vec(vec![2, 3], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![1],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
        lhs_rank: 2,
        rhs_rank: 2,
    };
    let mut buffers = BufferPool::new();
    let out = dot_general(&mut buffers, &lhs, &rhs, &config).expect("dot_general should succeed");

    assert_eq!(out.shape, vec![2, 2]);
    assert_eq!(out.host_data(), &[89.0, 116.0, 98.0, 128.0]);
}

#[cfg(feature = "cpu-faer")]
#[test]
fn faer_strided_gemm_accumulates_with_nontrivial_beta() {
    let a = [1.0, 0.0, 0.0, 1.0];
    let b = [10.0, 20.0, 30.0, 40.0];
    let mut c = [1.0, 2.0, 3.0, 4.0];
    let ctx = CpuContext::with_threads(1);

    unsafe {
        <f64 as FaerGemm>::strided_gemm(
            &ctx,
            1.0,
            a.as_ptr(),
            2,
            2,
            1,
            2,
            b.as_ptr(),
            2,
            1,
            2,
            2.0,
            c.as_mut_ptr(),
            1,
            2,
        );
    }

    assert_eq!(c, [12.0, 24.0, 36.0, 48.0]);
}

#[cfg(feature = "cpu-faer")]
#[test]
fn faer_strided_gemm_accumulates_with_unit_beta_without_prescaling() {
    let a = [1.0, 0.0, 0.0, 1.0];
    let b = [10.0, 20.0, 30.0, 40.0];
    let mut c = [1.0, 2.0, 3.0, 4.0];
    let ctx = CpuContext::with_threads(1);

    unsafe {
        <f64 as FaerGemm>::strided_gemm(
            &ctx,
            1.0,
            a.as_ptr(),
            2,
            2,
            1,
            2,
            b.as_ptr(),
            2,
            1,
            2,
            1.0,
            c.as_mut_ptr(),
            1,
            2,
        );
    }

    assert_eq!(c, [11.0, 22.0, 33.0, 44.0]);
}
