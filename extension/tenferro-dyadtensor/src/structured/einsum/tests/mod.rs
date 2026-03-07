use super::*;
use crate::{set_default_runtime, RuntimeContext};
use tenferro_prims::{CpuContext, CudaContext};
use tenferro_tensor::MemoryOrder;

fn vector(values: &[f64]) -> Tensor<f64> {
    Tensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn matrix(values: &[f64], rows: usize, cols: usize) -> Tensor<f64> {
    Tensor::<f64>::from_slice(values, &[rows, cols], MemoryOrder::ColumnMajor).unwrap()
}

fn as_slice(tensor: &Tensor<f64>) -> &[f64] {
    tensor
        .buffer()
        .as_slice()
        .unwrap_or_else(|| panic!("expected CPU-backed contiguous tensor"))
}

#[test]
fn public_structured_ops_cover_cpu_and_runtime_error_paths() {
    let diag = StructuredTensor::from_diagonal_vector(vector(&[2.0, 3.0]), 2).unwrap();
    let dense_layout = StructuredTensor::from_dense(matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2));
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);

    {
        let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
        assert_eq!(dense_layout.to_dense().unwrap().dims(), &[2, 2]);
        assert_eq!(as_slice(&diag.to_dense().unwrap()), &[2.0, 0.0, 0.0, 3.0]);

        let out = StructuredTensor::einsum_with_subscripts(&subs, &[&diag, &diag]).unwrap();
        assert!(out.is_diag());
        assert_eq!(as_slice(out.payload()), &[4.0, 9.0]);

        let err = StructuredTensor::<f64>::einsum_with_subscripts(&subs, &[]).unwrap_err();
        assert!(matches!(err, Error::InvalidAdTensor { .. }));
    }

    {
        let _guard = set_default_runtime(RuntimeContext::Cuda(CudaContext::new()));
        let err = diag.to_dense().unwrap_err();
        assert!(matches!(
            err,
            Error::UnsupportedRuntimeOp {
                op: "structured_to_dense",
                runtime: "cuda"
            }
        ));

        let err = StructuredTensor::einsum_with_subscripts(&subs, &[&diag, &diag]).unwrap_err();
        assert!(matches!(
            err,
            Error::UnsupportedRuntimeOp {
                op: "structured_einsum",
                runtime: "cuda"
            }
        ));
    }
}

#[test]
fn compression_and_tangent_accumulation_cover_dense_and_diag_paths() {
    let mut ctx = CpuContext::new(1);
    let dense_layout = StructuredTensor::from_dense(matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2));
    let diag_layout = StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap();

    let dense =
        compress_dense_to_layout_in_ctx(&mut ctx, dense_layout.payload(), &dense_layout).unwrap();
    assert!(dense.is_dense());

    let diag = compress_dense_to_layout_in_ctx(
        &mut ctx,
        &matrix(&[5.0, 7.0, 11.0, 13.0], 2, 2),
        &diag_layout,
    )
    .unwrap();
    assert_eq!(as_slice(diag.payload()), &[5.0, 13.0]);

    let err =
        compress_dense_to_layout_in_ctx(&mut ctx, &vector(&[1.0, 2.0]), &diag_layout).unwrap_err();
    assert!(matches!(err, Error::InvalidAdTensor { .. }));

    let diag_rhs = diag_layout.with_payload_like(vector(&[3.0, 4.0])).unwrap();
    let sum = accumulate_tangent(diag_layout.clone(), &diag_rhs).unwrap();
    assert_eq!(as_slice(sum.payload()), &[4.0, 6.0]);

    let err = accumulate_tangent(diag_layout, &dense_layout).unwrap_err();
    assert!(matches!(err, Error::InvalidAdTensor { .. }));
}

#[test]
fn internal_helpers_cover_normalization_and_error_branches() {
    let mut ctx = CpuContext::new(1);

    assert_eq!(unique_ids_first_appearance(&[3, 3, 1, 3, 1]), vec![3, 1]);
    assert_eq!(first_duplicate_pair(&[4, 5, 4]), Some((0, 2)));
    assert_eq!(first_duplicate_pair(&[4, 5, 6]), None);

    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let rev = reverse_subscripts(&subs, 1);
    assert_eq!(rev.inputs, vec![vec![0, 2], vec![0, 1]]);
    assert_eq!(rev.output, vec![1, 2]);

    let (same_payload, same_roots) =
        normalize_payload_for_roots(&mut ctx, &vector(&[1.0, 2.0]), &[0]).unwrap();
    assert_eq!(same_roots, vec![0]);
    assert_eq!(as_slice(&same_payload), &[1.0, 2.0]);

    let (diag_payload, diag_roots) =
        normalize_payload_for_roots(&mut ctx, &matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2), &[0, 0])
            .unwrap();
    assert_eq!(diag_roots, vec![0]);
    assert_eq!(as_slice(&diag_payload), &[1.0, 4.0]);

    let err = normalize_payload_for_roots(&mut ctx, &vector(&[1.0, 2.0]), &[0, 0]).unwrap_err();
    assert!(matches!(err, Error::InvalidAdTensor { .. }));

    let rank1_subs = Subscripts::new(&[&[0]], &[0]);
    let diag = StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap();
    let err = einsum_with_subscripts_in_ctx(&mut ctx, &rank1_subs, &[&diag]).unwrap_err();
    assert!(matches!(err, Error::InvalidAdTensor { .. }));

    let err = usize_vec_to_u32(&[usize::MAX]).unwrap_err();
    assert!(matches!(err, Error::InvalidAdTensor { .. }));
}
