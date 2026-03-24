use super::*;
use tenferro_prims::CpuContext;

#[test]
fn diag_extract_keeps_diagonal_axis_before_batches() {
    let input = Tensor::from_slice(
        &[
            1.0_f64, 4.0, 7.0, 8.0, 2.0, 5.0, 9.0, 6.0, 3.0, // batch 0
            10.0, 40.0, 70.0, 80.0, 20.0, 50.0, 90.0, 60.0, 30.0, // batch 1
        ],
        &[3, 3, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let diagonal = crate::ad_helpers::diag_extract(&input).unwrap();

    assert_eq!(diagonal.dims(), &[3, 2]);
    assert_eq!(
        tensor_data(&diagonal),
        vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0]
    );
}

#[test]
fn trace_tensor_reduces_matrix_axes_and_preserves_batches() {
    let mut ctx = CpuContext::new(1);
    let input = Tensor::from_slice(
        &[
            1.0_f64, 4.0, 9.0, 2.0, // batch 0
            10.0, 40.0, 90.0, 20.0, // batch 1
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let trace = crate::ad_helpers::trace_tensor(&mut ctx, &input).unwrap();

    assert_eq!(trace.dims(), &[2]);
    assert_eq!(tensor_data(&trace), vec![3.0, 30.0]);
}

#[test]
fn diag_scatter_embeds_diagonal_vectors_and_batch_scalars() {
    let mut ctx = CpuContext::new(1);

    let diagonal_values = Tensor::from_slice(
        &[1.0_f64, 2.0, 3.0, 10.0, 20.0, 30.0],
        &[3, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let embedded = crate::ad_helpers::diag_scatter(&mut ctx, &diagonal_values, &[3, 3, 2]).unwrap();
    assert_eq!(embedded.dims(), &[3, 3, 2]);
    assert_eq!(
        tensor_data(&embedded),
        vec![
            1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 10.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0,
            30.0,
        ]
    );

    let batch_scalars =
        Tensor::from_slice(&[5.0_f64, 7.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let scaled_identity =
        crate::ad_helpers::diag_embed(&mut ctx, &batch_scalars, &[3, 3, 2]).unwrap();
    assert_eq!(
        tensor_data(&scaled_identity),
        vec![
            5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 7.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0,
            7.0,
        ]
    );
}

#[test]
fn diag_scatter_add_only_updates_diagonal_entries() {
    let mut ctx = CpuContext::new(1);
    let diagonal =
        Tensor::from_slice(&[1.0_f64, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();
    let base = Tensor::from_slice(&[5.0_f64; 9], &[3, 3], MemoryOrder::ColumnMajor).unwrap();

    let updated = crate::ad_helpers::diag_scatter_add(&mut ctx, &diagonal, &base).unwrap();

    assert_eq!(
        tensor_data(&updated),
        vec![6.0, 5.0, 5.0, 5.0, 7.0, 5.0, 5.0, 5.0, 8.0]
    );
    assert_eq!(tensor_data(&base), vec![5.0; 9]);
}
