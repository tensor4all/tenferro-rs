use super::*;
use crate::exec::{ExecInstruction, ExecOp, ExecProgram};
use tenferro_cpu::CpuBackend;
use tenferro_ops::{dim_expr::DimExpr, ShapeExtent};
use tenferro_tensor::backend::{ElementwiseFusionInputView, ElementwiseFusionOp};
use tenferro_tensor::{DType, Tensor};

fn dim_shape(shape: &[usize]) -> Vec<DimExpr> {
    shape.iter().copied().map(DimExpr::Const).collect()
}

fn exact_extents(shape: &[usize]) -> Vec<ShapeExtent<DimExpr>> {
    dim_shape(shape)
        .into_iter()
        .map(ShapeExtent::exact)
        .collect()
}

fn broadcast_with_shape(
    input: usize,
    output: usize,
    shape: &[usize],
    dims: Vec<usize>,
) -> ExecInstruction {
    ExecInstruction {
        op: ExecOp::BroadcastInDim {
            shape: dim_shape(shape),
            dims,
        },
        input_slots: vec![input],
        output_slots: vec![output],
        dtype: DType::F64,
        output_shapes: vec![dim_shape(shape)].into(),
        output_extents: vec![exact_extents(shape)].into(),
        last_use: vec![true],
    }
}

fn multiply_with_shape(lhs: usize, rhs: usize, output: usize, shape: &[usize]) -> ExecInstruction {
    ExecInstruction {
        op: ExecOp::Multiply,
        input_slots: vec![lhs, rhs],
        output_slots: vec![output],
        dtype: DType::F64,
        output_shapes: vec![dim_shape(shape)].into(),
        output_extents: vec![exact_extents(shape)].into(),
        last_use: vec![true, true],
    }
}

fn add_with_shape(lhs: usize, rhs: usize, output: usize, shape: &[usize]) -> ExecInstruction {
    ExecInstruction {
        op: ExecOp::Add,
        input_slots: vec![lhs, rhs],
        output_slots: vec![output],
        dtype: DType::F64,
        output_shapes: vec![dim_shape(shape)].into(),
        output_extents: vec![exact_extents(shape)].into(),
        last_use: vec![true, true],
    }
}

fn broadcast(input: usize, output: usize, dims: Vec<usize>) -> ExecInstruction {
    ExecInstruction {
        op: ExecOp::BroadcastInDim {
            shape: vec![DimExpr::Const(2), DimExpr::Const(2)],
            dims,
        },
        input_slots: vec![input],
        output_slots: vec![output],
        dtype: DType::F64,
        output_shapes: vec![vec![DimExpr::Const(2), DimExpr::Const(2)]].into(),
        output_extents: vec![vec![
            ShapeExtent::exact(DimExpr::Const(2)),
            ShapeExtent::exact(DimExpr::Const(2)),
        ]]
        .into(),
        last_use: vec![true],
    }
}

fn multiply(lhs: usize, rhs: usize, output: usize) -> ExecInstruction {
    ExecInstruction {
        op: ExecOp::Multiply,
        input_slots: vec![lhs, rhs],
        output_slots: vec![output],
        dtype: DType::F64,
        output_shapes: vec![vec![DimExpr::Const(2), DimExpr::Const(2)]].into(),
        output_extents: vec![vec![
            ShapeExtent::exact(DimExpr::Const(2)),
            ShapeExtent::exact(DimExpr::Const(2)),
        ]]
        .into(),
        last_use: vec![true, true],
    }
}

#[test]
fn elementwise_fusion_plan_uses_dense_segment_value_ids() {
    let instructions = vec![
        add_with_shape(10, 20, 30, &[4]),
        multiply_with_shape(30, 10, 40, &[4]),
    ];

    let plan = build_elementwise_fusion_plan(&instructions, &[10, 20], &[40])
        .expect("add-multiply segment should build a fusion plan");

    assert_eq!(plan.dtype(), DType::F64);
    assert_eq!(plan.input_count(), 2);
    assert!(plan.input_views().iter().all(|view| view.is_identity()));
    assert_eq!(plan.outputs(), &[3]);
    assert_eq!(plan.ops().len(), 2);
    assert_eq!(plan.ops()[0].op(), ElementwiseFusionOp::Add);
    assert_eq!(plan.ops()[0].inputs(), &[0, 1]);
    assert_eq!(plan.ops()[1].op(), ElementwiseFusionOp::Multiply);
    assert_eq!(plan.ops()[1].inputs(), &[2, 0]);
}

#[test]
fn elementwise_fusion_plan_absorbs_broadcast_input_views() {
    let instructions = vec![
        broadcast_with_shape(0, 2, &[3, 2], vec![0]),
        broadcast_with_shape(1, 3, &[3, 2], vec![1]),
        multiply_with_shape(2, 3, 4, &[3, 2]),
    ];

    let plan = build_elementwise_fusion_plan(&instructions, &[0, 1], &[4])
        .expect("broadcast-multiply segment should build a fusion plan");

    assert_eq!(plan.input_count(), 2);
    assert_eq!(plan.outputs(), &[2]);
    assert_eq!(plan.ops().len(), 1);
    assert_eq!(plan.ops()[0].op(), ElementwiseFusionOp::Multiply);
    assert_eq!(plan.ops()[0].inputs(), &[0, 1]);
    assert_eq!(
        plan.input_views()[0],
        ElementwiseFusionInputView::broadcast_in_dim(vec![3, 2], vec![0])
    );
    assert_eq!(
        plan.input_views()[1],
        ElementwiseFusionInputView::broadcast_in_dim(vec![3, 2], vec![1])
    );
}

#[test]
fn segmenter_isolates_consecutive_broadcast_multiply_triples() {
    let program = ExecProgram {
        instructions: vec![
            broadcast(0, 4, vec![0]),
            broadcast(1, 5, vec![1]),
            multiply(4, 5, 6),
            broadcast(2, 7, vec![0]),
            broadcast(3, 8, vec![1]),
            multiply(7, 8, 9),
        ],
        input_slots: vec![0, 1, 2, 3],
        output_slots: vec![6, 9],
        n_slots: 10,
    };

    let segments = segment_exec_program(&program);

    assert_eq!(segments.len(), 2);
    for segment in segments {
        assert!(matches!(
            segment,
            Segment::Fused { instructions, .. } if instructions.len() == 3
        ));
    }
}

#[test]
fn segmenter_keeps_broadcast_multiply_chain_when_outputs_are_reused() {
    let program = ExecProgram {
        instructions: vec![
            broadcast_with_shape(0, 2, &[3, 2], vec![0]),
            broadcast_with_shape(1, 3, &[3, 2], vec![1]),
            multiply_with_shape(2, 3, 4, &[3, 2]),
            add_with_shape(4, 2, 5, &[3, 2]),
        ],
        input_slots: vec![0, 1],
        output_slots: vec![5],
        n_slots: 6,
    };

    let segments = segment_exec_program(&program);

    assert_eq!(segments.len(), 1);
    assert!(matches!(
        &segments[0],
        Segment::Fused { instructions, .. } if instructions.len() == 4
    ));
}

#[test]
fn segmenter_isolates_single_broadcast_multiply_pairs() {
    let program = ExecProgram {
        instructions: vec![
            broadcast(0, 4, vec![0]),
            multiply(4, 1, 5),
            broadcast(2, 6, vec![1]),
            multiply(3, 6, 7),
        ],
        input_slots: vec![0, 1, 2, 3],
        output_slots: vec![5, 7],
        n_slots: 8,
    };

    let segments = segment_exec_program(&program);

    assert_eq!(segments.len(), 2);
    for segment in segments {
        assert!(matches!(
            segment,
            Segment::Fused { instructions, .. } if instructions.len() == 2
        ));
    }
}

#[test]
fn single_broadcast_multiply_pair_handles_reused_broadcast_output() {
    let instructions = vec![broadcast(0, 4, vec![0]), multiply(4, 4, 5)];

    let pair = single_broadcast_multiply_pair(&instructions, &[5]);

    assert!(matches!(
        pair,
        Some(SingleBroadcastMultiplyPair::ReusedBroadcast { .. })
    ));
}

#[test]
fn segment_use_summary_tracks_program_outputs_and_future_inputs() {
    let program = ExecProgram {
        instructions: vec![
            broadcast(0, 2, vec![0]),
            multiply(2, 1, 3),
            multiply(3, 1, 4),
        ],
        input_slots: vec![0, 1],
        output_slots: vec![4],
        n_slots: 5,
    };

    let summary = SegmentUseSummary::new(&program);

    assert!(summary.is_program_output(4));
    assert!(!summary.is_program_output(3));
    assert!(!summary.is_used_at_or_after(0, 1));
    assert!(summary.is_used_at_or_after(1, 2));
    assert!(summary.is_used_at_or_after(3, 2));
    assert!(!summary.is_used_at_or_after(3, 3));
}

#[test]
fn segmented_eval_executes_single_broadcast_multiply_pairs() {
    let program = ExecProgram {
        instructions: vec![
            broadcast(0, 4, vec![0]),
            multiply(4, 1, 5),
            broadcast(2, 6, vec![1]),
            multiply(3, 6, 7),
        ],
        input_slots: vec![0, 1, 2, 3],
        output_slots: vec![5, 7],
        n_slots: 8,
    };
    let mut backend = CpuBackend::new();
    let inputs = vec![
        Tensor::from_vec_col_major(vec![2], vec![10.0_f64, 20.0]).unwrap(),
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
        Tensor::from_vec_col_major(vec![2], vec![5.0_f64, 7.0]).unwrap(),
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
    ];

    let outputs = eval_exec_segmented(&mut backend, &program, inputs).unwrap();

    assert_eq!(
        outputs[0].as_slice::<f64>().unwrap(),
        &[10.0, 40.0, 30.0, 80.0]
    );
    assert_eq!(
        outputs[1].as_slice::<f64>().unwrap(),
        &[5.0, 10.0, 21.0, 28.0]
    );
}

#[test]
fn segmented_eval_executes_reused_broadcast_multiply_pair() {
    let program = ExecProgram {
        instructions: vec![broadcast(0, 1, vec![0]), multiply(1, 1, 2)],
        input_slots: vec![0],
        output_slots: vec![2],
        n_slots: 3,
    };
    let mut backend = CpuBackend::new();
    let inputs = vec![Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap()];

    let outputs = eval_exec_segmented(&mut backend, &program, inputs).unwrap();

    assert_eq!(outputs[0].as_slice::<f64>().unwrap(), &[4.0, 9.0, 4.0, 9.0]);
}

#[test]
fn segmented_eval_executes_single_broadcast_multiply_with_implicit_scalar_other() {
    let program = ExecProgram {
        instructions: vec![
            broadcast_with_shape(1, 2, &[3], vec![]),
            multiply_with_shape(0, 2, 3, &[3]),
        ],
        input_slots: vec![0, 1],
        output_slots: vec![3],
        n_slots: 4,
    };
    let mut backend = CpuBackend::new();
    let inputs = vec![
        Tensor::from_vec_col_major(vec![], vec![10.0_f64]).unwrap(),
        Tensor::from_vec_col_major(vec![], vec![2.0_f64]).unwrap(),
    ];

    let outputs = eval_exec_segmented(&mut backend, &program, inputs).unwrap();

    assert_eq!(outputs[0].shape(), &[3]);
    assert_eq!(outputs[0].as_slice::<f64>().unwrap(), &[20.0, 20.0, 20.0]);
}
