use super::*;
use crate::exec::{ExecInstruction, ExecOp, ExecProgram};
use tenferro_cpu::CpuBackend;
use tenferro_ops::{dim_expr::DimExpr, ShapeExtent};
use tenferro_tensor::{DType, Tensor};

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
