//! Unit tests for linalg AD rules.

use chainrules_core::PrimitiveOp;
use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, OpMode, ValRef};
use tenferro_tensor::DType;

use crate::ad::context::ShapeGuardContext;
use crate::input_key::TensorInputKey;
use crate::std_tensor_op::StdTensorOp;
use crate::TensorMeta;

fn tensor_input(id: u64) -> TensorInputKey {
    TensorInputKey::User { id }
}

fn input_key(id: u64) -> GlobalValKey<StdTensorOp> {
    GlobalValKey::Input(tensor_input(id))
}

fn meta(shape: &[usize]) -> TensorMeta {
    TensorMeta::exact(DType::F64, shape.iter().copied().map(Into::into).collect())
}

#[test]
fn transpose_triangular_solve_returns_matrix_cotangent_when_matrix_is_active() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let cot = builder.add_input(tensor_input(1));
    let a_key = input_key(2);
    let b_key = input_key(3);
    ctx.insert_metadata(a_key.clone(), meta(&[2, 2]));
    ctx.insert_metadata(b_key.clone(), meta(&[2, 1]));
    let inputs = vec![ValRef::External(a_key), ValRef::External(b_key)];
    let op = StdTensorOp::TriangularSolve {
        left_side: true,
        lower: true,
        transpose_a: false,
        unit_diagonal: false,
    };

    let result = op.transpose_rule(
        &mut builder,
        &[Some(cot)],
        &inputs,
        &OpMode::Linear {
            active_mask: vec![true, false],
        },
        &mut ctx,
    );

    assert!(
        result[0].is_some(),
        "active triangular matrix input must receive a cotangent"
    );
    assert_eq!(result[1], None, "inactive RHS cotangent must stay None");
}

#[test]
fn transpose_full_piv_lu_solve_returns_matrix_cotangent_when_matrix_is_active() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let cot = builder.add_input(tensor_input(4));
    let a_key = input_key(5);
    let b_key = input_key(6);
    ctx.insert_metadata(a_key.clone(), meta(&[2, 2]));
    ctx.insert_metadata(b_key.clone(), meta(&[2, 1]));
    let inputs = vec![ValRef::External(a_key), ValRef::External(b_key)];
    let op = StdTensorOp::FullPivLuSolve { transpose_a: false };

    let result = op.transpose_rule(
        &mut builder,
        &[Some(cot)],
        &inputs,
        &OpMode::Linear {
            active_mask: vec![true, false],
        },
        &mut ctx,
    );

    assert!(
        result[0].is_some(),
        "active LU matrix input must receive a cotangent"
    );
    assert_eq!(result[1], None, "inactive RHS cotangent must stay None");
}
