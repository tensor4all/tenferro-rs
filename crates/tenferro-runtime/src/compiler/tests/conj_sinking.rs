use super::{
    conj_sinking, dim_shape, eliminate_dead_code, make_exec_instr_with_meta, make_exec_program,
    resolve_slot_redirect, DType, DimExpr, Error, ExecOp, ShapeExtent,
};
use crate::compiler::{collect_input_slot_meta, record_producer, slot_shape};
use std::sync::Arc;
use tenferro_ops::ext_op::{ExtensionAliasDeclaration, ExtensionEffectDeclaration, ExtensionOp};

#[derive(Clone, Debug)]
struct MixedOutputTypes;

impl ExtensionOp for MixedOutputTypes {
    fn family_id(&self) -> &'static str {
        "test.conj_sinking.mixed_output_types"
    }
    fn payload_hash(&self, _hasher: &mut dyn std::hash::Hasher) {}
    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().is::<Self>()
    }
    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn input_count(&self) -> usize {
        1
    }
    fn output_count(&self) -> usize {
        2
    }
    fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
        ExtensionEffectDeclaration::Declared(&[])
    }
    fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
        ExtensionAliasDeclaration::AllFresh
    }
    fn infer_output_meta(
        &self,
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<tenferro_ops::SymDim>)>> {
        let shape = ctx.input_shape(0)?.to_vec();
        Ok(vec![(DType::I64, shape.clone()), (DType::C64, shape)])
    }
}

#[test]
fn metadata_lookup_borrows_input_and_instruction_shapes() {
    let shapes = [dim_shape(&[2, 3])];
    let instr = make_exec_instr_with_meta(
        ExecOp::Transpose { perm: vec![1, 0] },
        vec![0],
        vec![1],
        DType::C64,
        vec![dim_shape(&[3, 2])],
    );
    let program = make_exec_program(vec![instr], vec![0], vec![1], 2);
    let mut metadata = collect_input_slot_meta(&program, &[DType::C64]).unwrap();
    record_producer(
        &mut metadata,
        &program.instructions[0],
        0,
        &program.instructions,
        &shapes,
    )
    .unwrap();
    let input_shape = slot_shape(0, &metadata, &program.instructions, &shapes).unwrap();
    assert_eq!(input_shape, dim_shape(&[2, 3]));
    assert_eq!(input_shape.as_ptr(), shapes[0].as_ptr());
    let output_shape = slot_shape(1, &metadata, &program.instructions, &shapes).unwrap();
    assert_eq!(output_shape, dim_shape(&[3, 2]));
    assert_eq!(
        output_shape.as_ptr(),
        program.instructions[0].output_shapes[0].as_ptr()
    );
}

#[test]
fn emitted_conjugates_use_the_selected_extension_output_dtype_and_extents() {
    let shape = dim_shape(&[6, 2]);
    let extents = vec![
        ShapeExtent::Unknown,
        ShapeExtent::UpperBound(DimExpr::Const(2)),
    ];
    let mut extension = make_exec_instr_with_meta(
        ExecOp::Extension(Arc::new(MixedOutputTypes)),
        vec![0],
        vec![1, 2],
        DType::I64,
        vec![shape.clone(), shape.clone()],
    );
    extension.output_extents[1] = extents.clone();
    let transpose = make_exec_instr_with_meta(
        ExecOp::Transpose { perm: vec![1, 0] },
        vec![2],
        vec![3],
        DType::C64,
        vec![dim_shape(&[2, 6])],
    );
    let conj = make_exec_instr_with_meta(
        ExecOp::Conj,
        vec![3],
        vec![4],
        DType::C64,
        vec![dim_shape(&[2, 6])],
    );
    let mut program = make_exec_program(vec![extension, transpose, conj], vec![0], vec![1, 4], 5);

    conj_sinking(&mut program, &[DType::C64], std::slice::from_ref(&shape)).unwrap();
    eliminate_dead_code(&mut program);

    assert_eq!(program.instructions.len(), 3);
    let conj = &program.instructions[1];
    assert!(matches!(conj.op, ExecOp::Conj));
    assert_eq!(conj.input_slots, [2]);
    assert_eq!(conj.dtype, DType::C64);
    assert_eq!(conj.output_shapes[0], shape);
    assert_eq!(conj.output_extents[0], extents);
    assert_eq!(program.instructions[0].output_slots, [1, 2]);
    assert_eq!(program.instructions[2].output_slots, [4]);
}

#[test]
fn redirects_preserve_roots_chains_and_out_of_range_targets() {
    assert_eq!(resolve_slot_redirect(0, &[]).unwrap(), 0);
    assert_eq!(resolve_slot_redirect(usize::MAX, &[0]).unwrap(), usize::MAX);
    assert_eq!(resolve_slot_redirect(0, &[0, 1]).unwrap(), 0);
    assert_eq!(resolve_slot_redirect(0, &[1, 2, 3, 3]).unwrap(), 3);
    assert_eq!(resolve_slot_redirect(0, &[1, 8]).unwrap(), 8);
    let length = 1024;
    let mut redirect: Vec<_> = (1..=length).collect();
    redirect[length - 1] = length - 1;
    assert_eq!(resolve_slot_redirect(0, &redirect).unwrap(), length - 1);
}

#[test]
fn redirects_reject_cycles_after_a_prefix() {
    for redirect in [vec![1, 2, 1], vec![1, 2, 3, 1], vec![1, 0, 2, 3]] {
        let err = resolve_slot_redirect(0, &redirect).unwrap_err();
        assert!(
            matches!(err, Error::Internal(message) if message.contains("redirect cycle") && message.contains("slot 0"))
        );
    }
}

#[test]
fn repeated_operands_share_one_emitted_conjugate() {
    let shape = dim_shape(&[2, 3]);
    let add = make_exec_instr_with_meta(
        ExecOp::Add,
        vec![0, 0],
        vec![1],
        DType::C64,
        vec![shape.clone()],
    );
    let conj = make_exec_instr_with_meta(
        ExecOp::Conj,
        vec![1],
        vec![2],
        DType::C64,
        vec![shape.clone()],
    );
    let mut program = make_exec_program(vec![add, conj], vec![0], vec![2], 3);

    conj_sinking(&mut program, &[DType::C64], &[shape]).unwrap();
    eliminate_dead_code(&mut program);

    assert_eq!(program.instructions.len(), 2);
    assert!(matches!(program.instructions[0].op, ExecOp::Conj));
    assert_eq!(program.instructions[0].input_slots, [0]);
    let conjugated = program.instructions[0].output_slots[0];
    assert!(matches!(program.instructions[1].op, ExecOp::Add));
    assert_eq!(
        program.instructions[1].input_slots,
        [conjugated, conjugated]
    );
    assert_eq!(program.instructions[1].output_slots, [2]);
}

#[test]
fn emitted_conjugates_preserve_input_metadata_for_supported_dtypes() {
    let shape = vec![
        DimExpr::add(DimExpr::Const(2), DimExpr::Const(3)),
        DimExpr::Const(4),
    ];
    for dtype in [DType::F32, DType::F64, DType::C32, DType::C64] {
        let mut program = make_exec_program(
            vec![
                make_exec_instr_with_meta(
                    ExecOp::Transpose { perm: vec![1, 0] },
                    vec![0],
                    vec![1],
                    dtype,
                    vec![dim_shape(&[4, 5])],
                ),
                make_exec_instr_with_meta(
                    ExecOp::Conj,
                    vec![1],
                    vec![2],
                    dtype,
                    vec![dim_shape(&[4, 5])],
                ),
            ],
            vec![0],
            vec![2],
            3,
        );
        conj_sinking(&mut program, &[dtype], std::slice::from_ref(&shape)).unwrap();
        eliminate_dead_code(&mut program);
        let conj = &program.instructions[0];
        assert!(matches!(conj.op, ExecOp::Conj));
        assert_eq!(conj.dtype, dtype);
        assert_eq!(conj.input_slots, [0]);
        assert_eq!(conj.output_shapes[0], shape);
        let expected: Vec<_> = shape.iter().cloned().map(ShapeExtent::exact).collect();
        assert_eq!(conj.output_extents[0], expected);
        assert_eq!(program.instructions[1].input_slots, conj.output_slots);
    }
}

#[test]
fn emitted_conjugates_preserve_instruction_extents_and_symbolic_shapes() {
    let shape = vec![
        DimExpr::InputDim {
            input_idx: 0,
            axis: 0,
        },
        DimExpr::Const(3),
    ];
    let extents = vec![
        ShapeExtent::UpperBound(DimExpr::Const(8)),
        ShapeExtent::Unknown,
    ];
    let mut exp = make_exec_instr_with_meta(
        ExecOp::Exp,
        vec![0],
        vec![1],
        DType::C64,
        vec![shape.clone()],
    );
    exp.output_extents = vec![extents.clone()].into_iter().collect();
    let transpose = make_exec_instr_with_meta(
        ExecOp::Transpose { perm: vec![1, 0] },
        vec![1],
        vec![2],
        DType::C64,
        vec![dim_shape(&[3, 8])],
    );
    let conj = make_exec_instr_with_meta(
        ExecOp::Conj,
        vec![2],
        vec![3],
        DType::C64,
        vec![dim_shape(&[3, 8])],
    );
    let mut program = make_exec_program(vec![exp, transpose, conj], vec![0], vec![3], 4);

    conj_sinking(&mut program, &[DType::C64], &[dim_shape(&[8, 3])]).unwrap();
    eliminate_dead_code(&mut program);

    assert_eq!(program.instructions.len(), 3);
    let conj = &program.instructions[1];
    assert!(matches!(conj.op, ExecOp::Conj));
    assert_eq!(conj.dtype, DType::C64);
    assert_eq!(conj.input_slots, [1]);
    assert_eq!(conj.output_shapes[0], shape);
    assert_eq!(conj.output_extents[0], extents);
    assert_eq!(program.instructions[2].input_slots, conj.output_slots);
}

#[test]
fn conjugation_helpers_do_not_copy_entire_slot_or_metadata_tables() {
    let source = include_str!("../mod.rs");
    let redirect = source
        .split_once("fn resolve_slot_redirect(")
        .unwrap()
        .1
        .split_once("fn conj_commuting_input_indices(")
        .unwrap()
        .0;
    assert!(
        !redirect.contains("vec![") && !redirect.contains("Vec::"),
        "redirect resolution must not allocate a slot-sized cycle table"
    );
    assert!(
        !source.contains("struct ProducerInfo"),
        "producer lookup must use emitted instruction indices instead of copying every operation"
    );
    let slot_meta = source
        .split_once("struct SlotMeta")
        .unwrap()
        .1
        .split_once("type AxisVec")
        .unwrap()
        .0;
    assert!(!slot_meta.contains("Vec<DimExpr>") && !slot_meta.contains("Vec<ShapeExtent"), "slot metadata must locate the owning instruction instead of deep-cloning every shape and extent");
    let dtypes = source
        .split_once("fn instruction_output_dtypes(")
        .unwrap()
        .1
        .split_once("struct ConjSinkingState")
        .unwrap()
        .0;
    assert!(
        !dtypes.contains(".clone()") && !dtypes.contains(".to_vec()"),
        "extension dtype inference must borrow slot shapes instead of cloning expression trees"
    );
}
