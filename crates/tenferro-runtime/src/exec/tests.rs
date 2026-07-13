use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use tenferro_core_ops::{all_primitive_descriptors, PrimitiveOpKind};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_ops::{ShapeExtent, SymDim};
use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
    Tensor,
};

use super::dispatch::{
    backend_dispatch_entry, ffi_dispatch_entry, host_dispatch_entry, FfiDispatchKey,
    HostDispatchKey, BACKEND_DISPATCH_TABLE,
};
use super::{
    collect_outputs_from, constant_tensor, get, initialize_exec_slots_in,
    tensor_value_for_lazy_view, validate_exec_program, ExecInstruction, ExecOp, ExecProgram,
    ExecSlot,
};
use tenferro_cpu::CpuBackend;

#[test]
fn exec_op_maps_to_catalog_kind() {
    assert_eq!(ExecOp::Add.primitive_kind(), Some(PrimitiveOpKind::Add));
    assert_eq!(
        ExecOp::Multiply.primitive_kind(),
        Some(PrimitiveOpKind::Mul)
    );
    assert_eq!(ExecOp::Negate.primitive_kind(), Some(PrimitiveOpKind::Neg));
    assert_eq!(ExecOp::Divide.primitive_kind(), Some(PrimitiveOpKind::Div));
    assert_eq!(
        ExecOp::ShapeOf { axis: 0 }.primitive_kind(),
        Some(PrimitiveOpKind::ShapeOf)
    );
    assert_eq!(
        ExecOp::Extension(Arc::new(TestExtension)).primitive_kind(),
        None
    );
}

#[test]
fn every_catalog_descriptor_has_an_exec_op_variant_with_matching_arity() {
    for descriptor in all_primitive_descriptors() {
        let op = ExecOp::sample_from_kind(descriptor.kind);
        assert_eq!(
            op.primitive_kind(),
            Some(descriptor.kind),
            "ExecOp variant for {:?} disagrees with descriptor",
            descriptor.kind
        );
        assert_eq!(
            op.input_arity_bounds(),
            Some((descriptor.min_inputs, descriptor.max_inputs)),
            "ExecOp arity for {:?} disagrees with descriptor",
            descriptor.kind
        );
    }
}

#[test]
fn backend_dispatch_table_covers_all_backend_exec_ops() {
    let cases = backend_dispatch_cases();
    assert_eq!(cases.len(), BACKEND_DISPATCH_TABLE.len());

    for (op, expected) in cases {
        assert_eq!(op.primitive_kind(), Some(expected), "{op:?}");
        let entry = backend_dispatch_entry(&op).unwrap_or_else(|| {
            panic!("missing backend dispatch table entry for {expected:?}: {op:?}")
        });
        assert_eq!(entry.key, expected);
    }
}

#[test]
fn backend_dispatch_table_excludes_host_and_ffi_exec_ops() {
    for op in non_backend_dispatch_cases() {
        assert!(backend_dispatch_entry(&op).is_none(), "{op:?}");
    }
}

#[test]
fn ffi_dispatch_table_covers_dot_and_extension_exec_ops() {
    let cases = ffi_dispatch_cases();
    assert_eq!(cases.len(), FfiDispatchKey::COUNT);

    for (op, expected) in cases {
        assert_eq!(FfiDispatchKey::for_op(&op), Some(expected), "{op:?}");
        let entry = ffi_dispatch_entry::<CpuBackend>(&op)
            .unwrap_or_else(|| panic!("missing FFI dispatch table entry for {expected:?}: {op:?}"));
        assert_eq!(entry.key, expected);
    }
}

#[test]
fn ffi_dispatch_table_excludes_host_and_backend_exec_ops() {
    for op in non_ffi_dispatch_cases() {
        assert_eq!(FfiDispatchKey::for_op(&op), None, "{op:?}");
        assert!(ffi_dispatch_entry::<CpuBackend>(&op).is_none(), "{op:?}");
    }
}

#[test]
fn host_dispatch_table_covers_host_exec_ops() {
    let cases = host_dispatch_cases();
    assert_eq!(cases.len(), HostDispatchKey::COUNT);

    for (op, expected) in cases {
        assert_eq!(HostDispatchKey::for_op(&op), Some(expected), "{op:?}");
        let entry = host_dispatch_entry::<CpuBackend>(&op).unwrap_or_else(|| {
            panic!("missing host dispatch table entry for {expected:?}: {op:?}")
        });
        assert_eq!(entry.key, expected);
    }
}

#[test]
fn host_dispatch_table_excludes_backend_and_ffi_exec_ops() {
    for op in non_host_dispatch_cases() {
        assert_eq!(HostDispatchKey::for_op(&op), None, "{op:?}");
        assert!(host_dispatch_entry::<CpuBackend>(&op).is_none(), "{op:?}");
    }
}

#[test]
fn exec_instruction_single_output_metadata_stays_inline() {
    let instr = ExecInstruction {
        op: ExecOp::Negate,
        input_slots: vec![0],
        output_slots: vec![1],
        dtype: DType::F64,
        output_shapes: vec![vec![DimExpr::Const(2)]].into(),
        output_extents: vec![vec![ShapeExtent::exact(DimExpr::Const(2))]].into(),
        last_use: vec![true],
    };

    assert!(!instr.output_shapes.spilled());
    assert!(!instr.output_extents.spilled());
}

#[test]
fn lazy_view_input_conversion_shares_live_owned_tensor() {
    let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let mut slots = vec![Some(ExecSlot::Owned(tensor))];

    let value = tensor_value_for_lazy_view(&mut slots, 0, false).unwrap();

    let output = value.as_tensor_arc().unwrap();
    let stored = match slots[0].as_ref().unwrap() {
        ExecSlot::Value(value) => value.as_tensor_arc().unwrap(),
        _ => panic!("expected promoted tensor value"),
    };
    assert!(Arc::ptr_eq(output, stored));
}

#[test]
fn exec_accessors_reject_bad_input_slot_index_without_panicking() {
    let slots = [Some(ExecSlot::Owned(
        Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
    ))];

    let err = get(&slots, &[], 0).unwrap_err();

    let message = err.to_string();
    assert!(message.contains("invalid compiled graph"), "{message}");
    assert!(message.contains("input index 0"), "{message}");
}

#[test]
fn exec_accessors_reject_out_of_range_slot_without_panicking() {
    let slots = [Some(ExecSlot::Owned(
        Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
    ))];

    let err = get(&slots, &[4], 0).unwrap_err();

    let message = err.to_string();
    assert!(message.contains("slot 4"), "{message}");
    assert!(message.contains("out of range"), "{message}");
}

#[test]
fn initialize_exec_slots_rejects_input_count_mismatch_without_panicking() {
    let program = empty_program(1);
    let mut slots = Vec::new();
    let inputs = vec![
        ExecSlot::Owned(Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap()),
        ExecSlot::Owned(Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap()),
    ];

    let err = initialize_exec_slots_in(&program, inputs, &mut slots).unwrap_err();

    let message = err.to_string();
    assert!(message.contains("expected 1 inputs"), "{message}");
    assert!(message.contains("got 2"), "{message}");
}

#[test]
fn collect_outputs_rejects_out_of_range_slot_without_panicking() {
    let mut program = empty_program(1);
    program.output_slots = vec![3];
    let mut slots = vec![Some(ExecSlot::Owned(
        Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
    ))];

    let err = collect_outputs_from(&program, &mut slots).unwrap_err();

    let message = err.to_string();
    assert!(message.contains("output slot 3"), "{message}");
    assert!(message.contains("out of range"), "{message}");
}

#[test]
fn validate_exec_program_rejects_empty_instruction_outputs() {
    let mut program = empty_program(1);
    program.instructions.push(ExecInstruction {
        op: ExecOp::Negate,
        input_slots: vec![0],
        output_slots: vec![],
        dtype: DType::F64,
        output_shapes: Vec::new().into(),
        output_extents: Vec::new().into(),
        last_use: vec![true],
    });

    let err = validate_exec_program(&program, "test").unwrap_err();

    let message = err.to_string();
    assert!(message.contains("instruction 0"), "{message}");
    assert!(message.contains("no output slots"), "{message}");
}

#[test]
fn validate_exec_program_rejects_terminal_value_instruction_input_arity() {
    let mut program = empty_program(0);
    program.n_slots = 1;
    program.output_slots = vec![0];
    program.instructions.push(ExecInstruction {
        op: ExecOp::Transpose { perm: vec![0] },
        input_slots: vec![],
        output_slots: vec![0],
        dtype: DType::F64,
        output_shapes: vec![vec![DimExpr::Const(1)]].into(),
        output_extents: vec![vec![ShapeExtent::exact(DimExpr::Const(1))]].into(),
        last_use: vec![],
    });

    let err = validate_exec_program(&program, "test").unwrap_err();

    let message = err.to_string();
    assert!(message.contains("instruction 0"), "{message}");
    assert!(message.contains("input slots"), "{message}");
}

#[test]
fn validate_exec_program_accepts_symbolic_shape_reference_inputs() {
    let mut program = empty_program(2);
    program.n_slots = 3;
    program.output_slots = vec![2];
    program.instructions.push(ExecInstruction {
        op: ExecOp::BroadcastInDim {
            shape: vec![DimExpr::InputDim {
                input_idx: 1,
                axis: 0,
            }],
            dims: vec![],
        },
        input_slots: vec![0, 1],
        output_slots: vec![2],
        dtype: DType::F64,
        output_shapes: vec![vec![DimExpr::InputDim {
            input_idx: 1,
            axis: 0,
        }]]
        .into(),
        output_extents: vec![vec![ShapeExtent::exact(DimExpr::InputDim {
            input_idx: 1,
            axis: 0,
        })]]
        .into(),
        last_use: vec![true],
    });

    validate_exec_program(&program, "test").unwrap();
}

#[test]
fn constant_tensor_rejects_payload_length_mismatch_without_panicking() {
    let err = constant_tensor(DType::F64, &[0; 4]).unwrap_err();

    let message = err.to_string();
    assert!(
        message.contains("constant F64 expected 8 bytes"),
        "{message}"
    );
    assert!(message.contains("got 4"), "{message}");
}

fn empty_program(input_count: usize) -> ExecProgram {
    ExecProgram {
        instructions: Vec::new(),
        input_slots: (0..input_count).collect(),
        output_slots: Vec::new(),
        n_slots: input_count,
        shape_guards: Vec::new(),
    }
}

fn backend_dispatch_cases() -> Vec<(ExecOp, PrimitiveOpKind)> {
    vec![
        (
            ExecOp::Transpose { perm: vec![0] },
            PrimitiveOpKind::Transpose,
        ),
        (
            ExecOp::Reshape {
                shape: vec![DimExpr::Const(1)],
            },
            PrimitiveOpKind::Reshape,
        ),
        (
            ExecOp::BroadcastInDim {
                shape: vec![DimExpr::Const(1)],
                dims: vec![0],
            },
            PrimitiveOpKind::BroadcastInDim,
        ),
        (ExecOp::Convert { to: DType::F64 }, PrimitiveOpKind::Convert),
        (
            ExecOp::ReduceSum { axes: vec![0] },
            PrimitiveOpKind::ReduceSum,
        ),
        (
            ExecOp::ExtractDiag {
                axis_a: 0,
                axis_b: 1,
            },
            PrimitiveOpKind::ExtractDiag,
        ),
        (
            ExecOp::EmbedDiag {
                axis_a: 0,
                axis_b: 1,
            },
            PrimitiveOpKind::EmbedDiag,
        ),
        (ExecOp::Tril { k: 0 }, PrimitiveOpKind::Tril),
        (ExecOp::Triu { k: 0 }, PrimitiveOpKind::Triu),
        (ExecOp::Add, PrimitiveOpKind::Add),
        (ExecOp::Subtract, PrimitiveOpKind::Sub),
        (ExecOp::Multiply, PrimitiveOpKind::Mul),
        (ExecOp::Negate, PrimitiveOpKind::Neg),
        (ExecOp::Conj, PrimitiveOpKind::Conj),
        (ExecOp::Divide, PrimitiveOpKind::Div),
        (ExecOp::Remainder, PrimitiveOpKind::Rem),
        (ExecOp::Abs, PrimitiveOpKind::Abs),
        (ExecOp::Sign, PrimitiveOpKind::Sign),
        (ExecOp::Maximum, PrimitiveOpKind::Maximum),
        (ExecOp::Minimum, PrimitiveOpKind::Minimum),
        (ExecOp::Compare(CompareDir::Eq), PrimitiveOpKind::Compare),
        (ExecOp::Select, PrimitiveOpKind::Select),
        (ExecOp::Clamp, PrimitiveOpKind::Clamp),
        (ExecOp::Exp, PrimitiveOpKind::Exp),
        (ExecOp::Log, PrimitiveOpKind::Log),
        (ExecOp::Sin, PrimitiveOpKind::Sin),
        (ExecOp::Cos, PrimitiveOpKind::Cos),
        (ExecOp::Tanh, PrimitiveOpKind::Tanh),
        (ExecOp::Sqrt, PrimitiveOpKind::Sqrt),
        (ExecOp::Rsqrt, PrimitiveOpKind::Rsqrt),
        (ExecOp::Pow, PrimitiveOpKind::Pow),
        (ExecOp::Expm1, PrimitiveOpKind::Expm1),
        (ExecOp::Log1p, PrimitiveOpKind::Log1p),
        (ExecOp::Gather(gather_config()), PrimitiveOpKind::Gather),
        (
            ExecOp::GatherDynamicSliceSizes {
                offset_dims: vec![],
                collapsed_slice_dims: vec![0],
                start_index_map: vec![0],
                index_vector_dim: 1,
                slice_sizes: vec![DimExpr::Const(1)],
            },
            PrimitiveOpKind::GatherDynamicSliceSizes,
        ),
        (ExecOp::Scatter(scatter_config()), PrimitiveOpKind::Scatter),
        (ExecOp::Slice(slice_config()), PrimitiveOpKind::Slice),
        (
            ExecOp::DynamicSlice {
                slice_sizes: vec![1],
            },
            PrimitiveOpKind::DynamicSlice,
        ),
        (
            ExecOp::DynamicUpdateSlice,
            PrimitiveOpKind::DynamicUpdateSlice,
        ),
        (ExecOp::Pad(pad_config()), PrimitiveOpKind::Pad),
        (
            ExecOp::Concatenate { axis: 0 },
            PrimitiveOpKind::Concatenate,
        ),
        (ExecOp::Reverse { axes: vec![0] }, PrimitiveOpKind::Reverse),
        (
            ExecOp::ReduceProd { axes: vec![0] },
            PrimitiveOpKind::ReduceProd,
        ),
        (
            ExecOp::ReduceMax { axes: vec![0] },
            PrimitiveOpKind::ReduceMax,
        ),
        (
            ExecOp::ReduceMin { axes: vec![0] },
            PrimitiveOpKind::ReduceMin,
        ),
    ]
}

fn ffi_dispatch_cases() -> Vec<(ExecOp, FfiDispatchKey)> {
    vec![
        (ExecOp::DotGeneral(dot_config()), FfiDispatchKey::DotGeneral),
        (
            ExecOp::DotGeneralWithConj {
                config: dot_config(),
                lhs_conj: true,
                rhs_conj: false,
            },
            FfiDispatchKey::DotGeneralWithConj,
        ),
        (
            ExecOp::Extension(Arc::new(TestExtension)),
            FfiDispatchKey::Extension,
        ),
    ]
}

fn host_dispatch_cases() -> Vec<(ExecOp, HostDispatchKey)> {
    vec![
        (ExecOp::ShapeOf { axis: 0 }, HostDispatchKey::ShapeOf),
        (
            ExecOp::DynamicTruncate { axis: 0 },
            HostDispatchKey::DynamicTruncate,
        ),
        (ExecOp::PadToMatch { axis: 0 }, HostDispatchKey::PadToMatch),
        (
            ExecOp::Constant {
                dtype: DType::F64,
                bytes: 1.0_f64.to_le_bytes().to_vec(),
            },
            HostDispatchKey::Constant,
        ),
    ]
}

fn non_backend_dispatch_cases() -> Vec<ExecOp> {
    vec![
        ExecOp::ShapeOf { axis: 0 },
        ExecOp::DynamicTruncate { axis: 0 },
        ExecOp::PadToMatch { axis: 0 },
        ExecOp::Constant {
            dtype: DType::F64,
            bytes: 1.0_f64.to_le_bytes().to_vec(),
        },
        ExecOp::DotGeneral(dot_config()),
        ExecOp::DotGeneralWithConj {
            config: dot_config(),
            lhs_conj: true,
            rhs_conj: false,
        },
        ExecOp::Extension(Arc::new(TestExtension)),
    ]
}

fn non_ffi_dispatch_cases() -> Vec<ExecOp> {
    let mut cases = backend_dispatch_cases()
        .into_iter()
        .map(|(op, _)| op)
        .collect::<Vec<_>>();
    cases.extend([
        ExecOp::ShapeOf { axis: 0 },
        ExecOp::DynamicTruncate { axis: 0 },
        ExecOp::PadToMatch { axis: 0 },
        ExecOp::Constant {
            dtype: DType::F64,
            bytes: 1.0_f64.to_le_bytes().to_vec(),
        },
    ]);
    cases
}

fn non_host_dispatch_cases() -> Vec<ExecOp> {
    let mut cases = backend_dispatch_cases()
        .into_iter()
        .map(|(op, _)| op)
        .collect::<Vec<_>>();
    cases.extend(ffi_dispatch_cases().into_iter().map(|(op, _)| op));
    cases
}

fn dot_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![0],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    }
}

fn gather_config() -> GatherConfig {
    GatherConfig {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1],
    }
}

fn scatter_config() -> ScatterConfig {
    ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    }
}

fn slice_config() -> SliceConfig {
    SliceConfig {
        starts: vec![0],
        limits: vec![1],
        strides: vec![1],
    }
}

fn pad_config() -> PadConfig {
    PadConfig {
        edge_padding_low: vec![0],
        edge_padding_high: vec![1],
        interior_padding: vec![0],
    }
}

#[derive(Clone, Debug)]
struct TestExtension;

impl ExtensionOp for TestExtension {
    fn family_id(&self) -> &'static str {
        "runtime.dispatch-test.v1"
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_u8(0);
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().is::<Self>()
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        Ok(vec![(ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec())])
    }
}
