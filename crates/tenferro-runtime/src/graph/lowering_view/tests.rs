use std::any::Any;
use std::sync::Arc;

use super::*;
use crate::exec::{ExecInstruction, ExecOp, ExecProgram};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_ops::{ShapeExtent, SymDim};
use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
    Tensor,
};

#[derive(Clone, Debug)]
struct DummyExtension;

impl ExtensionOp for DummyExtension {
    fn family_id(&self) -> &'static str {
        "test.lowering_view.v1"
    }

    fn payload_hash(&self, _hasher: &mut dyn std::hash::Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>().is_some()
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
        input_dtypes: &[DType],
        input_shapes: &[&[SymDim]],
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        Ok(vec![(input_dtypes[0], input_shapes[0].to_vec())])
    }

    fn eager_execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
        Ok(vec![inputs[0].clone()])
    }
}

fn shape(shape: &[usize]) -> Vec<DimExpr> {
    DimExpr::from_concrete(shape)
}

fn dim(value: usize) -> DimExpr {
    shape(&[value]).into_iter().next().unwrap()
}

fn exact_extents(shape: &[DimExpr]) -> Vec<ShapeExtent<DimExpr>> {
    shape.iter().cloned().map(ShapeExtent::exact).collect()
}

fn instruction_with_extents(
    op: ExecOp,
    input_slots: Vec<usize>,
    output_slots: Vec<usize>,
    dtype: DType,
    output_extents: Vec<Vec<ShapeExtent<DimExpr>>>,
) -> ExecInstruction {
    let output_shapes = output_extents
        .iter()
        .map(|extents| {
            extents
                .iter()
                .map(|extent| match extent {
                    ShapeExtent::Exact(dim) | ShapeExtent::UpperBound(dim) => dim.clone(),
                    ShapeExtent::Unknown => DimExpr::from_concrete(&[1]).remove(0),
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    ExecInstruction {
        op,
        input_slots,
        output_slots,
        dtype,
        output_shapes: output_shapes.into(),
        output_extents: output_extents.into(),
        last_use: Vec::new(),
    }
}

fn instruction(op: ExecOp, input_slots: Vec<usize>, output_slots: Vec<usize>) -> ExecInstruction {
    instruction_with_extents(
        op,
        input_slots,
        output_slots,
        DType::F64,
        vec![exact_extents(&shape(&[2]))],
    )
}

fn dot_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![1],
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

#[test]
fn lowering_view_debug_and_instruction_accessors_are_stable() {
    let add = instruction(ExecOp::Add, vec![0, 0], vec![1]);
    let program = ExecProgram {
        instructions: vec![add],
        input_slots: vec![0],
        output_slots: vec![1],
        n_slots: 2,
    };
    let view = GraphProgramLoweringView::new(&program);

    assert_eq!(view.slot_count(), 2);
    assert_eq!(view.input_slots(), &[0]);
    assert_eq!(view.output_slots(), &[1]);
    assert_eq!(view.instructions().len(), 1);
    assert!(format!("{view:?}").contains("instruction_count"));

    let inst = view.instructions().next().unwrap();
    assert_eq!(inst.op_name(), "Add");
    assert_eq!(inst.input_slots(), &[0, 0]);
    assert_eq!(inst.output_slots(), &[1]);
    assert_eq!(inst.dtype(), DType::F64);
    assert_eq!(inst.static_output_shape(0, &[&[2], &[2]]).unwrap(), vec![2]);
    assert!(format!("{inst:?}").contains("GraphInstructionView"));

    let err = inst.static_output_shape(1, &[&[2], &[2]]).unwrap_err();
    assert_eq!(
        err,
        GraphProgramLoweringShapeError::MissingOutput {
            op: "Add",
            output_index: 1
        }
    );
}

#[test]
fn static_output_shape_reports_non_static_extents() {
    let upper = instruction_with_extents(
        ExecOp::DynamicTruncate { axis: 0 },
        vec![0, 1],
        vec![2],
        DType::F64,
        vec![vec![ShapeExtent::UpperBound(dim(4))]],
    );
    let upper_err = GraphInstructionView::new(&upper)
        .static_output_shape(0, &[&[4], &[]])
        .unwrap_err();
    assert_eq!(
        upper_err,
        GraphProgramLoweringShapeError::NonStatic {
            op: "DynamicTruncate",
            output_index: 0,
            axis: 0,
            kind: "an upper bound"
        }
    );

    let unknown = instruction_with_extents(
        ExecOp::ShapeOf { axis: 0 },
        vec![0],
        vec![1],
        DType::I64,
        vec![vec![ShapeExtent::Unknown]],
    );
    let unknown_err = GraphInstructionView::new(&unknown)
        .static_output_shape(0, &[&[4]])
        .unwrap_err();
    assert_eq!(
        unknown_err,
        GraphProgramLoweringShapeError::NonStatic {
            op: "ShapeOf",
            output_index: 0,
            axis: 0,
            kind: "unknown"
        }
    );
}

#[test]
fn graph_op_view_names_and_debug_cover_lowering_variants() {
    let bytes = 1.0_f64.to_le_bytes();
    let dims = vec![1, 0];
    let axes = vec![0];
    let config = dot_config();
    let extension = DummyExtension;

    let cases = [
        (
            GraphOpView::Constant {
                dtype: DType::F64,
                bytes: &bytes,
            },
            "Constant",
            "byte_len",
        ),
        (GraphOpView::Add, "Add", "Add"),
        (GraphOpView::Multiply, "Multiply", "Multiply"),
        (GraphOpView::Negate, "Negate", "Negate"),
        (GraphOpView::Divide, "Divide", "Divide"),
        (GraphOpView::Abs, "Abs", "Abs"),
        (GraphOpView::Exp, "Exp", "Exp"),
        (GraphOpView::Log, "Log", "Log"),
        (GraphOpView::Sin, "Sin", "Sin"),
        (GraphOpView::Cos, "Cos", "Cos"),
        (GraphOpView::Tanh, "Tanh", "Tanh"),
        (GraphOpView::Sqrt, "Sqrt", "Sqrt"),
        (GraphOpView::Rsqrt, "Rsqrt", "Rsqrt"),
        (GraphOpView::Pow, "Pow", "Pow"),
        (GraphOpView::Expm1, "Expm1", "Expm1"),
        (GraphOpView::Log1p, "Log1p", "Log1p"),
        (GraphOpView::Convert { to: DType::F32 }, "Convert", "to"),
        (GraphOpView::Reshape, "Reshape", "Reshape"),
        (
            GraphOpView::BroadcastInDim { dims: &dims },
            "BroadcastInDim",
            "dims",
        ),
        (GraphOpView::Transpose { perm: &dims }, "Transpose", "perm"),
        (GraphOpView::ReduceSum { axes: &axes }, "ReduceSum", "axes"),
        (
            GraphOpView::DotGeneral { config: &config },
            "DotGeneral",
            "config",
        ),
        (
            GraphOpView::Extension { op: &extension },
            "Extension",
            "test.lowering_view.v1",
        ),
        (
            GraphOpView::Unsupported { name: "Exp" },
            "Exp",
            "Unsupported",
        ),
    ];

    for (op, name, debug_fragment) in cases {
        assert_eq!(op.name(), name);
        assert!(
            format!("{op:?}").contains(debug_fragment),
            "{op:?} did not contain {debug_fragment}"
        );
    }
}

#[test]
fn phase_one_elementwise_exec_ops_have_lowering_variants() {
    let cases = [
        (ExecOp::Divide, "Divide"),
        (ExecOp::Abs, "Abs"),
        (ExecOp::Exp, "Exp"),
        (ExecOp::Log, "Log"),
        (ExecOp::Sin, "Sin"),
        (ExecOp::Cos, "Cos"),
        (ExecOp::Tanh, "Tanh"),
        (ExecOp::Sqrt, "Sqrt"),
        (ExecOp::Rsqrt, "Rsqrt"),
        (ExecOp::Pow, "Pow"),
        (ExecOp::Expm1, "Expm1"),
        (ExecOp::Log1p, "Log1p"),
    ];

    for (op, name) in cases {
        let inst = instruction(op, vec![0, 0], vec![1]);
        let op_view = GraphInstructionView::new(&inst).op();
        assert_eq!(op_view.name(), name);
        assert!(
            !matches!(op_view, GraphOpView::Unsupported { .. }),
            "ExecOp::{name} should be exposed as a lowerable GraphOpView variant"
        );
    }
}

#[test]
fn exec_op_name_covers_supported_and_unsupported_variants() {
    let cases = vec![
        (ExecOp::Transpose { perm: vec![0] }, "Transpose"),
        (ExecOp::Reshape { shape: shape(&[2]) }, "Reshape"),
        (
            ExecOp::BroadcastInDim {
                shape: shape(&[2]),
                dims: vec![0],
            },
            "BroadcastInDim",
        ),
        (ExecOp::Convert { to: DType::F32 }, "Convert"),
        (
            ExecOp::Constant {
                dtype: DType::F64,
                bytes: 1.0_f64.to_le_bytes().to_vec(),
            },
            "Constant",
        ),
        (ExecOp::DotGeneral(dot_config()), "DotGeneral"),
        (
            ExecOp::DotGeneralWithConj {
                config: dot_config(),
                lhs_conj: true,
                rhs_conj: false,
            },
            "DotGeneralWithConj",
        ),
        (ExecOp::ReduceSum { axes: vec![0] }, "ReduceSum"),
        (
            ExecOp::ExtractDiag {
                axis_a: 0,
                axis_b: 1,
            },
            "ExtractDiag",
        ),
        (
            ExecOp::EmbedDiag {
                axis_a: 0,
                axis_b: 1,
            },
            "EmbedDiag",
        ),
        (ExecOp::Tril { k: 0 }, "Tril"),
        (ExecOp::Triu { k: 0 }, "Triu"),
        (ExecOp::Add, "Add"),
        (ExecOp::Multiply, "Multiply"),
        (ExecOp::Negate, "Negate"),
        (ExecOp::Conj, "Conj"),
        (ExecOp::Divide, "Divide"),
        (ExecOp::Abs, "Abs"),
        (ExecOp::Sign, "Sign"),
        (ExecOp::Maximum, "Maximum"),
        (ExecOp::Minimum, "Minimum"),
        (ExecOp::Compare(CompareDir::Eq), "Compare"),
        (ExecOp::Select, "Select"),
        (ExecOp::Clamp, "Clamp"),
        (ExecOp::Exp, "Exp"),
        (ExecOp::Log, "Log"),
        (ExecOp::Sin, "Sin"),
        (ExecOp::Cos, "Cos"),
        (ExecOp::Tanh, "Tanh"),
        (ExecOp::Sqrt, "Sqrt"),
        (ExecOp::Rsqrt, "Rsqrt"),
        (ExecOp::Pow, "Pow"),
        (ExecOp::Expm1, "Expm1"),
        (ExecOp::Log1p, "Log1p"),
        (ExecOp::Gather(gather_config()), "Gather"),
        (
            ExecOp::GatherDynamicSliceSizes {
                offset_dims: vec![],
                collapsed_slice_dims: vec![0],
                start_index_map: vec![0],
                index_vector_dim: 1,
                slice_sizes: shape(&[1]),
            },
            "GatherDynamicSliceSizes",
        ),
        (ExecOp::Scatter(scatter_config()), "Scatter"),
        (ExecOp::Slice(slice_config()), "Slice"),
        (
            ExecOp::DynamicSlice {
                slice_sizes: vec![1],
            },
            "DynamicSlice",
        ),
        (ExecOp::DynamicUpdateSlice, "DynamicUpdateSlice"),
        (ExecOp::Pad(pad_config()), "Pad"),
        (ExecOp::Concatenate { axis: 0 }, "Concatenate"),
        (ExecOp::Reverse { axes: vec![0] }, "Reverse"),
        (ExecOp::ShapeOf { axis: 0 }, "ShapeOf"),
        (ExecOp::DynamicTruncate { axis: 0 }, "DynamicTruncate"),
        (ExecOp::PadToMatch { axis: 0 }, "PadToMatch"),
        (ExecOp::ReduceProd { axes: vec![0] }, "ReduceProd"),
        (ExecOp::ReduceMax { axes: vec![0] }, "ReduceMax"),
        (ExecOp::ReduceMin { axes: vec![0] }, "ReduceMin"),
        (ExecOp::Extension(Arc::new(DummyExtension)), "Extension"),
    ];

    for (op, name) in cases {
        assert_eq!(exec_op_name(&op), name);
    }
}
