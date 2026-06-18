use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use lru::LruCache;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_ops::{ShapeExtent, SymDim};
use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
    Tensor,
};

use super::*;

#[test]
fn cache_key_uses_structural_program_key() {
    let program = unary_program(ExecOp::Reshape {
        shape: vec![DimExpr::Const(4)],
    });

    let key = compute_cache_key(&program);

    assert_eq!(key.fingerprint.input_slots, vec![0]);
    assert_eq!(key.fingerprint.output_slots, vec![1]);
    assert_eq!(key.fingerprint.n_slots, 2);
    assert_eq!(key.fingerprint.instructions.len(), 1);
    assert_eq!(
        key.fingerprint.instructions[0].op,
        ExecOpKey::Reshape {
            shape: vec![DimExpr::Const(4)]
        }
    );
}

#[test]
fn extension_payload_equality_still_breaks_payload_hash_collisions() {
    let left = extension_program(1);
    let right = extension_program(2);

    let left_key = compute_cache_key(&left);
    let right_key = compute_cache_key(&right);

    assert_eq!(left_key.fingerprint, right_key.fingerprint);
    assert_ne!(left_key, right_key);
}

#[test]
fn compile_cache_stats_include_structural_key_payloads() {
    let program = unary_program(ExecOp::Constant {
        dtype: DType::F64,
        bytes: vec![7; 256],
    });
    let key = compute_cache_key(&program);
    let mut cache = LruCache::unbounded();

    cache.put(key, program);

    let stats = compile_cache_stats(&cache);
    assert_eq!(stats.entries, 1);
    assert!(stats.retained_bytes >= 256);
}

#[test]
fn cache_key_and_stats_cover_exec_op_payload_variants() {
    let ops = vec![
        ExecOp::Transpose { perm: vec![1, 0] },
        ExecOp::Reshape {
            shape: vec![DimExpr::Const(2), DimExpr::Const(2)],
        },
        ExecOp::BroadcastInDim {
            shape: vec![DimExpr::Const(2), DimExpr::Const(2)],
            dims: vec![0, 1],
        },
        ExecOp::Convert { to: DType::I64 },
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
        ExecOp::ReduceSum { axes: vec![0] },
        ExecOp::ExtractDiag {
            axis_a: 0,
            axis_b: 1,
        },
        ExecOp::EmbedDiag {
            axis_a: 0,
            axis_b: 1,
        },
        ExecOp::Tril { k: -1 },
        ExecOp::Triu { k: 1 },
        ExecOp::Add,
        ExecOp::Multiply,
        ExecOp::Negate,
        ExecOp::Conj,
        ExecOp::Divide,
        ExecOp::Abs,
        ExecOp::Sign,
        ExecOp::Maximum,
        ExecOp::Minimum,
        ExecOp::Compare(CompareDir::Le),
        ExecOp::Select,
        ExecOp::Clamp,
        ExecOp::Exp,
        ExecOp::Log,
        ExecOp::Sin,
        ExecOp::Cos,
        ExecOp::Tanh,
        ExecOp::Sqrt,
        ExecOp::Rsqrt,
        ExecOp::Pow,
        ExecOp::Expm1,
        ExecOp::Log1p,
        ExecOp::Gather(gather_config()),
        ExecOp::GatherDynamicSliceSizes {
            offset_dims: vec![1],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 1,
            slice_sizes: vec![DimExpr::Const(1)],
        },
        ExecOp::Scatter(scatter_config()),
        ExecOp::Slice(slice_config()),
        ExecOp::DynamicSlice {
            slice_sizes: vec![1, 2],
        },
        ExecOp::DynamicUpdateSlice,
        ExecOp::Pad(pad_config()),
        ExecOp::Concatenate { axis: 1 },
        ExecOp::Reverse { axes: vec![0, 1] },
        ExecOp::ShapeOf { axis: 0 },
        ExecOp::DynamicTruncate { axis: 0 },
        ExecOp::PadToMatch { axis: 1 },
        ExecOp::ReduceProd { axes: vec![0] },
        ExecOp::ReduceMax { axes: vec![1] },
        ExecOp::ReduceMin { axes: vec![0, 1] },
        ExecOp::Extension(Arc::new(CollidingExtension { payload: 3 })),
    ];
    let expected_len = ops.len();
    let program = program_with_ops(ops);

    let key = compute_cache_key(&program);
    assert_eq!(key.fingerprint.instructions.len(), expected_len);
    assert_eq!(key.extensions.len(), 1);
    assert!(matches!(
        key.fingerprint.instructions[10].op,
        ExecOpKey::Tril { k: -1 }
    ));
    assert!(matches!(
        key.fingerprint.instructions[35].op,
        ExecOpKey::GatherDynamicSliceSizes {
            index_vector_dim: 1,
            ..
        }
    ));
    assert!(matches!(
        key.fingerprint.instructions[49].op,
        ExecOpKey::Extension {
            family_id: "runtime.cache-key-collision-test.v1",
            ..
        }
    ));

    let mut cache = LruCache::unbounded();
    cache.put(key, program);

    let stats = compile_cache_stats(&cache);
    assert_eq!(stats.entries, 1);
    assert!(
        stats.retained_bytes > expected_len * std::mem::size_of::<ExecInstruction>(),
        "stats should account for retained structural payloads"
    );
}

#[test]
fn retained_byte_accounting_saturates_on_overflow() {
    assert_eq!(saturating_sum([usize::MAX, 1]), usize::MAX);
    assert_eq!(saturating_sum([usize::MAX - 4, 2, 8]), usize::MAX);
}

fn unary_program(op: ExecOp) -> ExecProgram {
    ExecProgram {
        instructions: vec![ExecInstruction {
            op,
            input_slots: vec![0],
            output_slots: vec![1],
            dtype: DType::F64,
            output_shapes: vec![vec![DimExpr::Const(4)]].into(),
            output_extents: vec![vec![ShapeExtent::exact(DimExpr::Const(4))]].into(),
            last_use: vec![true],
        }],
        input_slots: vec![0],
        output_slots: vec![1],
        n_slots: 2,
    }
}

fn program_with_ops(ops: Vec<ExecOp>) -> ExecProgram {
    let instructions = ops
        .into_iter()
        .enumerate()
        .map(|(index, op)| ExecInstruction {
            op,
            input_slots: vec![0],
            output_slots: vec![index + 1],
            dtype: DType::F64,
            output_shapes: vec![vec![DimExpr::Const(4)]].into(),
            output_extents: vec![vec![ShapeExtent::exact(DimExpr::Const(4))]].into(),
            last_use: vec![index % 2 == 0],
        })
        .collect::<Vec<_>>();
    let n_slots = instructions.len() + 1;
    ExecProgram {
        instructions,
        input_slots: vec![0],
        output_slots: vec![n_slots - 1],
        n_slots,
    }
}

fn extension_program(payload: u8) -> ExecProgram {
    unary_program(ExecOp::Extension(Arc::new(CollidingExtension { payload })))
}

fn dot_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![0],
        rhs_batch_dims: vec![1],
    }
}

fn gather_config() -> GatherConfig {
    GatherConfig {
        offset_dims: vec![1],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1, 2],
    }
}

fn scatter_config() -> ScatterConfig {
    ScatterConfig {
        update_window_dims: vec![1],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    }
}

fn slice_config() -> SliceConfig {
    SliceConfig {
        starts: vec![0, 1],
        limits: vec![2, 3],
        strides: vec![1, 2],
    }
}

fn pad_config() -> PadConfig {
    PadConfig {
        edge_padding_low: vec![0, 1],
        edge_padding_high: vec![1, 0],
        interior_padding: vec![0, 2],
    }
}

#[derive(Clone, Debug)]
struct CollidingExtension {
    payload: u8,
}

impl ExtensionOp for CollidingExtension {
    fn family_id(&self) -> &'static str {
        "runtime.cache-key-collision-test.v1"
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_u8(0);
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| self.payload == other.payload)
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
    ) -> Vec<(DType, Vec<SymDim>)> {
        vec![(input_dtypes[0], input_shapes[0].to_vec())]
    }

    fn eager_execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
        Ok(vec![inputs[0].clone()])
    }
}
