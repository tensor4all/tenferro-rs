use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use lru::LruCache;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_ops::{ShapeExtent, SymDim};
use tenferro_tensor::{DType, Tensor};

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

fn extension_program(payload: u8) -> ExecProgram {
    unary_program(ExecOp::Extension(Arc::new(CollidingExtension { payload })))
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
