use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use tenferro_cpu::CpuBackend;
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_ops::SymDim;
use tenferro_tensor::{DType, Tensor};

use super::{ExtensionCacheStore, ExtensionExecutionContext};
use crate::exec::{ExecInstruction, ExecOp, ExecProgram};

#[test]
fn core_exec_program_context_rejects_nested_extension_ops() {
    let program = ExecProgram {
        instructions: vec![ExecInstruction {
            op: ExecOp::Extension(Arc::new(TestExtension)),
            input_slots: vec![],
            output_slots: vec![0],
            dtype: DType::F64,
            output_shapes: vec![vec![]].into(),
            output_extents: vec![vec![]].into(),
            last_use: vec![],
        }],
        input_slots: vec![],
        output_slots: vec![0],
        n_slots: 1,
    };

    let mut backend = CpuBackend::new();
    let mut caches = ExtensionCacheStore::new();
    let mut ctx = ExtensionExecutionContext::new(&mut backend, &mut caches);
    let err = ctx
        .execute_core_exec_program_unsegmented(&program, vec![])
        .unwrap_err();

    let message = err.to_string();
    assert!(message.contains("core ExecProgram"), "{message}");
    assert!(message.contains("tenferro-tests.nested.v1"), "{message}");
}

#[derive(Clone, Debug)]
struct TestExtension;

impl ExtensionOp for TestExtension {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.nested.v1"
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
        0
    }

    fn output_count(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        _input_dtypes: &[DType],
        _input_shapes: &[&[SymDim]],
    ) -> Vec<(DType, Vec<SymDim>)> {
        vec![(DType::F64, vec![])]
    }

    fn eager_execute(&self, _inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
        Ok(vec![Tensor::from_vec_col_major(vec![], vec![1.0_f64])])
    }
}
