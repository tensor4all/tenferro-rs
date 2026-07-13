use computegraph::types::OperationRole;
use tenferro_ops::{dim_expr::DimExpr, ShapeExtent};
use tenferro_tensor::DType;

use super::*;

#[derive(Clone, Debug)]
struct TestExtension {
    input_count: usize,
    output_count: usize,
    inferred_outputs: usize,
}

impl ExtensionOp for TestExtension {
    fn family_id(&self) -> &'static str {
        "test.extension"
    }

    fn payload_hash(&self, _hasher: &mut dyn std::hash::Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>().is_some()
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn input_count(&self) -> usize {
        self.input_count
    }

    fn output_count(&self) -> usize {
        self.output_count
    }

    fn infer_output_meta(
        &self,
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        let dtype = ctx.input_dtype(0)?;
        let shape = ctx.input_shape(0)?.to_vec();
        Ok((0..self.inferred_outputs)
            .map(|_| (dtype, shape.clone()))
            .collect())
    }
}

#[test]
fn apply_returns_error_for_input_count_mismatch() {
    let op = Arc::new(TestExtension {
        input_count: 2,
        output_count: 1,
        inferred_outputs: 1,
    });
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();

    let Err(err) = apply(op, &[&x]) else {
        panic!("input count mismatch should be an error");
    };

    let message = err.to_string();
    assert!(message.contains("test.extension"), "{message}");
    assert!(message.contains("expects 2 inputs, got 1"), "{message}");
}

#[test]
fn apply_returns_error_for_output_metadata_count_mismatch() {
    let op = Arc::new(TestExtension {
        input_count: 1,
        output_count: 2,
        inferred_outputs: 1,
    });
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();

    let Err(err) = apply(op, &[&x]) else {
        panic!("output metadata mismatch should be an error");
    };

    match err {
        Error::TensorRuntime(tenferro_tensor::Error::InvalidConfig { op, message }) => {
            assert_eq!(op, "extension");
            assert_eq!(
                message,
                "family_id=\"test.extension\": infer_output_meta produced 1 output metadata entries; op declared 2 outputs"
            );
        }
        other => panic!("expected structured extension InvalidConfig, got {other:?}"),
    }
}

#[test]
fn execute_lowered_program_with_backend_cache_rejects_nested_extension_ops() {
    let ext = Arc::new(TestExtension {
        input_count: 1,
        output_count: 1,
        inferred_outputs: 1,
    });
    let program = crate::exec::ExecProgram {
        instructions: vec![crate::exec::ExecInstruction {
            op: crate::exec::ExecOp::Extension(ext),
            input_slots: vec![0],
            output_slots: vec![1],
            dtype: DType::F64,
            output_shapes: vec![vec![DimExpr::Const(1)]].into(),
            output_extents: vec![vec![ShapeExtent::exact(DimExpr::Const(1))]].into(),
            last_use: vec![true],
        }],
        input_slots: vec![0],
        output_slots: vec![1],
        n_slots: 2,
    };
    let mut backend = tenferro_cpu::CpuBackend::new();
    let mut backend_cache = Default::default();
    let input = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();

    let err = execute_lowered_program_with_backend_cache(
        &mut backend,
        &program,
        vec![input],
        &mut backend_cache,
    )
    .unwrap_err();

    let message = err.to_string();
    assert!(message.contains("core ExecProgram"), "{message}");
    assert!(message.contains("extension family_id"), "{message}");
}

#[test]
fn apply_expanded_graph_builds_standard_op_without_extension() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let y = TracedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();

    let outputs = apply_expanded_graph(
        &[&x, &y],
        vec![(DType::F64, vec![SymDim::from(2)])],
        |builder, inputs| {
            Ok(builder.add_operation(StdTensorOp::Add, inputs.to_vec(), OperationRole::Primary))
        },
    )
    .expect("expanded graph should build");

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].rank, 1);
    assert_eq!(outputs[0].dtype, DType::F64);
    assert!(outputs[0]
        .graph
        .operations()
        .iter()
        .all(|node| !matches!(node.operation, StdTensorOp::Extension(_))));
}

#[test]
fn apply_expanded_graph_rejects_output_metadata_count_mismatch() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();

    let result = apply_expanded_graph(&[&x], vec![], |builder, inputs| {
        Ok(builder.add_operation(StdTensorOp::Neg, inputs.to_vec(), OperationRole::Primary))
    });
    let Err(err) = result else {
        panic!("expanded graph output metadata mismatch should error");
    };

    let message = err.to_string();
    assert!(message.contains("expanded graph"), "{message}");
    assert!(message.contains("returned 1 outputs"), "{message}");
    assert!(message.contains("0 output metadata entries"), "{message}");
}
