use computegraph::types::OperationRole;
use std::sync::atomic::{AtomicUsize, Ordering};
use tenferro_tensor::{DType, Tensor, ValidationError};

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
    let trace = tenferro_runtime::TraceContext::new();
    let op = Arc::new(TestExtension {
        input_count: 2,
        output_count: 1,
        inferred_outputs: 1,
    });
    let x = trace.from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();

    let Err(err) = apply(op, &[&x]) else {
        panic!("input count mismatch should be an error");
    };

    let message = err.to_string();
    assert!(message.contains("test.extension"), "{message}");
    assert!(message.contains("expects 2 inputs, got 1"), "{message}");
}

#[test]
fn apply_returns_error_for_output_metadata_count_mismatch() {
    let trace = tenferro_runtime::TraceContext::new();
    let op = Arc::new(TestExtension {
        input_count: 1,
        output_count: 2,
        inferred_outputs: 1,
    });
    let x = trace.from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();

    let Err(err) = apply(op, &[&x]) else {
        panic!("output metadata mismatch should be an error");
    };

    match err {
        Error::TensorRuntime(tenferro_tensor::Error::Validation { op, source }) => {
            assert_eq!(op, "extension");
            assert!(matches!(
                source,
                ValidationError::InvalidArgument {
                    argument: "output metadata",
                    message,
                } if message.contains("family_id=\"test.extension\"")
                    && message.contains("declared 2 outputs")
            ));
        }
        other => panic!("expected structured extension validation error, got {other:?}"),
    }
}

#[test]
fn apply_expanded_graph_builds_standard_op_without_extension() {
    let trace = tenferro_runtime::TraceContext::new();
    let x = trace.from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let y = trace.from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();

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
    assert!(outputs[0].constraint_scopes.materialize().is_empty());
    assert!(outputs[0]
        .graph
        .operations()
        .iter()
        .all(|node| !matches!(node.operation, StdTensorOp::Extension(_))));
}

#[test]
fn apply_expanded_graph_rejects_output_metadata_count_mismatch() {
    let trace = tenferro_runtime::TraceContext::new();
    let x = trace.from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();

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

#[derive(Clone, Debug)]
struct CountedConstraintExtension {
    calls: Arc<AtomicUsize>,
}

impl ExtensionOp for CountedConstraintExtension {
    fn family_id(&self) -> &'static str {
        "test.counted-constraint"
    }

    fn payload_hash(&self, _hasher: &mut dyn std::hash::Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| Arc::ptr_eq(&self.calls, &other.calls))
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn input_count(&self) -> usize {
        2
    }

    fn output_count(&self) -> usize {
        2
    }

    fn infer_output_meta(
        &self,
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        ctx.require_axes_equal((0, 0), (1, 0))?;
        let meta = (ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec());
        Ok(vec![meta.clone(), meta])
    }
}
