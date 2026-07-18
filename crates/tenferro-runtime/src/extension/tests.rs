use computegraph::types::OperationRole;
use std::sync::atomic::{AtomicUsize, Ordering};
use tenferro_ops::{dim_expr::DimExpr, ShapeExtent};
use tenferro_tensor::{DType, ValidationError};

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
        shape_guards: Vec::new(),
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
    assert!(outputs[0].constraint_scopes.materialize().is_empty());
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

#[test]
fn constraint_scope_analysis_invokes_extension_inference_once_and_attaches_one_scope() {
    let calls = Arc::new(AtomicUsize::new(0));
    let lhs = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();
    let rhs = TracedTensor::from_vec_col_major(vec![3], vec![2.0_f64; 3]).unwrap();

    let outputs = apply(
        Arc::new(CountedConstraintExtension {
            calls: Arc::clone(&calls),
        }),
        &[&lhs, &rhs],
    )
    .unwrap();

    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(outputs[0].constraint_scopes.materialize().len(), 1);
    let constraints = outputs[0].constraint_scopes.as_slice()[0].constraints();
    assert_eq!(constraints.len(), 1);
    assert_eq!(
        constraints[0].origins,
        outputs
            .iter()
            .map(|output| output.graph.values()[output.val].key.clone())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        constraints[0].inputs,
        vec![
            lhs.graph.values()[lhs.val].key.clone(),
            rhs.graph.values()[rhs.val].key.clone(),
        ]
    );
}

fn counted_constraint_output(calls: &Arc<AtomicUsize>) -> TracedTensor {
    let lhs = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();
    let rhs = TracedTensor::from_vec_col_major(vec![3], vec![2.0_f64; 3]).unwrap();
    apply(
        Arc::new(CountedConstraintExtension {
            calls: Arc::clone(calls),
        }),
        &[&lhs, &rhs],
    )
    .unwrap()
    .remove(0)
}

fn reachable_constraint_count(tensor: &TracedTensor) -> usize {
    tensor
        .constraint_scopes
        .as_slice()
        .iter()
        .map(|scope| scope.constraints().len())
        .sum()
}

#[test]
fn graph_local_constraint_scope_extension_child_does_not_reinfer_or_duplicate_ancestor() {
    let calls = Arc::new(AtomicUsize::new(0));
    let ancestor = counted_constraint_output(&calls);
    assert_eq!(calls.load(Ordering::Relaxed), 1);

    let child = apply(
        Arc::new(TestExtension {
            input_count: 1,
            output_count: 1,
            inferred_outputs: 1,
        }),
        &[&ancestor],
    )
    .unwrap()
    .remove(0);

    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(child.constraint_scopes.materialize().len(), 1);
    assert_eq!(reachable_constraint_count(&child), 1);
}

#[test]
fn graph_local_constraint_scope_expanded_child_does_not_reinfer_or_duplicate_ancestor() {
    let calls = Arc::new(AtomicUsize::new(0));
    let ancestor = counted_constraint_output(&calls);
    assert_eq!(calls.load(Ordering::Relaxed), 1);

    let child = apply_expanded_graph(
        &[&ancestor],
        vec![(DType::F64, vec![SymDim::from(3)])],
        |builder, inputs| {
            Ok(builder.add_operation(StdTensorOp::Neg, inputs.to_vec(), OperationRole::Primary))
        },
    )
    .unwrap()
    .remove(0);

    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(child.constraint_scopes.materialize().len(), 1);
    assert_eq!(reachable_constraint_count(&child), 1);
}

#[test]
fn extension_chain_graph_analysis_visit_count_is_root_local() {
    const DEPTH: usize = 16;
    let input = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();

    let (result, graph_visits, operation_visits) =
        crate::metadata::test_support::with_visit_count(|| {
            let mut value = input;
            for _ in 0..DEPTH {
                value = apply(
                    Arc::new(TestExtension {
                        input_count: 1,
                        output_count: 1,
                        inferred_outputs: 1,
                    }),
                    &[&value],
                )?
                .remove(0);
            }
            Ok::<_, Error>(value)
        });

    result.unwrap();
    assert_eq!(graph_visits, DEPTH);
    assert_eq!(operation_visits, DEPTH);
}

#[test]
fn expanded_chain_graph_analysis_visit_count_is_root_local() {
    const DEPTH: usize = 16;
    let input = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();

    let (result, graph_visits, operation_visits) =
        crate::metadata::test_support::with_visit_count(|| {
            let mut value = input;
            for _ in 0..DEPTH {
                value = apply_expanded_graph(
                    &[&value],
                    vec![(DType::F64, vec![SymDim::from(3)])],
                    |builder, inputs| {
                        Ok(builder.add_operation(
                            StdTensorOp::Neg,
                            inputs.to_vec(),
                            OperationRole::Primary,
                        ))
                    },
                )?
                .remove(0);
            }
            Ok::<_, Error>(value)
        });

    result.unwrap();
    assert_eq!(graph_visits, DEPTH);
    assert_eq!(operation_visits, DEPTH);
}

#[test]
fn graph_analysis_resolves_unregistered_multi_output_parent_on_demand() {
    let input = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();

    let mut parent_builder = GraphBuilder::new();
    parent_builder.add_parent(input.graph.clone());
    let parent_outputs = parent_builder.add_operation(
        StdTensorOp::Extension(Arc::new(TestExtension {
            input_count: 1,
            output_count: 2,
            inferred_outputs: 2,
        })),
        vec![ValueRef::External(
            input.graph.values()[input.val].key.clone(),
        )],
        OperationRole::Primary,
    );
    parent_builder.set_outputs(parent_outputs.clone());
    let parent = Arc::new(parent_builder.build());

    let mut child_builder = GraphBuilder::new();
    child_builder.add_parent(Arc::clone(&parent));
    let child_output = child_builder.add_operation(
        StdTensorOp::Add,
        parent_outputs
            .iter()
            .map(|&output| ValueRef::External(parent.values()[output].key.clone()))
            .collect(),
        OperationRole::Primary,
    )[0];
    child_builder.set_outputs(vec![child_output]);
    let child = child_builder.build();

    let analysis = crate::metadata::register_scoped_graph_analysis(&child, []).unwrap();
    assert!(analysis.constraints.is_empty());
    let child_meta = crate::metadata::registered_meta(&child.values()[child_output].key).unwrap();
    assert_eq!(child_meta.dtype, DType::F64);
    assert_eq!(child_meta.rank(), 1);
    for &output in &parent_outputs {
        assert_eq!(
            crate::metadata::registered_meta(&parent.values()[output].key)
                .unwrap()
                .bound_shape(),
            Some(vec![SymDim::from(3)])
        );
    }
}

#[test]
fn graph_analysis_scope_retains_registered_parent_metadata_it_borrows() {
    let input = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();

    let mut parent_builder = GraphBuilder::new();
    parent_builder.add_parent(Arc::clone(&input.graph));
    let parent_output = parent_builder.add_operation(
        StdTensorOp::Neg,
        vec![ValueRef::External(
            input.graph.values()[input.val].key.clone(),
        )],
        OperationRole::Primary,
    )[0];
    parent_builder.set_outputs(vec![parent_output]);
    let parent = Arc::new(parent_builder.build());
    let parent_key = parent.values()[parent_output].key.clone();
    let parent_analysis =
        crate::metadata::register_scoped_graph_analysis(parent.as_ref(), []).unwrap();

    let mut child_builder = GraphBuilder::new();
    child_builder.add_parent(Arc::clone(&parent));
    let child_output = child_builder.add_operation(
        StdTensorOp::Neg,
        vec![ValueRef::External(parent_key.clone())],
        OperationRole::Primary,
    )[0];
    child_builder.set_outputs(vec![child_output]);
    let child = child_builder.build();
    let child_analysis = crate::metadata::register_scoped_graph_analysis(&child, []).unwrap();

    drop(parent_analysis);

    let retained = crate::metadata::registered_meta(&parent_key)
        .expect("child analysis scope must retain metadata borrowed from its parent");
    assert_eq!(retained.dtype, DType::F64);
    assert_eq!(retained.bound_shape(), Some(vec![SymDim::from(3)]));
    drop(child_analysis);
    assert!(
        crate::metadata::registered_meta(&parent_key).is_err(),
        "dropping the final analysis scope must release the borrowed metadata"
    );
}

#[test]
fn parent_owner_index_is_built_once_for_many_parents_and_missing_inputs() {
    const PARENTS: usize = 8;
    let mut inputs = Vec::new();
    let mut parents = Vec::new();
    let mut parent_outputs = Vec::new();
    for _ in 0..PARENTS {
        let input = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();
        let mut builder = GraphBuilder::new();
        builder.add_parent(Arc::clone(&input.graph));
        let output = builder.add_operation(
            StdTensorOp::Neg,
            vec![ValueRef::External(
                input.graph.values()[input.val].key.clone(),
            )],
            OperationRole::Primary,
        )[0];
        builder.set_outputs(vec![output]);
        let parent = Arc::new(builder.build());
        parent_outputs.push(parent.values()[output].key.clone());
        parents.push(parent);
        inputs.push(input);
    }

    let mut root_builder = GraphBuilder::new();
    for parent in &parents {
        root_builder.add_parent(Arc::clone(parent));
    }
    let mut root_outputs = Vec::new();
    for parent_output in parent_outputs {
        root_outputs.extend(root_builder.add_operation(
            StdTensorOp::Neg,
            vec![ValueRef::External(parent_output)],
            OperationRole::Primary,
        ));
    }
    root_builder.set_outputs(root_outputs);
    let root = root_builder.build();

    let (analysis, index_builds, parent_value_visits) =
        crate::metadata::test_support::with_parent_owner_index_count(|| {
            crate::metadata::register_scoped_graph_analysis(&root, [])
        });

    analysis.unwrap();
    assert_eq!(index_builds, 1);
    assert_eq!(parent_value_visits, PARENTS);
}

#[test]
fn parent_owner_index_is_not_built_for_registered_inputs() {
    let input = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();

    let (output, index_builds, parent_value_visits) =
        crate::metadata::test_support::with_parent_owner_index_count(|| {
            apply(
                Arc::new(TestExtension {
                    input_count: 1,
                    output_count: 1,
                    inferred_outputs: 1,
                }),
                &[&input],
            )
        });

    output.unwrap();
    assert_eq!(index_builds, 0);
    assert_eq!(parent_value_visits, 0);
}
