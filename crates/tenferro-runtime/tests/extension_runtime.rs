use std::any::Any;
use std::hash::Hasher;
use std::num::NonZeroUsize;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use tenferro_cpu::CpuBackend;
use tenferro_ops::ext_op::{ExtensionOp, HostReference};
use tenferro_ops::SymDim;
use tenferro_runtime::{
    ExtensionCacheKey, ExtensionCacheLimits, ExtensionCacheSelector, ExtensionExecutionContext,
    ExtensionExecutor, ExtensionRegistry, ExtensionRuntime, ExtensionRuntimeRegistryError,
    HostReferenceRuntime,
};
use tenferro_tensor::{
    Buffer, BufferHandle, DType, MemoryKind, Placement, Tensor, TensorOwnedView, TensorRead,
    TypedTensor,
};

#[derive(Clone, Debug)]
struct IdentityRuntimeOp {
    family: &'static str,
}

impl ExtensionOp for IdentityRuntimeOp {
    fn family_id(&self) -> &'static str {
        self.family
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write(self.family.as_bytes());
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<IdentityRuntimeOp>()
            .is_some_and(|op| op.family == self.family)
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

    fn host_reference(&self) -> Option<&dyn HostReference> {
        Some(self)
    }
}

impl HostReference for IdentityRuntimeOp {
    fn execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
        Ok(vec![inputs[0].clone()])
    }
}

#[derive(Clone, Debug)]
struct BackendOnlyRuntimeOp {
    family: &'static str,
}

impl ExtensionOp for BackendOnlyRuntimeOp {
    fn family_id(&self) -> &'static str {
        self.family
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write(self.family.as_bytes());
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<BackendOnlyRuntimeOp>()
            .is_some_and(|op| op.family == self.family)
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
}

#[derive(Debug)]
struct IdentityRuntime {
    family: &'static str,
}

impl ExtensionRuntime<CpuBackend> for IdentityRuntime {
    fn family_id(&self) -> &'static str {
        self.family
    }

    fn execute(
        &self,
        _op: &dyn ExtensionOp,
        inputs: &[&Tensor],
        ctx: &mut ExtensionExecutionContext<'_, CpuBackend>,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        let _ = ctx.backend();
        let _ = ctx.backend_mut();
        let key = ExtensionCacheKey::new(self.family, "identity", 0);
        let _ = ctx.caches();
        ctx.caches_mut().put(key, String::from("cached plan"), 11);
        assert!(ctx.caches_mut().get::<String>(&key).is_some());
        Ok(vec![inputs[0].clone()])
    }

    fn execute_reads(
        &self,
        op: &dyn ExtensionOp,
        inputs: &[TensorRead<'_>],
        ctx: &mut ExtensionExecutionContext<'_, CpuBackend>,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        let materialized_inputs: Vec<Tensor> = inputs
            .iter()
            .map(TensorRead::to_tensor)
            .collect::<tenferro_tensor::Result<_>>()?;
        let input_refs: Vec<&Tensor> = materialized_inputs.iter().collect();
        self.execute(op, &input_refs, ctx)
    }
}

#[derive(Debug)]
struct WrongOutputCountRuntime {
    family: &'static str,
    return_count: usize,
}

impl ExtensionRuntime<CpuBackend> for WrongOutputCountRuntime {
    fn family_id(&self) -> &'static str {
        self.family
    }

    fn execute(
        &self,
        _op: &dyn ExtensionOp,
        inputs: &[&Tensor],
        _ctx: &mut ExtensionExecutionContext<'_, CpuBackend>,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        Ok(std::iter::repeat_with(|| inputs[0].clone())
            .take(self.return_count)
            .collect())
    }

    fn execute_reads(
        &self,
        op: &dyn ExtensionOp,
        inputs: &[TensorRead<'_>],
        ctx: &mut ExtensionExecutionContext<'_, CpuBackend>,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        let materialized_inputs: Vec<Tensor> = inputs
            .iter()
            .map(TensorRead::to_tensor)
            .collect::<tenferro_tensor::Result<_>>()?;
        let input_refs: Vec<&Tensor> = materialized_inputs.iter().collect();
        self.execute(op, &input_refs, ctx)
    }
}

#[test]
fn extension_registry_rejects_malformed_and_is_idempotent() {
    let mut registry = ExtensionRegistry::<CpuBackend>::new();
    assert!(registry.is_empty());
    assert_eq!(registry.len(), 0);

    let malformed = registry
        .register(Arc::new(IdentityRuntime { family: "bad" }))
        .expect_err("malformed family should be rejected");
    assert!(matches!(
        malformed,
        ExtensionRuntimeRegistryError::MalformedFamilyId { family_id: "bad" }
    ));

    let family = "runtime.identity.v1";
    registry
        .register(Arc::new(IdentityRuntime { family }))
        .expect("first runtime registration");
    registry
        .register(Arc::new(IdentityRuntime { family }))
        .expect("duplicate runtime registration is idempotent");
    assert!(registry.contains(family));
    assert!(registry.get(family).is_some());
    assert_eq!(registry.len(), 1);
    assert!(!registry.is_empty());
}

#[test]
fn extension_executor_executes_registered_runtime_and_manages_caches() {
    let family = "runtime.execute.v1";
    let mut registry = ExtensionRegistry::<CpuBackend>::new();
    registry
        .register(Arc::new(IdentityRuntime { family }))
        .expect("runtime registration");
    let mut executor = ExtensionExecutor::with_parts(registry, Default::default());
    assert!(executor.registry().contains(family));
    assert!(executor.registry_mut().contains(family));
    assert_eq!(
        executor.caches().stats(ExtensionCacheSelector::All).entries,
        0
    );
    assert_eq!(
        executor
            .caches_mut()
            .stats(ExtensionCacheSelector::All)
            .entries,
        0
    );

    let limits = ExtensionCacheLimits::new(NonZeroUsize::new(2).unwrap());
    executor.set_cache_limits(limits);
    assert_eq!(executor.cache_limits().max_entries().get(), 2);

    let mut backend = CpuBackend::new();
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let output = executor
        .execute(&mut backend, &IdentityRuntimeOp { family }, &[&input])
        .expect("extension execution");
    assert_eq!(output[0].as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    assert_eq!(executor.cache_stats().entries, 1);

    executor.clear_caches();
    assert_eq!(executor.cache_stats().entries, 0);
}

#[test]
fn host_reference_runtime_delegates_to_optional_host_reference() {
    let family = "runtime.host-reference.v1";
    let mut executor = ExtensionExecutor::<CpuBackend>::new();
    executor
        .registry_mut()
        .register(Arc::new(HostReferenceRuntime::<CpuBackend>::new(family)))
        .expect("host reference runtime registration");

    let mut backend = CpuBackend::new();
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let output = executor
        .execute(&mut backend, &IdentityRuntimeOp { family }, &[&input])
        .expect("host reference execution");

    assert_eq!(output[0].as_slice::<f64>().unwrap(), &[1.0, 2.0]);
}

#[test]
fn host_reference_runtime_reports_backend_only_family() {
    let family = "runtime.backend-only.v1";
    let mut executor = ExtensionExecutor::<CpuBackend>::new();
    executor
        .registry_mut()
        .register(Arc::new(HostReferenceRuntime::<CpuBackend>::new(family)))
        .expect("host reference runtime registration");

    let mut backend = CpuBackend::new();
    let input = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    let err = executor
        .execute(&mut backend, &BackendOnlyRuntimeOp { family }, &[&input])
        .expect_err("backend-only op should not fabricate a host reference");

    assert!(matches!(
        err,
        tenferro_tensor::Error::NoHostReference {
            family_id: "runtime.backend-only.v1"
        }
    ));
}

#[test]
fn extension_executor_rejects_runtime_output_count_mismatch() {
    let family = "runtime.output-count.v1";
    let mut registry = ExtensionRegistry::<CpuBackend>::new();
    registry
        .register(Arc::new(WrongOutputCountRuntime {
            family,
            return_count: 0,
        }))
        .expect("runtime registration");
    let mut executor = ExtensionExecutor::with_parts(registry, Default::default());
    let mut backend = CpuBackend::new();
    let input = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();

    let err = executor
        .execute(&mut backend, &IdentityRuntimeOp { family }, &[&input])
        .expect_err("runtime output count mismatch should error");

    let message = err.to_string();
    assert!(
        message.contains("family_id \"runtime.output-count.v1\""),
        "{message}"
    );
    assert!(message.contains("returned 0 outputs"), "{message}");
    assert!(message.contains("declared 1 outputs"), "{message}");
}

#[test]
fn extension_executor_rejects_input_count_mismatch_before_runtime_call() {
    let family = "runtime.input-count.v1";
    let mut registry = ExtensionRegistry::<CpuBackend>::new();
    registry
        .register(Arc::new(IdentityRuntime { family }))
        .expect("runtime registration");
    let mut executor = ExtensionExecutor::with_parts(registry, Default::default());
    let mut backend = CpuBackend::new();

    let err = executor
        .execute(&mut backend, &IdentityRuntimeOp { family }, &[])
        .expect_err("input count mismatch should error before runtime call");

    let message = err.to_string();
    assert!(message.contains("expects 1 inputs, got 0"), "{message}");
}

#[test]
fn extension_executor_rejects_read_runtime_output_count_mismatch() {
    let family = "runtime.read-output-count.v1";
    let mut registry = ExtensionRegistry::<CpuBackend>::new();
    registry
        .register(Arc::new(WrongOutputCountRuntime {
            family,
            return_count: 2,
        }))
        .expect("runtime registration");
    let mut executor = ExtensionExecutor::with_parts(registry, Default::default());
    let input = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    let read = TensorRead::from_tensor(&input);
    let mut backend = CpuBackend::new();

    let err = executor
        .execute_reads(&mut backend, &IdentityRuntimeOp { family }, &[read])
        .expect_err("runtime read output count mismatch should error");

    let message = err.to_string();
    assert!(
        message.contains("family_id \"runtime.read-output-count.v1\""),
        "{message}"
    );
    assert!(message.contains("returned 2 outputs"), "{message}");
    assert!(message.contains("declared 1 outputs"), "{message}");
}

#[test]
fn extension_executor_rejects_read_input_count_mismatch_before_runtime_call() {
    let family = "runtime.read-input-count.v1";
    let mut registry = ExtensionRegistry::<CpuBackend>::new();
    registry
        .register(Arc::new(IdentityRuntime { family }))
        .expect("runtime registration");
    let mut executor = ExtensionExecutor::with_parts(registry, Default::default());
    let mut backend = CpuBackend::new();

    let err = executor
        .execute_reads(&mut backend, &IdentityRuntimeOp { family }, &[])
        .expect_err("read input count mismatch should error before runtime call");

    let message = err.to_string();
    assert!(message.contains("expects 1 inputs, got 0"), "{message}");
}

#[test]
fn extension_executor_read_fallback_reports_backend_view_materialization_error_without_panic() {
    let family = "runtime.backend-view.v1";
    let mut registry = ExtensionRegistry::<CpuBackend>::new();
    registry
        .register(Arc::new(IdentityRuntime { family }))
        .expect("runtime registration");
    let mut executor = ExtensionExecutor::with_parts(registry, Default::default());
    let base = Arc::new(Tensor::F64(
        TypedTensor::<f64>::from_buffer_col_major(
            vec![1],
            Buffer::Backend(Arc::new(BufferHandle::<f64>::new_with_len(91, 1))),
            Placement {
                memory_kind: MemoryKind::Device,
                device: None,
            },
        )
        .unwrap(),
    ));
    let view = TensorOwnedView::from_tensor(base);
    let read = view.tensor_read();
    let mut backend = CpuBackend::new();

    let result = catch_unwind(AssertUnwindSafe(|| {
        executor.execute_reads(&mut backend, &IdentityRuntimeOp { family }, &[read])
    }));

    assert!(
        result.is_ok(),
        "backend view materialization should return Err, not panic"
    );
    let err = result
        .unwrap()
        .expect_err("backend view materialization should error");
    let message = err.to_string();
    assert!(
        message.contains("backend buffers cannot be materialized"),
        "{message}"
    );
    assert!(message.contains("download explicitly first"), "{message}");
}

#[test]
fn extension_executor_default_read_path_materializes_views_at_runtime_boundary() {
    let family = "runtime.execute_reads.v1";
    let mut registry = ExtensionRegistry::<CpuBackend>::new();
    registry
        .register(Arc::new(IdentityRuntime { family }))
        .expect("runtime registration");
    let mut executor = ExtensionExecutor::with_parts(registry, Default::default());

    let base = Arc::new(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    );
    let view = TensorOwnedView::from_parts(Arc::clone(&base), vec![3, 2], vec![2, 1], 0).unwrap();
    let read = TensorRead::from_view(view.tensor_view());
    let mut backend = CpuBackend::new();
    let op = IdentityRuntimeOp { family };

    let outputs = executor
        .execute_reads(&mut backend, &op, &[read])
        .expect("execute read input through fallback runtime");

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape(), &[3, 2]);
    assert_eq!(
        outputs[0].as_slice::<f64>().unwrap(),
        &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    );
}

#[test]
fn extension_executor_reports_missing_runtime() {
    let mut executor = ExtensionExecutor::<CpuBackend>::new();
    let mut backend = CpuBackend::new();
    let input = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    let err = executor
        .execute(
            &mut backend,
            &IdentityRuntimeOp {
                family: "runtime.missing.v1",
            },
            &[&input],
        )
        .expect_err("missing runtime should error");
    assert!(err.to_string().contains("missing runtime"));
}
