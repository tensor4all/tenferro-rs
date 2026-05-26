use std::any::Any;
use std::hash::Hasher;
use std::num::NonZeroUsize;
use std::sync::Arc;

use tenferro_ops::ext_op::ExtensionOp;
use tenferro_ops::SymDim;
use tenferro_runtime::{
    ExtensionCacheKey, ExtensionCacheLimits, ExtensionCacheSelector, ExtensionExecutionContext,
    ExtensionExecutor, ExtensionRegistry, ExtensionRuntime, ExtensionRuntimeRegistryError,
};
use tenferro_tensor::{cpu::CpuBackend, DType, Tensor};

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

    fn n_inputs(&self) -> usize {
        1
    }

    fn n_outputs(&self) -> usize {
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
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let output = executor
        .execute(&mut backend, &IdentityRuntimeOp { family }, &[&input])
        .expect("extension execution");
    assert_eq!(output[0].as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    assert_eq!(executor.cache_stats().entries, 1);

    executor.clear_caches();
    assert_eq!(executor.cache_stats().entries, 0);
}

#[test]
fn extension_executor_reports_missing_runtime() {
    let mut executor = ExtensionExecutor::<CpuBackend>::new();
    let mut backend = CpuBackend::new();
    let input = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]);
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
