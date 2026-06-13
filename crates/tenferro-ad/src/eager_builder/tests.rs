use std::sync::Arc;

use tenferro_cpu::CpuBackend;
use tenferro_tensor::Tensor;

use crate::extension_runtime::ExtensionExecutor;

use super::EagerPrimitiveBuilder;

#[test]
fn debug_summarizes_builder_without_tensor_payloads() {
    let mut backend = CpuBackend::new();
    let mut builder = EagerPrimitiveBuilder::new(&mut backend);
    let id = builder.push_tensor(Arc::new(Tensor::from_vec_col_major(vec![1], vec![1.0_f64])));
    let _tensor = builder.tensor(id);

    let debug = format!("{builder:?}");

    assert!(debug.contains("EagerPrimitiveBuilder"));
    assert!(debug.contains("backend_type"));
    assert!(debug.contains("has_extension_executor: false"));
    assert!(debug.contains("results_len: 1"));
}

#[test]
fn debug_reports_extension_executor_presence() {
    let mut backend = CpuBackend::new();
    let mut executor = ExtensionExecutor::<CpuBackend>::new();
    let builder = EagerPrimitiveBuilder::with_extension_executor(&mut backend, &mut executor);

    let debug = format!("{builder:?}");

    assert!(debug.contains("has_extension_executor: true"));
}
