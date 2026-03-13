use crate::runtime::dispatch::{unsupported_runtime_capability, with_runtime};
use crate::Result;
use crate::{AdTensor, AdValue};
use tenferro_prims::CpuContext;
use tenferro_tensor::Tensor;

pub(crate) fn with_cpu_runtime<R>(
    op: &'static str,
    f: impl FnOnce(&mut CpuContext) -> Result<R>,
) -> Result<R> {
    with_runtime(
        f,
        |_ctx| Err(unsupported_runtime_capability(op, "cuda")),
        |_ctx| Err(unsupported_runtime_capability(op, "rocm")),
    )
}

pub(crate) fn as_slice<T: tenferro_algebra::Scalar>(t: &Tensor<T>) -> &[T] {
    t.buffer()
        .as_slice()
        .unwrap_or_else(|| panic!("expected CPU-backed contiguous tensor"))
}

pub(crate) fn assert_primal_mode(t: &AdTensor<f64>) {
    assert!(matches!(t.as_value(), AdValue::Primal(_)));
}
