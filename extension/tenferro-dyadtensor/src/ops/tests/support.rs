use crate::runtime::dispatch::{unsupported_runtime_capability, with_runtime};
use crate::Result;
use crate::{AdMode, AdTensor, StructuredTensor};
use chainrules::Tape;
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
    assert_eq!(t.mode(), AdMode::Primal);
    assert!(t.tangent().is_none());
    assert!(t.node_id().is_none());
}

pub(crate) fn assert_forward_mode<T: tenferro_algebra::Scalar>(t: &AdTensor<T>) {
    assert_eq!(t.mode(), AdMode::Forward);
    assert!(t.tangent().is_some());
    assert!(t.node_id().is_none());
}

pub(crate) fn reverse_leaf_f64(
    tensor: impl Into<StructuredTensor<f64>>,
    tape: &Tape<crate::DynTensor>,
) -> AdTensor<f64> {
    AdTensor::new_reverse_leaf(tensor, tape).unwrap()
}

pub(crate) fn assert_reverse_on_tape<T: tenferro_algebra::Scalar>(
    tensor: &AdTensor<T>,
    tape: &Tape<crate::DynTensor>,
) {
    assert_eq!(tensor.mode(), AdMode::Reverse);
    assert!(tensor.node_id().is_some());
    let actual_tape = tensor
        .tape()
        .expect("reverse tensor should expose its tape");
    assert!(actual_tape.same_tape(tape));
}
