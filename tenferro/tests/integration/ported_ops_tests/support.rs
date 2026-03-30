use crate::core::AdMode;
use crate::runtime::dispatch::{unsupported_runtime_capability, with_runtime};
use crate::structured::StructuredTensor;
use crate::AdTensor;
use crate::Result;
use num_complex::Complex64;
use tenferro_internal_ad_core::DynAdTensor;
use tenferro_prims::CpuContext;
use tenferro_tensor::Tensor as DenseTensor;
use tidu::expert::Tape;

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

pub(crate) fn as_slice<T: tenferro_algebra::Scalar>(t: &DenseTensor<T>) -> &[T] {
    t.buffer()
        .as_slice()
        .unwrap_or_else(|| panic!("expected CPU-backed contiguous tensor"))
}

pub(crate) trait AdTensorLike {
    fn mode(&self) -> AdMode;
    fn has_tangent(&self) -> bool;
    fn node_id(&self) -> Option<crate::NodeId>;
    fn tape(&self) -> Option<Tape<crate::DynTensor>>;
}

impl<T: tenferro_algebra::Scalar> AdTensorLike for AdTensor<T> {
    fn mode(&self) -> AdMode {
        AdTensor::mode(self)
    }

    fn has_tangent(&self) -> bool {
        AdTensor::tangent(self).is_some()
    }

    fn node_id(&self) -> Option<crate::NodeId> {
        AdTensor::node_id(self)
    }

    fn tape(&self) -> Option<Tape<crate::DynTensor>> {
        AdTensor::tape(self)
    }
}

impl AdTensorLike for DynAdTensor {
    fn mode(&self) -> AdMode {
        DynAdTensor::mode(self)
    }

    fn has_tangent(&self) -> bool {
        DynAdTensor::has_tangent(self)
    }

    fn node_id(&self) -> Option<crate::NodeId> {
        DynAdTensor::node_id(self)
    }

    fn tape(&self) -> Option<Tape<crate::DynTensor>> {
        DynAdTensor::tape(self)
    }
}

pub(crate) fn assert_primal_mode<T: AdTensorLike>(t: &T) {
    assert_eq!(t.mode(), AdMode::Primal);
    assert!(!t.has_tangent());
    assert!(t.node_id().is_none());
}

pub(crate) fn assert_forward_mode<T: AdTensorLike>(t: &T) {
    assert_eq!(t.mode(), AdMode::Forward);
    assert!(t.has_tangent());
    assert!(t.node_id().is_none());
}

pub(crate) fn reverse_leaf_f64(
    tensor: impl Into<StructuredTensor<f64>>,
    tape: &Tape<crate::DynTensor>,
) -> AdTensor<f64> {
    AdTensor::new_reverse_leaf(tensor, tape).unwrap()
}

pub(crate) fn reverse_leaf_c64(
    tensor: impl Into<StructuredTensor<Complex64>>,
    tape: &Tape<crate::DynTensor>,
) -> AdTensor<Complex64> {
    AdTensor::new_reverse_leaf(tensor, tape).unwrap()
}

pub(crate) fn expect_dyn_f64(tensor: &DynAdTensor) -> &AdTensor<f64> {
    tensor.as_f64().expect("expected f64 dyn AD tensor in test")
}

pub(crate) fn assert_reverse_on_tape<T: AdTensorLike>(tensor: &T, tape: &Tape<crate::DynTensor>) {
    assert_eq!(tensor.mode(), AdMode::Reverse);
    assert!(tensor.node_id().is_some());
    let actual_tape = tensor
        .tape()
        .expect("reverse tensor should expose its tape");
    assert!(actual_tape.same_tape(tape));
}
