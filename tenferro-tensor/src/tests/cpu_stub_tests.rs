use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::backend::TensorBackend;
use crate::config::{GatherConfig, PadConfig, ScatterConfig};
use crate::cpu::CpuBackend;
use crate::{Tensor, TypedTensor};

fn dummy() -> Tensor {
    Tensor::F64(TypedTensor::from_vec(vec![2], vec![1.0, 2.0]))
}

fn dummy2() -> (Tensor, Tensor) {
    (dummy(), dummy())
}

macro_rules! assert_panics {
    ($name:expr, $body:expr) => {
        let result = catch_unwind(AssertUnwindSafe($body));
        assert!(
            result.is_err(),
            "{} should panic (not yet implemented)",
            $name
        );
    };
}

#[test]
fn cpu_backend_unimplemented_indexing_ops_panic_explicitly() {
    let mut b = CpuBackend::new();
    let d = dummy();
    let (d1, d2) = dummy2();

    assert_panics!("gather", || b.gather(&d, &GatherConfig {}));
    assert_panics!("scatter", || b.scatter(&d1, &d2, &ScatterConfig {}));
    assert_panics!("dynamic_slice", || b.dynamic_slice(&d1, &d2));
    assert_panics!("pad", || b.pad(&d, &PadConfig {}));
}
