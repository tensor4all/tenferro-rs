use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::backend::TensorBackend;
use crate::config::{CompareDir, GatherConfig, PadConfig, ScatterConfig, SliceConfig};
use crate::cpu::CpuBackend;
use crate::types::{Tensor, TypedTensor};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(shape, data))
}

fn assert_panics(label: &str, f: impl FnOnce()) {
    let result = catch_unwind(AssertUnwindSafe(f));
    assert!(result.is_err(), "{label} should panic until implemented");
}

#[test]
fn cpu_backend_remaining_unimplemented_elementwise_and_reduction_ops_panic_explicitly() {
    let mut backend = CpuBackend::new();
    let a = f64_tensor(vec![2], vec![1.0, 2.0]);
    let b = f64_tensor(vec![2], vec![3.0, 4.0]);

    assert_panics("div", || {
        let _ = backend.div(&a, &b);
    });
    assert_panics("abs", || {
        let _ = backend.abs(&a);
    });
    assert_panics("sign", || {
        let _ = backend.sign(&a);
    });
    assert_panics("maximum", || {
        let _ = backend.maximum(&a, &b);
    });
    assert_panics("minimum", || {
        let _ = backend.minimum(&a, &b);
    });
    assert_panics("compare", || {
        let _ = backend.compare(&a, &b, &CompareDir::Eq);
    });
    assert_panics("select", || {
        let _ = backend.select(&a, &b, &a);
    });
    assert_panics("clamp", || {
        let _ = backend.clamp(&a, &b, &a);
    });
    assert_panics("reduce_prod", || {
        let _ = backend.reduce_prod(&a, &[0]);
    });
    assert_panics("reduce_max", || {
        let _ = backend.reduce_max(&a, &[0]);
    });
    assert_panics("reduce_min", || {
        let _ = backend.reduce_min(&a, &[0]);
    });
}

#[test]
fn cpu_backend_unimplemented_indexing_and_linalg_ops_panic_explicitly() {
    let mut backend = CpuBackend::new();
    let a = f64_tensor(vec![2], vec![1.0, 2.0]);
    let b = f64_tensor(vec![2], vec![3.0, 4.0]);

    assert_panics("gather", || {
        let _ = backend.gather(&a, &GatherConfig {});
    });
    assert_panics("scatter", || {
        let _ = backend.scatter(&a, &b, &ScatterConfig {});
    });
    assert_panics("slice", || {
        let _ = backend.slice(
            &a,
            &SliceConfig {
                starts: vec![0],
                limits: vec![1],
                strides: vec![1],
            },
        );
    });
    assert_panics("dynamic_slice", || {
        let _ = backend.dynamic_slice(&a, &b);
    });
    assert_panics("pad", || {
        let _ = backend.pad(&a, &PadConfig {});
    });
    assert_panics("concatenate", || {
        let _ = backend.concatenate(&[&a, &b], 0);
    });
    assert_panics("reverse", || {
        let _ = backend.reverse(&a, &[0]);
    });
    assert_panics("cholesky", || {
        let _ = backend.cholesky(&a);
    });
    assert_panics("svd", || {
        let _ = backend.svd(&a);
    });
    assert_panics("qr", || {
        let _ = backend.qr(&a);
    });
    assert_panics("eigh", || {
        let _ = backend.eigh(&a);
    });
    assert_panics("solve", || {
        let _ = backend.solve(&a, &b);
    });
}

#[cfg(feature = "cpu-faer")]
#[test]
fn cpu_faer_linalg_stub_panics_explicitly() {
    assert_panics("faer_linalg::unavailable", || {
        crate::cpu::linalg::faer_linalg::unavailable();
    });
}
