//! Clearing the accounted cache leaves live values alone.
//!
//! #1789 requires retention, clear, and statistics controls for custom storage, and states
//! that clearing free buffers must not recycle live values. The contribution's reuse path is
//! the runtime's accounted extension cache rather than a private one, so the runtime reports
//! it. This test fills it with two executions, keeps the second output live, clears the
//! runtime's caches, and checks three things: the statistics show the retained bytes were
//! actually released, the value that was live across the clear is byte for byte unchanged,
//! and a later execution still produces the same result.

use std::sync::Arc;

use tenferro_cpu::CpuBackend;
use tenferro_df64_proof::extension::{module, Df64QrVjp, DF64_SCALAR_IDENTITY};
use tenferro_df64_proof::Df64;
use tenferro_runtime::extension::apply;
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
use tenferro_tensor::Tensor;
use tenferro_tensor_core::{ErasedHostTensor, HostTensor};

fn external(values: Vec<Df64>, shape: Vec<usize>) -> Tensor {
    Tensor::external(ErasedHostTensor::new(
        HostTensor::from_vec_col_major(shape, values).expect("shape matches data"),
    ))
}

fn runtime() -> Runtime {
    let backend = CpuBackend::with_threads(1).expect("single-threaded CPU backend");
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&backend).expect("engine"))
        .expect("register the CPU engine");
    builder
        .install_extension_module(module().expect("module"))
        .expect("install the Df64 module");
    builder.build().expect("runtime with the module")
}

fn leaf(values: Vec<Df64>, shape: Vec<usize>) -> TracedTensor {
    TracedTensor::from_tensor_concrete_shape_declaring_scalar(
        external(values, shape),
        DF64_SCALAR_IDENTITY,
    )
    .expect("traced leaf")
}

fn payload(tensor: &Tensor) -> Vec<Df64> {
    match tensor.external_payload() {
        Some(value) => value
            .downcast_ref::<Df64>()
            .expect("external element type")
            .as_slice()
            .to_vec(),
        _ => panic!("expected an externally defined payload"),
    }
}

#[test]
fn clearing_the_accounted_cache_leaves_a_live_value_unchanged() {
    // The adjoint is the body that leaves scratch in the accounted cache, so it is the one
    // whose retention the clear has to release.
    let order = 8;
    let elements = order * order;
    let q_leaf = leaf(vec![Df64::from_f64(0.5); elements], vec![order, order]);
    let r_leaf = leaf(vec![Df64::from_f64(2.0); elements], vec![order, order]);
    let cotangent_leaf = leaf(vec![Df64::from_f64(1.0); elements], vec![order, order]);
    let adjoint = apply(
        Arc::new(Df64QrVjp::of(false, true)),
        &[&q_leaf, &r_leaf, &cotangent_leaf],
    )
    .expect("traced adjoint");

    let q_value = external(vec![Df64::from_f64(0.5); elements], vec![order, order]);
    let r_value = external(vec![Df64::from_f64(2.0); elements], vec![order, order]);
    let cotangent_value = external(vec![Df64::from_f64(1.0); elements], vec![order, order]);

    let runtime = runtime();
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&adjoint[0]).expect("compiled the adjoint");

    // The first execution leaves scratch behind; the second reuses it.
    drop(
        runtime
            .run_compiled(&program, &[&q_value, &r_value, &cotangent_value])
            .expect("first adjoint"),
    );
    let live = runtime
        .run_compiled(&program, &[&q_value, &r_value, &cotangent_value])
        .expect("second adjoint");
    let live_payload = payload(&live[0]);
    assert!(!live_payload.is_empty(), "the adjoint produced no output");

    let retained = runtime.cache_stats().expect("cache statistics");
    assert!(
        retained.extensions.entries > 0 && retained.extensions.retained_bytes > 0,
        "the reuse path retained nothing, so there is nothing to clear: {retained:?}"
    );

    // Clearing free buffers must not recycle live values.
    runtime.clear_caches().expect("clear the runtime caches");
    let cleared = runtime
        .cache_stats()
        .expect("cache statistics after the clear");
    assert_eq!(
        cleared.extensions.entries, 0,
        "the clear left extension cache entries behind"
    );
    assert_eq!(
        cleared.extensions.retained_bytes, 0,
        "the clear left retained bytes behind"
    );

    // The value that was live across the clear is unchanged, and the body still works.
    assert_eq!(
        payload(&live[0]),
        live_payload,
        "clearing the cache changed a value that was live across it"
    );
    let after = runtime
        .run_compiled(&program, &[&q_value, &r_value, &cotangent_value])
        .expect("adjoint after the clear");
    assert_eq!(
        payload(&after[0]),
        live_payload,
        "the body produces a different result after the cache was cleared"
    );
}
