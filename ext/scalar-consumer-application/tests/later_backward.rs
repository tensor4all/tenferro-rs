//! #1790's "later backward" checkpoint, survival half.
//!
//! A forward program's factors must outlive the runtime that produced them and still serve
//! a later program. The forward runtime is dropped before the later program is built, so
//! nothing can recompute them, and the test writes to the retained storage to show that the
//! later program reads those factors rather than a fresh factorization.

use std::sync::Arc;

use tenferro_ad::semantic_extension::SemanticExtensionRuleSet;
use tenferro_cpu::CpuBackend;
use tenferro_df64_proof::ad::Df64VjpRule;
use tenferro_df64_proof::extension::{module, Df64Qr, Df64QrVjp, DF64_SCALAR_IDENTITY};
use tenferro_df64_proof::Df64;
use tenferro_runtime::extension::apply;
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
use tenferro_tensor::{DType, Tensor};
use tenferro_tensor_core::{ErasedHostTensor, HostTensor};

fn external(values: Vec<Df64>, shape: Vec<usize>) -> Tensor {
    Tensor::external(ErasedHostTensor::new(
        HostTensor::from_vec_col_major(shape, values).expect("shape matches data"),
    ))
}

fn payload(tensor: &Tensor) -> Vec<Df64> {
    match tensor {
        Tensor::External(value, _) => value
            .downcast_ref::<Df64>()
            .expect("external element type")
            .as_slice()
            .to_vec(),
        other => panic!(
            "expected an externally defined payload, found {:?}",
            other.dtype()
        ),
    }
}

/// A runtime that composes the CPU engine with the contribution's module.
fn runtime() -> Runtime {
    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&backend).expect("engine"))
        .expect("register the CPU engine");
    builder
        .install_extension_module(module().expect("module"))
        .expect("install the Df64 module");
    builder.build().expect("runtime with the module")
}

/// Overwrite one element of a caller-owned payload, which only the storage's owner can do.
fn overwrite(tensor: &mut Tensor, index: usize, value: Df64) {
    match tensor {
        Tensor::External(payload, _) => {
            payload
                .downcast_mut::<Df64>()
                .expect("external element type")
                .as_mut_slice()[index] = value;
        }
        other => panic!(
            "expected an externally defined payload, found {:?}",
            other.dtype()
        ),
    }
}

#[test]
fn the_factors_outlive_the_forward_runtime_and_still_serve_a_later_program() {
    // 1. The forward program runs in its own runtime and its factors leave with it.
    let forward_runtime = runtime();
    let input = TracedTensor::from_tensor_concrete_shape_declaring_scalar(
        external(vec![Df64::from_f64(3.0), Df64::from_f64(4.0)], vec![2, 1]),
        DF64_SCALAR_IDENTITY,
    )
    .expect("traced input");
    let outputs = apply(Arc::new(Df64Qr), &[&input]).expect("traced QR");

    let mut compiler = GraphCompiler::new();
    let forward = compiler
        .compile_many(&[&outputs[0], &outputs[1]])
        .expect("compiled QR");
    let mut factors = forward_runtime
        .run_compiled(&forward, &[])
        .expect("executed QR")
        .into_iter();
    let mut q = factors.next().expect("the factor");
    let r = factors.next().expect("the triangular factor");
    assert_eq!(payload(&r)[0].hi, 5.0);

    // The forward session ends: its runtime and engine handles are gone.
    drop(forward_runtime);

    // 2. The retained factors are still the caller's storage, so a write to one of them is
    // what the later program reads.
    overwrite(&mut q, 0, Df64::from_f64(0.3));
    overwrite(&mut q, 1, Df64::from_f64(0.4));

    // 3. A later program consumes the retained factors in a new runtime. The adjoint is
    // `Q (R R_bar^T) R^{-T}`, which for one column is `Q R_bar`, so the retained factor's
    // written values are exactly what the later program must produce.
    let q_leaf = TracedTensor::from_tensor_concrete_shape_declaring_scalar(q, DF64_SCALAR_IDENTITY)
        .expect("traced factor");
    let r_leaf = TracedTensor::from_tensor_concrete_shape_declaring_scalar(r, DF64_SCALAR_IDENTITY)
        .expect("traced triangular factor");
    let cotangent = TracedTensor::from_tensor_concrete_shape_declaring_scalar(
        external(vec![Df64::from_f64(1.0)], vec![1, 1]),
        DF64_SCALAR_IDENTITY,
    )
    .expect("traced cotangent");
    let adjoint = apply(
        Arc::new(Df64QrVjp::of(false, true)),
        &[&q_leaf, &r_leaf, &cotangent],
    )
    .expect("traced adjoint");

    let later_runtime = runtime();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile(&adjoint[0])
        .expect("compiled the later program");
    let results = later_runtime
        .run_compiled(&program, &[])
        .expect("executed the later program");
    let gradient = payload(&results[0]);
    assert_eq!(gradient.len(), 2);
    assert!(
        (gradient[0] - Df64::from_f64(0.3)).abs_hi() < 1e-30,
        "the later program did not read the retained factor: {gradient:?}"
    );
    assert!(
        (gradient[1] - Df64::from_f64(0.4)).abs_hi() < 1e-30,
        "{gradient:?}"
    );

    // The extension's AD rules are what a consumer installs for the later backward pass.
    let rules = SemanticExtensionRuleSet::new()
        .with_primal_vjp(Arc::new(Df64VjpRule))
        .expect("one rule per family");
    assert!(rules
        .lookup_primal_vjp("tenferro-df64-proof.df64_ops.v1")
        .is_some());
}

#[test]
fn a_retained_value_survives_an_eager_session_it_was_computed_in() {
    // The eager path retains the same caller-owned storage, so a value computed inside an
    // admitted session is still usable after the session borrow ends.
    use tenferro_ad::EagerRuntime;
    use tenferro_tensor::BackendSessionHost;

    let mut backend = CpuBackend::new();
    let retained = backend.with_backend_session(|session| {
        let values = external(vec![Df64::from_f64(1.0), Df64::from_f64(2.0)], vec![2]);
        let copied = session
            .to_contiguous_read(tenferro_tensor::TensorRead::from_tensor(&values))
            .expect("a contiguous read inside the session");
        assert_eq!(
            copied.dtype(),
            DType::External(std::any::TypeId::of::<Df64>())
        );
        copied
    });
    assert_eq!(
        payload(&retained),
        vec![Df64::from_f64(1.0), Df64::from_f64(2.0)]
    );

    // And the eager runtime's own retention keeps an external payload alive the same way.
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).expect("cpu runtime");
    let leaf = tenferro_ad::EagerTensor::from_tensor_in(
        external(vec![Df64::from_f64(7.0)], vec![1]),
        Arc::clone(&runtime),
    )
    .expect("eager tensor");
    drop(runtime);
    assert_eq!(
        payload(&leaf.to_tensor().expect("round trip")),
        vec![Df64::from_f64(7.0)]
    );
}
