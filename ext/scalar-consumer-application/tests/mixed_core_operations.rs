//! The storage and core operations of #1785 in the mixed configuration.
//!
//! #1785 asks for the extended-precision example, the reduction, and both conversions to run in
//! a Df64-only configuration *and* in a mixed one. `ext/df64-proof` covers them where they live,
//! and the application's `configurations` tests run the connected program in a runtime that
//! carries the canonical support beside the contribution. This test closes the remaining
//! reading: the core operations themselves run in that mixed runtime, on a value that the
//! runtime produced, with the standard path still available in the same process.

use std::sync::Arc;

use tenferro_cpu::{scalar_binary_into, scalar_fold, AddOp, CpuBackend};
use tenferro_df64_proof::extension::{module, Df64Total, DF64_SCALAR_IDENTITY};
use tenferro_df64_proof::{Df64, Df64Add};
use tenferro_runtime::extension::apply;
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
use tenferro_tensor::Tensor;
use tenferro_tensor_core::{ErasedHostTensor, HostTensor};

/// A runtime carrying the standard engine and the contribution's module together.
fn mixed_runtime() -> Runtime {
    let backend = CpuBackend::with_threads(1).expect("single-threaded CPU backend");
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&backend).expect("engine"))
        .expect("register the canonical CPU engine");
    builder
        .install_extension_module(module().expect("module"))
        .expect("install the Df64 module");
    builder.build().expect("runtime with both supports")
}

fn external(values: Vec<Df64>, shape: Vec<usize>) -> Tensor {
    Tensor::external(ErasedHostTensor::new(
        HostTensor::from_vec_col_major(shape, values).expect("shape matches data"),
    ))
}

fn payload(tensor: &Tensor) -> Vec<Df64> {
    match tensor.external_payload() {
        Some(value) => value
            .downcast_ref::<Df64>()
            .expect("external element type")
            .as_slice()
            .to_vec(),
        None => panic!("expected an externally defined payload"),
    }
}

#[test]
fn the_core_operations_run_in_the_mixed_configuration() {
    let low = 2f64.powi(-80);
    let runtime = mixed_runtime();

    // A value the runtime itself produces, so the operations below act on a result of the mixed
    // configuration rather than on a hand-built tensor.
    let input = TracedTensor::from_tensor_concrete_shape_declaring_scalar(
        external(
            vec![Df64::from_f64(1.0), Df64 { hi: low, lo: 0.0 }],
            vec![2],
        ),
        DF64_SCALAR_IDENTITY,
    )
    .expect("traced leaf");
    let total = apply(Arc::new(Df64Total), &[&input]).expect("traced total");
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&total[0]).expect("compiled total");
    let produced = runtime
        .run_compiled(
            &program,
            &[&external(
                vec![Df64::from_f64(1.0), Df64 { hi: low, lo: 0.0 }],
                vec![2],
            )],
        )
        .expect("execution in the mixed runtime");
    let value = payload(&produced[0])[0];
    assert_eq!(
        value,
        Df64 { hi: 1.0, lo: low },
        "the mixed runtime did not produce the extended-precision value"
    );

    // The precision example: add the low component, then subtract the leading one.
    let mut sum = HostTensor::from_vec_col_major(vec![1], vec![Df64::from_f64(1.0)])
        .expect("shape matches data");
    let increment = HostTensor::from_vec_col_major(vec![1], vec![Df64 { hi: low, lo: 0.0 }])
        .expect("shape matches data");
    let leading = sum.clone();
    scalar_binary_into::<Df64, Df64Add>("add", &mut sum, &leading, &increment)
        .expect("addition in the extended scalar");
    let difference = sum.as_slice()[0] - Df64::from_f64(1.0);
    assert_eq!(difference, Df64 { hi: low, lo: 0.0 });
    assert_eq!(difference.narrow_to_f64(), low);

    // The reduction keeps the low component through the shared fold.
    let values = HostTensor::from_vec_col_major(
        vec![2],
        vec![Df64::from_f64(1.0), Df64 { hi: low, lo: 0.0 }],
    )
    .expect("shape matches data");
    let reduced = scalar_fold::<Df64, Df64Add>("sum", &values, Df64::zero())
        .expect("reduction in the extended scalar");
    assert_eq!(reduced - Df64::from_f64(1.0), Df64 { hi: low, lo: 0.0 });

    // The canonical path is still present and still ordinary in the same process.
    let ordinary = HostTensor::from_vec_col_major(vec![2], vec![1.0_f64, low]).expect("shape");
    let ordinary_total = scalar_fold::<f64, AddOp>("sum", &ordinary, 0.0_f64).expect("f64 sum");
    assert_eq!(
        ordinary_total - 1.0,
        0.0,
        "the standard path must keep its ordinary f64 behavior"
    );
}
