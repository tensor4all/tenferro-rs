//! The contribution-owned QR factorization and the arithmetic behind it.
//!
//! The factorization runs in the external scalar, so its reconstruction and
//! orthogonality are limited by that precision rather than by `f64`.

use std::sync::Arc;

use tenferro_ad::extension::ExtensionOp;
use tenferro_cpu::CpuBackend;
use tenferro_df64_proof::extension::{Df64Qr, DF64_SCALAR_IDENTITY};
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

/// Factor one column-major matrix through the traced, compiled, and executed path,
/// which is the path a consumer uses.
fn factor(values: Vec<Df64>, rows: usize, columns: usize) -> (Vec<Df64>, Vec<Df64>) {
    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&backend).expect("engine"))
        .expect("register the CPU engine");
    builder
        .install_extension_module(tenferro_df64_proof::extension::module().expect("module"))
        .expect("install the Df64 module");
    let runtime = builder.build().expect("runtime with the module");

    // The traced leaf needs bound data to compile, and the values the runtime is given
    // below are what the factorization actually reads.
    let dtype = DType::External(std::any::TypeId::of::<Df64>());
    let traced_input = TracedTensor::from_tensor_concrete_shape_declaring_scalar(
        external(vec![Df64::zero(); rows * columns], vec![rows, columns]),
        DF64_SCALAR_IDENTITY,
    )
    .expect("traced input");
    assert_eq!(
        traced_input.rank, 2,
        "the traced input keeps the rank for {dtype:?}"
    );
    let outputs = apply(Arc::new(Df64Qr), &[&traced_input]).expect("traced QR");
    assert_eq!(<Df64Qr as ExtensionOp>::output_count(&Df64Qr), 2);
    assert_eq!(outputs.len(), 2);

    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_many(&[&outputs[0], &outputs[1]])
        .expect("compiled QR");
    let input = external(values, vec![rows, columns]);
    let results = runtime
        .run_compiled(&program, &[&input])
        .expect("executed QR");
    assert_eq!(results.len(), 2);
    let q = payload(&results[0]);
    let r = payload(&results[1]);
    assert_eq!(q.len(), rows * columns);
    assert_eq!(r.len(), columns * columns);
    (q, r)
}

#[test]
fn a_square_factorization_reconstructs_orthogonally_with_a_positive_diagonal() {
    let low = 2f64.powi(-80);
    let source = vec![
        Df64::from_f64(3.0),
        Df64::from_f64(4.0),
        Df64::from_f64(0.0),
        Df64::from_f64(4.0),
        Df64::from_f64(-3.0),
        Df64::from_f64(0.0),
        Df64::from_f64(0.0),
        Df64::from_f64(0.0),
        Df64::from_f64(1.0) + Df64::from_f64(low),
    ];
    let (q, r) = factor(source.clone(), 3, 3);

    // A = Q R in the external scalar, so a reconstruction error below the f64
    // resolution is only reachable when both factors keep the low components.
    let mut worst = 0.0_f64;
    for column in 0..3 {
        for row in 0..3 {
            let mut entry = Df64::zero();
            for inner in 0..3 {
                entry = entry + q[row + inner * 3] * r[inner + column * 3];
            }
            worst = worst.max((entry - source[row + column * 3]).abs_hi());
        }
    }
    assert!(
        worst < 1e-30,
        "reconstruction error {worst:e} is not better than f64"
    );

    for index in 0..3 {
        assert!(
            r[index + index * 3].hi > 0.0,
            "R[{index},{index}] is not positive"
        );
    }

    for first in 0..3 {
        for second in 0..3 {
            let mut inner = Df64::zero();
            for row in 0..3 {
                inner = inner + q[row + first * 3] * q[row + second * 3];
            }
            let expected = if first == second { 1.0 } else { 0.0 };
            assert!(
                (inner - Df64::from_f64(expected)).abs_hi() < 1e-30,
                "Q column inner product [{first},{second}] is {inner:?}"
            );
        }
    }
}

#[test]
fn the_single_column_case_returns_the_norm_and_a_unit_column() {
    // A = [[3], [4]] is the 3-4-5 triangle, so R = [[5]] and Q = [[0.6], [0.8]].
    let (q, r) = factor(vec![Df64::from_f64(3.0), Df64::from_f64(4.0)], 2, 1);
    assert_eq!(r[0].hi, 5.0, "R is {r:?}");
    assert_eq!(q[0].hi, 0.6, "Q is {q:?}");
    assert_eq!(q[1].hi, 0.8, "Q is {q:?}");
    // The column is a unit vector, which is the property the consumer relies on.
    assert!(
        (q[0] * q[0] + q[1] * q[1] - Df64::from_f64(1.0)).abs_hi() < 1e-30,
        "Q is {q:?}"
    );
}

#[test]
fn a_low_component_in_the_input_reaches_the_factors() {
    let low = 2f64.powi(-80);
    // The norm of [[3], [4 + 2^-80]] differs from 5 in its low component, so a
    // factorization that narrowed to f64 would return exactly 5.
    let (q, r) = factor(
        vec![
            Df64::from_f64(3.0),
            Df64::from_f64(4.0) + Df64::from_f64(low),
        ],
        2,
        1,
    );
    assert_ne!(r[0], Df64::from_f64(5.0), "R lost the low component");
    // R squared is 9 + (4 + 2^-80)^2, which only the extended scalar can hold.
    let second = Df64::from_f64(4.0) + Df64::from_f64(low);
    let expected = Df64::from_f64(9.0) + second * second;
    assert!((r[0] * r[0] - expected).abs_hi() < 1e-30, "R is {r:?}");
    // Q stays a unit column, which the low component must not disturb.
    let norm = q[0] * q[0] + q[1] * q[1];
    assert!((norm - Df64::from_f64(1.0)).abs_hi() < 1e-30, "Q is {q:?}");
}

#[test]
fn the_arithmetic_refines_beyond_f64_precision() {
    // Division and square root are what the factorization needs, and both carry a low
    // component that the single-component `f64` result cannot.
    let third = Df64::from_f64(1.0).ratio(Df64::from_f64(3.0));
    assert_ne!(third.lo, 0.0);
    assert!((third * Df64::from_f64(3.0) - Df64::from_f64(1.0)).abs_hi() < 1e-31);

    let root = Df64::from_f64(2.0).sqrt();
    assert_ne!(root.lo, 0.0);
    assert!((root * root - Df64::from_f64(2.0)).abs_hi() < 1e-31);

    // The control: the same computation in f64 cannot resolve the residual at all.
    let f64_residual = (1.0_f64 / 3.0) * 3.0 - 1.0;
    assert_eq!(f64_residual, 0.0, "f64 saw a residual it cannot represent");
}

#[test]
fn the_adjoint_of_the_factorization_matches_the_analytic_gradient() {
    use tenferro_df64_proof::extension::Df64QrVjp;

    // dL/dA for L = R[0,0]^2 at A = [[3], [4]] is 2 R A / |A| = [[6], [8]].
    let q = vec![
        Df64::from_f64(3.0).ratio(Df64::from_f64(5.0)),
        Df64::from_f64(4.0).ratio(Df64::from_f64(5.0)),
    ];
    let r = vec![Df64::from_f64(5.0)];
    let r_bar = vec![Df64::from_f64(10.0)];

    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&backend).expect("engine"))
        .expect("register the CPU engine");
    builder
        .install_extension_module(tenferro_df64_proof::extension::module().expect("module"))
        .expect("install the Df64 module");
    let runtime = builder.build().expect("runtime");

    let dtype = DType::External(std::any::TypeId::of::<Df64>());
    let q_leaf = TracedTensor::from_tensor_concrete_shape_declaring_scalar(
        external(q.clone(), vec![2, 1]),
        DF64_SCALAR_IDENTITY,
    )
    .expect("traced Q");
    let r_leaf = TracedTensor::from_tensor_concrete_shape_declaring_scalar(
        external(r, vec![1, 1]),
        DF64_SCALAR_IDENTITY,
    )
    .expect("traced R");
    let r_bar_leaf = TracedTensor::from_tensor_concrete_shape_declaring_scalar(
        external(r_bar, vec![1, 1]),
        DF64_SCALAR_IDENTITY,
    )
    .expect("traced cotangent");
    assert_eq!(q_leaf.dtype, dtype);

    // The adjoint is told that only the triangular factor carries a cotangent.
    let adjoint = apply(
        Arc::new(tenferro_df64_proof::extension::Df64QrVjp::of(false, true)),
        &[&q_leaf, &r_leaf, &r_bar_leaf],
    )
    .expect("traced adjoint");
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&adjoint[0]).expect("compiled adjoint");
    let results = runtime
        .run_compiled(&program, &[])
        .expect("executed adjoint");
    let gradient = payload(&results[0]);
    assert_eq!(gradient.len(), 2);
    assert!(
        (gradient[0] - Df64::from_f64(6.0)).abs_hi() < 1e-30,
        "the adjoint is {gradient:?}"
    );
    assert!(
        (gradient[1] - Df64::from_f64(8.0)).abs_hi() < 1e-30,
        "the adjoint is {gradient:?}"
    );
    assert_eq!(
        <Df64QrVjp as ExtensionOp>::input_count(&Df64QrVjp::of(false, true)),
        3
    );
}
