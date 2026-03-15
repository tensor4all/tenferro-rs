use num_complex::{Complex32, Complex64};
use tenferro::{
    backward, forward_ad, grad, set_default_runtime, BackwardOptions, Error, GradOptions,
    RuntimeContext, Tensor,
};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

fn cpu_guard() -> impl Drop {
    set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)))
}

fn c32_vector(values: &[Complex32]) -> Tensor {
    Tensor::from_tensor(
        DenseTensor::<Complex32>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor)
            .unwrap(),
    )
}

fn c64_vector(values: &[Complex64]) -> Tensor {
    Tensor::from_tensor(
        DenseTensor::<Complex64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor)
            .unwrap(),
    )
}

#[test]
fn grad_and_backward_reject_create_graph() {
    let x = Tensor::from_slice(&[1.0_f64], &[1]).unwrap();

    let grad_err = match grad(
        &[&x],
        &[&x],
        None,
        GradOptions {
            create_graph: true,
            ..GradOptions::default()
        },
    ) {
        Ok(_) => panic!("grad(create_graph=true) should fail"),
        Err(err) => err,
    };
    assert!(matches!(
        grad_err,
        Error::UnsupportedAdOp {
            op: "grad(create_graph)"
        }
    ));

    let backward_err = match backward(
        &[&x],
        None,
        &[&x],
        BackwardOptions {
            create_graph: true,
            ..BackwardOptions::default()
        },
    ) {
        Ok(_) => panic!("backward(create_graph=true) should fail"),
        Err(err) => err,
    };
    assert!(matches!(
        backward_err,
        Error::UnsupportedAdOp {
            op: "backward(create_graph)"
        }
    ));
}

#[test]
fn grad_rejects_grad_output_length_mismatch() {
    let output = Tensor::from_slice(&[1.0_f64], &[1]).unwrap();
    let grad_output = Tensor::from_slice(&[1.0_f64], &[1]).unwrap();
    let extra = Tensor::from_slice(&[2.0_f64], &[1]).unwrap();

    let err = match grad(
        &[&output],
        &[&output],
        Some(&[grad_output, extra]),
        GradOptions::default(),
    ) {
        Ok(_) => panic!("grad should reject mismatched grad_outputs"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::InvalidAdTensor { .. }));
}

#[test]
fn grad_rejects_outputs_from_distinct_reverse_graphs() {
    let _guard = cpu_guard();

    let mut x = Tensor::from_slice(&[1.0_f64], &[1]).unwrap();
    let mut y = Tensor::from_slice(&[2.0_f64], &[1]).unwrap();
    x.set_requires_grad(true).unwrap();
    y.set_requires_grad(true).unwrap();

    let out_x = x.exp().unwrap().sum().unwrap();
    let out_y = y.exp().unwrap().sum().unwrap();

    let err = match grad(&[&out_x, &out_y], &[&x, &y], None, GradOptions::default()) {
        Ok(_) => panic!("grad should reject outputs from different reverse graphs"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::MixedReverseTape { .. }));
}

#[test]
fn grad_accumulates_multiple_outputs_on_the_same_graph() {
    let _guard = cpu_guard();

    let mut x = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
    x.set_requires_grad(true).unwrap();

    let out_exp = x.exp().unwrap().sum().unwrap();
    let out_double = x.add(&x).unwrap().sum().unwrap();
    let grads = grad(
        &[&out_exp, &out_double],
        &[&x],
        None,
        GradOptions::default(),
    )
    .unwrap();
    let grad = grads[0].as_ref().unwrap().as_f64().unwrap();
    let values = grad.primal().buffer().as_slice().unwrap();

    assert!((values[0] - (std::f64::consts::E + 2.0)).abs() < 1e-12);
    assert!((values[1] - (std::f64::consts::E.powi(2) + 2.0)).abs() < 1e-12);
}

#[test]
fn forward_dual_level_rejects_nested_scopes_and_recovers() {
    let _guard = cpu_guard();

    let nested_err = forward_ad::dual_level(|_| forward_ad::dual_level(|_| Ok(()))).unwrap_err();
    assert!(matches!(
        nested_err,
        Error::UnsupportedAdOp {
            op: "forward_ad::dual_level(nested)"
        }
    ));

    let x = Tensor::from_slice(&[1.0_f64], &[1]).unwrap();
    let dx = Tensor::from_slice(&[0.5_f64], &[1]).unwrap();
    let (_primal, tangent) = forward_ad::dual_level(|fw| {
        let dual = fw.make_dual(&x, &dx)?;
        fw.unpack_dual(&dual)
    })
    .unwrap();
    assert!(tangent.is_some());
}

#[test]
fn forward_make_dual_requires_matching_dtype_and_unpack_dual_handles_primals() {
    let x = Tensor::from_slice(&[1.0_f64], &[1]).unwrap();
    let wrong_dx = Tensor::from_slice(&[1.0_f32], &[1]).unwrap();

    forward_ad::dual_level(|fw| {
        let err = match fw.make_dual(&x, &wrong_dx) {
            Ok(_) => panic!("forward_ad::make_dual should reject dtype mismatch"),
            Err(err) => err,
        };
        assert!(matches!(err, Error::InvalidAdTensor { .. }));
        Ok(())
    })
    .unwrap();

    let (primal, tangent) = forward_ad::dual_level(|fw| fw.unpack_dual(&x)).unwrap();
    assert_eq!(primal.dims(), &[1]);
    assert!(tangent.is_none());
}

#[test]
fn forward_make_dual_and_unpack_dual_cover_all_runtime_dtypes() {
    let cases = [
        (
            Tensor::from_slice(&[1.0_f32, -2.0], &[2]).unwrap(),
            Tensor::from_slice(&[0.25_f32, 0.5], &[2]).unwrap(),
        ),
        (
            Tensor::from_slice(&[1.0_f64, -2.0], &[2]).unwrap(),
            Tensor::from_slice(&[0.25_f64, 0.5], &[2]).unwrap(),
        ),
        (
            c32_vector(&[Complex32::new(1.0, -1.0), Complex32::new(0.5, 2.0)]),
            c32_vector(&[Complex32::new(0.25, 0.0), Complex32::new(-0.5, 1.0)]),
        ),
        (
            c64_vector(&[Complex64::new(1.0, -1.0), Complex64::new(0.5, 2.0)]),
            c64_vector(&[Complex64::new(0.25, 0.0), Complex64::new(-0.5, 1.0)]),
        ),
    ];

    for (primal, tangent) in cases {
        let expected = primal.scalar_type();
        let (unpacked_primal, unpacked_tangent) = forward_ad::dual_level(|fw| {
            let dual = fw.make_dual(&primal, &tangent)?;
            fw.unpack_dual(&dual)
        })
        .unwrap();
        assert_eq!(unpacked_primal.scalar_type(), expected);
        assert_eq!(unpacked_tangent.unwrap().scalar_type(), expected);
    }
}
