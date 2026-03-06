use num_complex::Complex64;
use tenferro_dyadtensor::{
    ad, AdTensor, DynAdScalar, DynAdTensor, NodeId, StructuredTensor, TapeId,
};
use tenferro_dyadtensor::{set_default_runtime, RuntimeContext};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor};

fn vector(values: &[f64]) -> Tensor<f64> {
    Tensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn c64_vector(values: &[Complex64]) -> Tensor<Complex64> {
    Tensor::<Complex64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn scalar(value: f64) -> Tensor<f64> {
    Tensor::<f64>::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn diag_scale_reverse_keeps_diag_cotangent_space() {
    let x: DynAdTensor = AdTensor::new_reverse(
        StructuredTensor::from_diagonal_vector(vector(&[2.0, 3.0]), 2).unwrap(),
        NodeId(1),
        TapeId(7),
        None::<StructuredTensor<f64>>,
    )
    .unwrap()
    .into();
    let a = DynAdScalar::from(2.0_f64);
    let y = x.scale(&a).unwrap();
    let cotangent = AdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector(&[1.0, 1.0]), 2).unwrap(),
    );

    let grads = ad::pullback_wrt(y.as_f64().unwrap(), &cotangent, &[x.as_f64().unwrap()]).unwrap();
    let grad = grads[0].as_ref().unwrap();
    assert!(grad.is_diag());
    assert_eq!(grad.payload().dims(), &[2]);
}

#[test]
fn diag_axpby_reverse_keeps_diag_cotangent_space() {
    let x: DynAdTensor = AdTensor::new_reverse(
        StructuredTensor::from_diagonal_vector(vector(&[2.0, 3.0]), 2).unwrap(),
        NodeId(11),
        TapeId(17),
        None::<StructuredTensor<f64>>,
    )
    .unwrap()
    .into();
    let y: DynAdTensor = AdTensor::new_reverse(
        StructuredTensor::from_diagonal_vector(vector(&[5.0, 7.0]), 2).unwrap(),
        NodeId(12),
        TapeId(17),
        None::<StructuredTensor<f64>>,
    )
    .unwrap()
    .into();
    let a = DynAdScalar::from(2.0_f64);
    let b = DynAdScalar::from(-1.0_f64);
    let out = x.axpby(&a, &y, &b).unwrap();
    let cotangent = AdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector(&[1.0, -0.5]), 2).unwrap(),
    );

    let grads = ad::pullback_wrt(
        out.as_f64().unwrap(),
        &cotangent,
        &[x.as_f64().unwrap(), y.as_f64().unwrap()],
    )
    .unwrap();

    assert!(grads[0].as_ref().unwrap().is_diag());
    assert!(grads[1].as_ref().unwrap().is_diag());
    assert_eq!(grads[0].as_ref().unwrap().payload().dims(), &[2]);
    assert_eq!(grads[1].as_ref().unwrap().payload().dims(), &[2]);
}

#[test]
fn diag_complex_real_part_reverse_keeps_diag_cotangent_space() {
    let x: DynAdTensor = AdTensor::new_reverse(
        StructuredTensor::from_diagonal_vector(
            c64_vector(&[Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)]),
            2,
        )
        .unwrap(),
        NodeId(21),
        TapeId(27),
        None::<StructuredTensor<Complex64>>,
    )
    .unwrap()
    .into();

    let out = x.real_part().unwrap();
    let cotangent = AdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector(&[0.5, -1.25]), 2).unwrap(),
    );
    let grads =
        ad::pullback_wrt_mixed(out.as_f64().unwrap(), &cotangent, &[x.as_c64().unwrap()]).unwrap();

    let grad = grads[0].as_ref().unwrap();
    assert!(grad.is_diag());
    assert_eq!(grad.payload().dims(), &[2]);
}

#[test]
fn diag_complex_compose_complex_reverse_keeps_diag_cotangent_space() {
    let re: DynAdTensor = AdTensor::new_reverse(
        StructuredTensor::from_diagonal_vector(vector(&[1.0, -3.0]), 2).unwrap(),
        NodeId(31),
        TapeId(37),
        None::<StructuredTensor<f64>>,
    )
    .unwrap()
    .into();
    let im: DynAdTensor = AdTensor::new_reverse(
        StructuredTensor::from_diagonal_vector(vector(&[2.0, 4.0]), 2).unwrap(),
        NodeId(32),
        TapeId(37),
        None::<StructuredTensor<f64>>,
    )
    .unwrap()
    .into();

    let out = DynAdTensor::compose_complex(re.clone(), im.clone()).unwrap();
    let cotangent = AdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(
            c64_vector(&[Complex64::new(0.5, -0.25), Complex64::new(1.0, 0.75)]),
            2,
        )
        .unwrap(),
    );
    let grads = ad::pullback_wrt_mixed(
        out.as_c64().unwrap(),
        &cotangent,
        &[re.as_f64().unwrap(), im.as_f64().unwrap()],
    )
    .unwrap();

    assert!(grads[0].as_ref().unwrap().is_diag());
    assert!(grads[1].as_ref().unwrap().is_diag());
    assert_eq!(grads[0].as_ref().unwrap().payload().dims(), &[2]);
    assert_eq!(grads[1].as_ref().unwrap().payload().dims(), &[2]);
}

#[test]
fn root_einsum_keeps_diag_output_in_structured_carrier() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let a = AdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap(),
    );
    let b = AdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector(&[3.0, 4.0]), 2).unwrap(),
    );

    let out = ad::einsum("ij,jk->ik", &[&a, &b]).unwrap();

    assert!(out.is_diag());
    assert_eq!(out.primal().dims(), &[2]);
    assert_eq!(out.dims(), &[2, 2]);
}

#[test]
fn root_einsum_reverse_keeps_diag_cotangent_space() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let a = AdTensor::new_reverse(
        StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap(),
        NodeId(41),
        TapeId(47),
        None,
    )
    .unwrap();
    let b = AdTensor::new_reverse(
        StructuredTensor::from_diagonal_vector(vector(&[3.0, 4.0]), 2).unwrap(),
        NodeId(42),
        TapeId(47),
        None,
    )
    .unwrap();

    let out = ad::einsum("ij,jk->ik", &[&a, &b]).unwrap();
    let cotangent = AdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector(&[0.5, -1.0]), 2).unwrap(),
    );
    let grads = ad::pullback_wrt(&out, &cotangent, &[&a, &b]).unwrap();

    assert!(out.is_diag());
    assert!(grads[0].as_ref().unwrap().is_diag());
    assert!(grads[1].as_ref().unwrap().is_diag());
    assert_eq!(grads[0].as_ref().unwrap().payload().dims(), &[2]);
    assert_eq!(grads[1].as_ref().unwrap().payload().dims(), &[2]);
}

#[test]
fn root_sum_reverse_keeps_diag_cotangent_space() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let x = AdTensor::new_reverse(
        StructuredTensor::from_diagonal_vector(vector(&[2.0, 3.0]), 2).unwrap(),
        NodeId(51),
        TapeId(57),
        None,
    )
    .unwrap();

    let out = ad::sum(&x).unwrap();
    let cotangent = AdTensor::new_primal(scalar(1.5));
    let grads = ad::pullback_wrt(&out, &cotangent, &[&x]).unwrap();

    assert_eq!(out.dims(), &[]);
    assert!(grads[0].as_ref().unwrap().is_diag());
    assert_eq!(grads[0].as_ref().unwrap().payload().dims(), &[2]);
}
