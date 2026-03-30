use num_complex::Complex64;
use tenferro::{
    backward, forward_ad, set_default_runtime, BackwardOptions, RuntimeContext, ScalarType,
    SvdResult, Tensor,
};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

fn scalar_f64(value: f64) -> DenseTensor<f64> {
    DenseTensor::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

fn vector_f64(values: &[f64]) -> DenseTensor<f64> {
    DenseTensor::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn diag_f64(values: &[f64]) -> Tensor {
    Tensor::diag(&Tensor::from_tensor(vector_f64(values))).unwrap()
}

#[test]
fn tensor_public_constructor_handles_dense_and_diag() {
    let dense = Tensor::from_tensor(vector_f64(&[1.0, 2.0]));
    assert!(dense.is_dense());
    assert_eq!(dense.dims(), &[2]);

    let diag = diag_f64(&[3.0, 4.0]);
    assert!(diag.is_diag());
    assert_eq!(diag.dims(), &[2, 2]);
}

#[test]
fn tensor_scalar_semantics_use_rank0_tensor_and_casts() {
    let x = Tensor::from_tensor(scalar_f64(2.0));
    let alpha = Tensor::from_tensor(
        DenseTensor::from_slice(&[Complex64::new(0.0, 3.0)], &[], MemoryOrder::ColumnMajor)
            .unwrap(),
    );

    let y = x.scale(&alpha).unwrap();
    assert!(y.dims().is_empty());
    assert_eq!(y.scalar_type(), ScalarType::C64);

    let cast = y.to_scalar_type(ScalarType::F64).unwrap();
    assert_eq!(cast.scalar_type(), ScalarType::F64);
}

#[test]
fn tensor_linalg_results_use_non_dyn_names() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let x = Tensor::from_tensor(
        DenseTensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap(),
    );

    let out: SvdResult = x.svd().unwrap();
    assert_eq!(out.s.dims(), &[2]);
}

#[test]
fn tensor_reverse_api_accumulates_gradients_on_requested_inputs() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let mut x = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
    let mut y = Tensor::from_slice(&[3.0_f64, 4.0], &[2]).unwrap();
    x.set_requires_grad(true).unwrap();
    y.set_requires_grad(true).unwrap();

    let out = x.add(&y).unwrap().sum().unwrap();
    backward(&[&out], None, &[&x, &y], BackwardOptions::default()).unwrap();

    let gx = x.grad().unwrap().unwrap();
    let gy = y.grad().unwrap().unwrap();
    assert_eq!(gx.dims(), &[2]);
    assert_eq!(gy.dims(), &[2]);
    assert_eq!(
        gx.as_f64().unwrap().primal().buffer().as_slice().unwrap(),
        &[1.0, 1.0]
    );
    assert_eq!(
        gy.as_f64().unwrap().primal().buffer().as_slice().unwrap(),
        &[1.0, 1.0]
    );
}

#[test]
fn tensor_forward_api_uses_scoped_dual_level() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let x = Tensor::from_slice(&[0.0_f64, 1.0], &[2]).unwrap();
    let dx = Tensor::from_slice(&[0.5_f64, -0.25], &[2]).unwrap();

    let (primal, tangent) = forward_ad::dual_level(|fw| {
        let dual = fw.make_dual(&x, &dx)?;
        let out = dual.exp()?;
        fw.unpack_dual(&out)
    })
    .unwrap();

    assert_eq!(primal.dims(), &[2]);
    let tangent = tangent.unwrap();
    let tangent_values = tangent
        .as_f64()
        .unwrap()
        .primal()
        .buffer()
        .as_slice()
        .unwrap();
    assert!((tangent_values[0] - 0.5).abs() < 1e-12);
    assert!((tangent_values[1] + 0.25 * std::f64::consts::E).abs() < 1e-12);
}
