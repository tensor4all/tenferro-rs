use super::*;
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

fn vector_f64(values: &[f64]) -> DenseTensor<f64> {
    DenseTensor::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn dyn_helpers_dispatch_on_dyn_refs() {
    let x = Tensor::from_tensor(vector_f64(&[1.0, 2.0]));
    let alpha = Tensor::from_tensor(
        DenseTensor::from_slice(&[3.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    );
    let y = Tensor::from_tensor(vector_f64(&[4.0, 5.0]));
    let beta = Tensor::from_tensor(
        DenseTensor::from_slice(&[2.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    );

    let scaled = scale_dyn(x.as_dyn_ad_ref(), alpha.as_dyn_ad_ref()).unwrap();
    assert_eq!(scaled.scalar_type(), x.scalar_type());
    assert_eq!(scaled.dims(), &[2]);

    let divided = div_scalar_dyn(x.as_dyn_ad_ref(), alpha.as_dyn_ad_ref()).unwrap();
    assert_eq!(divided.scalar_type(), x.scalar_type());
    assert_eq!(divided.dims(), &[2]);

    let combined = axpby_dyn(
        x.as_dyn_ad_ref(),
        alpha.as_dyn_ad_ref(),
        y.as_dyn_ad_ref(),
        beta.as_dyn_ad_ref(),
    )
    .unwrap();
    assert_eq!(combined.scalar_type(), x.scalar_type());
    assert_eq!(combined.dims(), &[2]);
}

#[test]
fn axpby_unifies_pending_reverse_inputs_before_adding_scaled_terms() {
    let mut x = Tensor::from_tensor(vector_f64(&[1.0, 4.0]));
    x.set_requires_grad(true).unwrap();
    let mut y = Tensor::from_tensor(vector_f64(&[3.0, -1.0]));
    y.set_requires_grad(true).unwrap();
    let mut a = Tensor::from_tensor(
        DenseTensor::from_slice(&[2.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    );
    a.set_requires_grad(true).unwrap();
    let mut b = Tensor::from_tensor(
        DenseTensor::from_slice(&[-1.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    );
    b.set_requires_grad(true).unwrap();

    let out = x.axpby(&a, &y, &b).unwrap();
    let cotangent = Tensor::from_tensor(vector_f64(&[1.0, 1.0]));
    let grads = out.pullback_wrt(&cotangent, &[&x, &a, &y, &b]).unwrap();
    assert_eq!(grads.len(), 4);
    assert_eq!(grads[0].as_ref().unwrap().dims(), &[2]);
    assert!(grads[1].as_ref().unwrap().dims().is_empty());
    assert_eq!(grads[2].as_ref().unwrap().dims(), &[2]);
    assert!(grads[3].as_ref().unwrap().dims().is_empty());
}
