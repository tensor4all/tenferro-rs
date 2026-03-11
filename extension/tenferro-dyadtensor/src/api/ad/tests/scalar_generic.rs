use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{add_ad, exp_ad, mean_ad, AdTensor};

#[test]
fn ad_unary_binary_reduction_generic_surface_exists() {
    let x = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let y = Tensor::<f64>::from_slice(&[3.0, 4.0], &[2], MemoryOrder::ColumnMajor).unwrap();

    let ad_x = AdTensor::new_primal(x);
    let ad_y = AdTensor::new_primal(y);

    let out_unary = exp_ad(&ad_x).run().unwrap();
    assert_eq!(out_unary.dims(), &[2]);

    let out_binary = add_ad(&ad_x, &ad_y).run().unwrap();
    assert_eq!(out_binary.dims(), &[2]);

    let out_reduced = mean_ad(&out_binary).run().unwrap();
    assert_eq!(out_reduced.dims(), &[]);

    let eager_unary = crate::ad::exp(&ad_x).unwrap();
    let eager_binary = crate::ad::add(&ad_x, &ad_y).unwrap();
    let eager_mean = crate::ad::mean(&eager_binary).unwrap();

    assert_eq!(eager_unary.dims(), &[2]);
    assert_eq!(eager_binary.dims(), &[2]);
    assert_eq!(eager_mean.dims(), &[]);
}
