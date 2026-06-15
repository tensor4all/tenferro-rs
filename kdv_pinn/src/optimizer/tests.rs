use super::*;
use tenferro_tensor::Tensor;

#[test]
fn sgd_updates_correctly() {
    let mut param = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let grad = Tensor::from_vec_col_major(vec![2], vec![0.5_f64, 1.0]);
    let mut opt = Sgd::new(0.1);
    opt.step(std::slice::from_mut(&mut param), &[grad]);
    let data = param.as_slice::<f64>().unwrap();
    assert!((data[0] - 0.95).abs() < 1e-9);
    assert!((data[1] - 1.9).abs() < 1e-9);
}
