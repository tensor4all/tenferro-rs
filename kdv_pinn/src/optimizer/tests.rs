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

#[test]
fn adam_updates_correctly() {
    let mut param = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let grad = Tensor::from_vec_col_major(vec![2], vec![0.5_f64, 1.0]);
    let mut opt = Adam::new(0.1);
    opt.step(std::slice::from_mut(&mut param), &[grad]);
    let data = param.as_slice::<f64>().unwrap();
    // Adam with lr=0.1 should move parameters toward zero.
    assert!(data[0] < 1.0);
    assert!(data[1] < 2.0);
}

#[test]
fn adam_set_lr_changes_step_size() {
    // After lowering the learning rate to zero, a step must leave the
    // parameters unchanged, proving `set_lr` overrides the constructor value.
    let mut param = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let grad = Tensor::from_vec_col_major(vec![2], vec![0.5_f64, 1.0]);
    let mut opt = Adam::new(0.1);
    opt.set_lr(0.0);
    opt.step(std::slice::from_mut(&mut param), &[grad]);
    let data = param.as_slice::<f64>().unwrap();
    assert!((data[0] - 1.0).abs() < 1e-12);
    assert!((data[1] - 2.0).abs() < 1e-12);
}

#[test]
fn step_decay_lr_has_three_stages() {
    let base = 1.0e-3;
    let total = 3000;
    // First half: full base rate.
    assert!((step_decay_lr(0, total, base) - base).abs() < 1e-18);
    assert!((step_decay_lr(total / 2 - 1, total, base) - base).abs() < 1e-18);
    // Second quarter: halved.
    assert!((step_decay_lr(total / 2, total, base) - base * 0.5).abs() < 1e-18);
    assert!((step_decay_lr(3 * total / 4 - 1, total, base) - base * 0.5).abs() < 1e-18);
    // Final quarter: quartered.
    assert!((step_decay_lr(3 * total / 4, total, base) - base * 0.25).abs() < 1e-18);
    assert!((step_decay_lr(total - 1, total, base) - base * 0.25).abs() < 1e-18);
}
