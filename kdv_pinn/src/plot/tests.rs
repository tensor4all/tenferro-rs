use super::*;

#[test]
fn loss_axis_bounds_brackets_positive_data() {
    let (lo, hi) = loss_axis_bounds(&[1.0, 0.5, 0.001]);
    assert!(lo > 0.0, "lower bound must stay positive for a log axis");
    assert!(lo <= 0.001, "lower bound must not exceed the smallest loss");
    assert!(hi >= 1.0, "upper bound must not be below the largest loss");
}

#[test]
fn loss_axis_bounds_ignores_nonpositive_and_nonfinite() {
    let (lo, hi) = loss_axis_bounds(&[f64::NAN, 0.0, -1.0, 0.01, f64::INFINITY]);
    assert!(lo > 0.0 && lo.is_finite());
    assert!(hi.is_finite());
    assert!(lo <= 0.01 && hi >= 0.01);
}

#[test]
fn loss_axis_bounds_defaults_when_no_positive_value() {
    assert_eq!(loss_axis_bounds(&[]), (1e-8, 1.0));
    assert_eq!(loss_axis_bounds(&[f64::NAN, 0.0, -2.0]), (1e-8, 1.0));
}
