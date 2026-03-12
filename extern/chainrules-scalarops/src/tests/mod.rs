mod behavior;
mod organization;

#[test]
fn exp_log_and_atan2_rules_match_expected_derivatives() {
    let (exp_y, exp_dy) = crate::exp_frule(1.0_f64, 0.25_f64);
    assert!((exp_y - std::f64::consts::E).abs() < 1.0e-12);
    assert!((exp_dy - 0.25_f64 * std::f64::consts::E).abs() < 1.0e-12);

    let log_dx = crate::log_rrule(2.0_f64, 3.0_f64);
    assert!((log_dx - 1.5_f64).abs() < 1.0e-12);

    let (lhs_grad, rhs_grad) = crate::atan2_rrule(3.0_f64, 4.0_f64, 2.0_f64);
    assert!((lhs_grad - 0.32_f64).abs() < 1.0e-12);
    assert!((rhs_grad + 0.24_f64).abs() < 1.0e-12);
}
