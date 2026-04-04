use crate::{AdMode, AdValue};

#[test]
fn ad_value_primal_reports_correct_mode() {
    let value = AdValue::forward(1_i32, 2_i32);
    assert_eq!(value.mode(), AdMode::Forward);
}

#[test]
fn ad_value_map_preserves_forward_mode() {
    let value = AdValue::forward(3_i32, 4_i32);
    let mapped = value.map_preserving_metadata(|v| v as f64);
    assert_eq!(mapped.mode(), AdMode::Forward);
    assert_eq!(mapped.primal_ref(), &3.0_f64);
    assert_eq!(mapped.tangent_ref(), Some(&4.0_f64));
}
