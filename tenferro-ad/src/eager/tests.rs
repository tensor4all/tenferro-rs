use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use chainrules_core::ADKey;
use computegraph::types::GlobalValKey;
use tenferro_cpu::CpuBackend;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::SymDim;
use tenferro_tensor::Tensor;

use super::backward::{
    eager_forward_input_metadata, eager_forward_value, missing_tangent_base_key,
};
use super::{
    eager_op_profile_enabled, maybe_print_eager_op_profile, print_and_reset_eager_op_profile,
    profile_eager_op_section, record_eager_op_profile, zero_like_tensor,
};

#[test]
fn eager_op_profile_helpers_cover_enabled_paths() {
    unsafe {
        std::env::set_var("TENFERRO_PROFILE_EAGER_OP_AGG", "1");
        std::env::set_var("TENFERRO_PROFILE_EAGER_OP_PRINT_EVERY", "2");
    }

    assert!(eager_op_profile_enabled());
    assert_eq!(profile_eager_op_section("coverage.profile", || 7), 7);
    record_eager_op_profile("nary_op.total", Duration::from_micros(3));
    record_eager_op_profile("nary_op.total", Duration::from_micros(5));
    maybe_print_eager_op_profile();
    print_and_reset_eager_op_profile();
}

#[test]
fn eager_forward_helpers_synthesize_tangent_values_from_primal_data() {
    let user = TensorInputKey::User { id: 7 };
    let base_key = GlobalValKey::Input(user.clone());
    let tangent_key = GlobalValKey::Input(user.tangent_of(11));
    let base = Arc::new(Tensor::from_vec_col_major(vec![2], vec![4.0_f64, 5.0]));
    let initial_data = HashMap::from([(base_key.clone(), Arc::clone(&base))]);

    assert_eq!(missing_tangent_base_key(&base_key), None);
    assert_eq!(
        missing_tangent_base_key(&tangent_key),
        Some(base_key.clone())
    );
    assert_eq!(
        eager_forward_input_metadata(&tangent_key, &initial_data).shape,
        vec![SymDim::from(2usize)]
    );

    let mut all_values = HashMap::new();
    let mut backend = CpuBackend::new();
    let tangent = eager_forward_value(&mut all_values, &tangent_key, &initial_data, &mut backend);
    assert_eq!(tangent.as_slice::<f64>().unwrap(), &[0.0, 0.0]);
}

#[test]
fn zero_like_tensor_covers_non_f64_dtypes() {
    let mut backend = CpuBackend::new();
    let cases = [
        Tensor::from_vec_col_major(vec![1], vec![1.0_f32]),
        Tensor::from_vec_col_major(vec![1], vec![1_i32]),
        Tensor::from_vec_col_major(vec![1], vec![1_i64]),
        Tensor::from_vec_col_major(vec![2], vec![true, false]),
        Tensor::from_vec_col_major(vec![1], vec![num_complex::Complex32::new(1.0, -1.0)]),
        Tensor::from_vec_col_major(vec![1], vec![num_complex::Complex64::new(1.0, -1.0)]),
    ];

    for input in cases {
        let zero = zero_like_tensor(&input, &mut backend);
        assert_eq!(zero.shape(), input.shape());
    }
}
