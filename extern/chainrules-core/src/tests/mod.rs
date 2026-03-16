use std::hint::black_box;

use crate::Differentiable;

#[test]
fn differentiable_f32_zero_tangent_impl_is_reachable() {
    let x = black_box(42.0_f32);
    assert_eq!(<f32 as Differentiable>::zero_tangent(&x), 0.0_f32);
}

#[test]
fn differentiable_f32_accumulate_tangent_impl_is_reachable() {
    let lhs = black_box(1.5_f32);
    let rhs = black_box(2.5_f32);
    assert_eq!(
        <f32 as Differentiable>::accumulate_tangent(lhs, &rhs),
        4.0_f32
    );
}

#[test]
fn differentiable_f32_num_elements_impl_is_reachable() {
    let x = black_box(42.0_f32);
    assert_eq!(<f32 as Differentiable>::num_elements(&x), 1);
}

#[test]
fn differentiable_f32_seed_cotangent_impl_is_reachable() {
    let x = black_box(42.0_f32);
    assert_eq!(<f32 as Differentiable>::seed_cotangent(&x), 1.0_f32);
}
