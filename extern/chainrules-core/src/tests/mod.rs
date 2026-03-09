use std::hint::black_box;

use super::Differentiable;

#[test]
fn f32_trait_methods_are_exercised_in_crate_unit_tests() {
    let x = black_box(42.0_f32);
    let y = black_box(2.5_f32);
    let zero_tangent: fn(&f32) -> f32 = <f32 as Differentiable>::zero_tangent;
    let accumulate_tangent: fn(f32, &f32) -> f32 = <f32 as Differentiable>::accumulate_tangent;
    let num_elements: fn(&f32) -> usize = <f32 as Differentiable>::num_elements;
    let seed_cotangent: fn(&f32) -> f32 = <f32 as Differentiable>::seed_cotangent;

    assert_eq!(black_box(zero_tangent)(&x), 0.0_f32);
    assert_eq!(black_box(accumulate_tangent)(x, &y), 44.5_f32);
    assert_eq!(black_box(num_elements)(&x), 1);
    assert_eq!(black_box(seed_cotangent)(&x), 1.0_f32);
}
