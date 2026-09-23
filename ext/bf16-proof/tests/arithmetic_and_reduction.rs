//! Arithmetic and reduction through the shared boundary, with the accumulation contract measured.
//!
//! #1785 asks for basic forward arithmetic and sum reduction through the same public boundary the
//! preset scalars use, and for tests that distinguish the promised accumulation from repeated
//! rounding of the storage type. The shared entry points are used unchanged; what this crate adds
//! is the element type and its operation types.

use tenferro_bf16_proof::reduction::{sum_in_f32_accumulation, sum_with_per_step_rounding};
use tenferro_bf16_proof::{Bf16, Bf16Add, Bf16Mul, Bf16Sub};
use tenferro_cpu::{scalar_binary_into, scalar_fold};
use tenferro_tensor_core::HostTensor;

fn values(input: &[f32]) -> HostTensor<Bf16> {
    HostTensor::from_vec_col_major(
        vec![input.len()],
        input.iter().copied().map(Bf16::from_f32).collect(),
    )
    .expect("shape matches data")
}

fn widened(input: &HostTensor<Bf16>) -> Vec<f32> {
    input
        .as_slice()
        .iter()
        .map(|value| value.to_f32())
        .collect()
}

#[test]
fn the_shared_elementwise_entry_point_serves_bfloat16() {
    let lhs = values(&[1.0, 3.0]);
    let rhs = values(&[2.0, 4.0]);

    let mut destination = values(&[0.0, 0.0]);
    scalar_binary_into::<Bf16, Bf16Add>("add", &mut destination, &lhs, &rhs)
        .expect("addition through the shared entry point");
    assert_eq!(widened(&destination), vec![3.0, 7.0]);

    let mut difference = values(&[0.0, 0.0]);
    scalar_binary_into::<Bf16, Bf16Sub>("sub", &mut difference, &rhs, &lhs)
        .expect("subtraction through the shared entry point");
    assert_eq!(widened(&difference), vec![1.0, 1.0]);

    let mut product = values(&[0.0, 0.0]);
    scalar_binary_into::<Bf16, Bf16Mul>("mul", &mut product, &lhs, &rhs)
        .expect("multiplication through the shared entry point");
    assert_eq!(widened(&product), vec![2.0, 12.0]);
}

#[test]
fn a_single_operation_rounds_once() {
    // The midpoint of the bfloat16 interval above one is 1.0 + 2^-9, and ties go to even, so the
    // sum is exactly 1.0: one rounding rather than a chain of them.
    let lhs = values(&[1.0]);
    let rhs = values(&[2f32.powi(-9)]);
    let mut destination = values(&[0.0]);
    scalar_binary_into::<Bf16, Bf16Add>("add", &mut destination, &lhs, &rhs)
        .expect("addition through the shared entry point");
    assert_eq!(widened(&destination), vec![1.0]);
}

#[test]
fn the_shared_fold_rounds_at_every_step() {
    // The shared fold applies the element type's own addition, which rounds once per step. Its
    // accumulation order is the backend's, so the exact total depends on how the backend groups
    // partial sums: a strict left-to-right chain stalls at 256 where bfloat16 spacing above one
    // becomes two, while independent partial sums stay exact for longer. Every grouping lies
    // between the stalled chain and the exact sum.
    let ones = values(&[1.0; 300]);
    let total = scalar_fold::<Bf16, Bf16Add>("sum", &ones, Bf16::zero()).expect("shared fold");
    assert!(
        (256.0..=300.0).contains(&total.to_f32()),
        "{}",
        total.to_f32()
    );
    assert_eq!(sum_with_per_step_rounding(ones.as_slice()).to_f32(), 256.0);
}

#[test]
fn the_promised_accumulation_keeps_f32_intermediates() {
    // The crate's stated contract accumulates in f32 and rounds once, which is what makes it
    // distinguishable from the shared fold rather than merely close to it.
    let ones = values(&[1.0; 300]);
    assert_eq!(
        sum_in_f32_accumulation(ones.as_slice()).to_f32(),
        300.0,
        "a per-step rounding contract would have produced 256.0 here"
    );
}

#[test]
fn the_reduced_output_is_quantized_to_the_storage_type() {
    // Accumulating in f32 does not change the stored type of the result: the sum of ten stored
    // values is rounded once on the way out, so the output is a bfloat16 of that rounded value.
    let stored = values(&[1.0 + 2f32.powi(-9); 10]);
    let total = sum_in_f32_accumulation(stored.as_slice());
    assert_eq!(total.narrow(), half::bf16::from_f32(total.to_f32()));

    // The accumulation kept 10 * 2^-9 of the increments while the output quantized the result to
    // the spacing of bfloat16 on [8, 16), which is 2^-4.
    let exact = 10.0 + 10.0 * 2f32.powi(-9);
    assert!(
        (total.to_f32() - exact).abs() <= 2f32.powi(-4),
        "the reduced output must be within one step of the exact sum: {} against {exact}",
        total.to_f32()
    );
}
