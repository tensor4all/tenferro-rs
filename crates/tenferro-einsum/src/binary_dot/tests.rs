use tenferro_tensor::DotGeneralConfig;

use super::{try_build_exact_output_binary_dot_plan, BinaryDotOperandOrder};

#[test]
fn compact_configs_match_full_plans_for_byte_and_integer_labels() {
    let terms = ["ij", "ji", "ijk", "kj", "j", "ii", "", "ipj", "jk"];
    let outputs = [
        "ip", "pi", "ij", "ji", "ikj", "kij", "", "i", "j", "ii", "z", "ipj", "pij",
    ];
    for lhs in terms {
        for rhs in terms {
            for output in outputs {
                let labels =
                    [lhs, rhs, output].map(|s| s.bytes().map(u32::from).collect::<Vec<_>>());
                let expected =
                    try_build_exact_output_binary_dot_plan(&labels[0], &labels[1], &labels[2])
                        .map(|plan| (plan.operand_order, plan.config));
                let bytes = super::try_build_exact_output_binary_dot_config(
                    lhs.as_bytes(),
                    rhs.as_bytes(),
                    output.as_bytes(),
                );
                let integers = super::try_build_exact_output_binary_dot_config(
                    &labels[0], &labels[1], &labels[2],
                );
                assert_eq!(bytes, expected, "{lhs},{rhs}->{output}");
                assert_eq!(integers, expected, "{lhs},{rhs}->{output}");
            }
        }
    }
}

#[test]
fn exact_binary_dot_plan_accepts_original_output_order() {
    let plan = try_build_exact_output_binary_dot_plan(
        &[b'i' as u32, b'j' as u32],
        &[b'j' as u32, b'k' as u32],
        &[b'i' as u32, b'k' as u32],
    )
    .expect("matmul should lower exactly");

    assert_eq!(plan.operand_order, BinaryDotOperandOrder::Original);
    assert_eq!(
        plan.config,
        DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        }
    );
}

#[test]
fn exact_binary_dot_plan_accepts_swapped_col_major_matmul() {
    let plan = try_build_exact_output_binary_dot_plan(
        &[b'j' as u32, b'i' as u32],
        &[b'k' as u32, b'j' as u32],
        &[b'k' as u32, b'i' as u32],
    )
    .expect("col-major matmul should lower exactly after swapping operands");

    assert_eq!(plan.operand_order, BinaryDotOperandOrder::Swapped);
    assert_eq!(
        plan.config,
        DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        }
    );
}

#[test]
fn exact_binary_dot_plan_accepts_swapped_col_major_batched_matmul() {
    let plan = try_build_exact_output_binary_dot_plan(
        &[b'j' as u32, b'i' as u32, b'b' as u32],
        &[b'k' as u32, b'j' as u32, b'b' as u32],
        &[b'k' as u32, b'i' as u32, b'b' as u32],
    )
    .expect("col-major batched matmul should lower exactly after swapping operands");

    assert_eq!(plan.operand_order, BinaryDotOperandOrder::Swapped);
    assert_eq!(
        plan.config,
        DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![2],
            rhs_batch_dims: vec![2],
        }
    );
}

#[test]
fn exact_binary_dot_plan_rejects_repeated_labels() {
    assert!(try_build_exact_output_binary_dot_plan(
        &[b'i' as u32, b'i' as u32],
        &[b'j' as u32],
        &[b'j' as u32],
    )
    .is_none());
}
