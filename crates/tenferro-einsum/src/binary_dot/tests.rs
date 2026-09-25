use tenferro_tensor::DotGeneralConfig;

use super::{try_build_exact_output_binary_dot_plan, BinaryDotOperandOrder};

#[test]
fn compact_configs_inline_four_axes_and_spill_larger_lists() {
    for (lhs, rhs, output, contracting, batch) in [
        ("abcde", "dcbaq", "eq", 4, 0),
        ("abcdef", "edcbaq", "fq", 5, 0),
        ("abcdefg", "abcdegh", "fhabcde", 1, 5),
    ] {
        let (_, config) = super::try_build_exact_output_binary_dot_config(
            lhs.as_bytes(),
            rhs.as_bytes(),
            output.as_bytes(),
        )
        .unwrap();
        assert_eq!(config.lhs_contracting_dims.len(), contracting);
        assert_eq!(config.rhs_contracting_dims.len(), contracting);
        assert_eq!(config.lhs_batch_dims.len(), batch);
        assert_eq!(config.rhs_batch_dims.len(), batch);
        assert_eq!(config.lhs_contracting_dims.spilled(), contracting > 4);
        assert_eq!(config.rhs_contracting_dims.spilled(), contracting > 4);
        assert_eq!(config.lhs_batch_dims.spilled(), batch > 4);
        assert_eq!(config.rhs_batch_dims.spilled(), batch > 4);
        let labels = [lhs, rhs, output].map(|s| s.bytes().map(u32::from).collect::<Vec<_>>());
        let full =
            try_build_exact_output_binary_dot_plan(&labels[0], &labels[1], &labels[2]).unwrap();
        assert_eq!(config, full.config);
    }
}

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
            lhs_contracting_dims: [1].as_slice().into(),
            rhs_contracting_dims: [0].as_slice().into(),
            lhs_batch_dims: [].as_slice().into(),
            rhs_batch_dims: [].as_slice().into(),
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
            lhs_contracting_dims: [1].as_slice().into(),
            rhs_contracting_dims: [0].as_slice().into(),
            lhs_batch_dims: [].as_slice().into(),
            rhs_batch_dims: [].as_slice().into(),
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
            lhs_contracting_dims: [1].as_slice().into(),
            rhs_contracting_dims: [0].as_slice().into(),
            lhs_batch_dims: [2].as_slice().into(),
            rhs_batch_dims: [2].as_slice().into(),
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
