use super::{
    resolve_cpu_affinity, resolve_cpu_affinity_with_override, CpuAffinityInput,
    CpuAffinityInputError, CpuAffinityPolicy, CpuAffinityResolutionError,
    CpuAffinitySelectionReason,
};
use tenferro_tensor::{CpuDomainId, DType, MemoryKind, Placement, Tensor, TypedTensor};

fn input(domain: Option<u64>, logical_bytes: usize) -> CpuAffinityInput {
    CpuAffinityInput {
        domain: domain.map(CpuDomainId::new),
        logical_bytes,
    }
}

#[test]
fn tensor_input_uses_checked_shape_product_and_dtype_width() {
    let mut tensor = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    tensor.set_placement(Placement {
        memory_kind: MemoryKind::UnpinnedHost,
        device: None,
        cpu_affinity: Some(CpuDomainId::new(7)),
    });

    let input = CpuAffinityInput::from_tensor(&Tensor::F64(tensor)).unwrap();

    assert_eq!(input.domain, Some(CpuDomainId::new(7)));
    assert_eq!(input.logical_bytes, 6 * std::mem::size_of::<f64>());
}

#[test]
fn scalar_and_zero_extent_logical_byte_counts_are_exact() {
    let scalar = CpuAffinityInput::from_parts(None, &[], DType::C64).unwrap();
    let zero =
        CpuAffinityInput::from_parts(Some(CpuDomainId::new(8)), &[usize::MAX, 2, 0], DType::F64)
            .unwrap();

    assert_eq!(
        scalar.logical_bytes,
        std::mem::size_of::<num_complex::Complex64>()
    );
    assert_eq!(zero.logical_bytes, 0);
    assert_eq!(zero.domain, Some(CpuDomainId::new(8)));
}

#[test]
fn shape_product_overflow_is_a_typed_input_error() {
    let error = CpuAffinityInput::from_parts(None, &[usize::MAX, 2], DType::F32).unwrap_err();

    assert_eq!(error, CpuAffinityInputError::ShapeProductOverflow);
}

#[test]
fn byte_width_multiplication_overflow_is_a_typed_input_error() {
    let error = CpuAffinityInput::from_parts(None, &[usize::MAX], DType::C64).unwrap_err();

    assert_eq!(
        error,
        CpuAffinityInputError::LogicalByteCountOverflow {
            element_count: usize::MAX,
            byte_width: std::mem::size_of::<num_complex::Complex64>(),
        }
    );
}

#[test]
fn input_overflow_remains_typed_through_the_error_trait() {
    let error = CpuAffinityInput::from_parts(None, &[usize::MAX], DType::C64).unwrap_err();
    let error: Box<dyn std::error::Error> = Box::new(error);

    assert!(matches!(
        error.downcast_ref::<CpuAffinityInputError>(),
        Some(CpuAffinityInputError::LogicalByteCountOverflow { .. })
    ));
}

#[test]
fn dominant_input_bytes_is_deterministic_and_keeps_inputs_in_place() {
    let inputs = [input(Some(9), 2), input(Some(4), 8), input(Some(9), 7)];
    let before = inputs;

    let selected = resolve_cpu_affinity(
        CpuAffinityPolicy::DominantInputBytes,
        &inputs,
        CpuDomainId::new(1),
    )
    .unwrap();

    assert_eq!(selected.domain, CpuDomainId::new(9));
    assert_eq!(
        selected.reason,
        CpuAffinitySelectionReason::DominantInputBytes
    );
    assert_eq!(inputs, before);
}

#[test]
fn dominant_input_bytes_aggregates_repeated_domains() {
    let selected = resolve_cpu_affinity(
        CpuAffinityPolicy::DominantInputBytes,
        &[input(Some(3), 4), input(Some(8), 7), input(Some(3), 4)],
        CpuDomainId::new(1),
    )
    .unwrap();

    assert_eq!(selected.domain, CpuDomainId::new(3));
}

#[test]
fn dominant_ties_use_the_smallest_id_independent_of_input_order() {
    let forward = [input(Some(9), 4), input(Some(2), 1), input(Some(2), 3)];
    let reverse = [input(Some(2), 3), input(Some(2), 1), input(Some(9), 4)];

    for inputs in [&forward[..], &reverse[..]] {
        let selected = resolve_cpu_affinity(
            CpuAffinityPolicy::DominantInputBytes,
            inputs,
            CpuDomainId::new(1),
        )
        .unwrap();
        assert_eq!(selected.domain, CpuDomainId::new(2));
    }
}

#[test]
fn unknown_affinities_do_not_contribute() {
    let selected = resolve_cpu_affinity(
        CpuAffinityPolicy::DominantInputBytes,
        &[input(None, usize::MAX), input(Some(7), 1)],
        CpuDomainId::new(1),
    )
    .unwrap();

    assert_eq!(selected.domain, CpuDomainId::new(7));
}

#[test]
fn all_unknown_inputs_use_the_default_domain() {
    let selected = resolve_cpu_affinity(
        CpuAffinityPolicy::DominantInputBytes,
        &[input(None, 10), input(None, usize::MAX)],
        CpuDomainId::new(5),
    )
    .unwrap();

    assert_eq!(selected.domain, CpuDomainId::new(5));
    assert_eq!(selected.reason, CpuAffinitySelectionReason::DefaultDomain);
}

#[test]
fn zero_byte_inputs_do_not_contribute_to_dominant_selection() {
    let selected = resolve_cpu_affinity(
        CpuAffinityPolicy::DominantInputBytes,
        &[input(Some(2), 0), input(Some(1), 0)],
        CpuDomainId::new(8),
    )
    .unwrap();

    assert_eq!(selected.domain, CpuDomainId::new(8));
    assert_eq!(selected.reason, CpuAffinitySelectionReason::DefaultDomain);
}

#[test]
fn positive_bytes_win_over_zero_byte_domains() {
    let selected = resolve_cpu_affinity(
        CpuAffinityPolicy::DominantInputBytes,
        &[input(Some(1), 0), input(Some(9), 1)],
        CpuDomainId::new(1),
    )
    .unwrap();

    assert_eq!(selected.domain, CpuDomainId::new(9));
}

#[test]
fn explicit_override_wins_before_policy_validation_or_accounting() {
    let inputs = [
        input(Some(1), usize::MAX),
        input(Some(1), 1),
        input(Some(2), 1),
    ];

    let selected = resolve_cpu_affinity_with_override(
        CpuAffinityPolicy::RequireSingleDomain,
        &inputs,
        CpuDomainId::new(3),
        Some(CpuDomainId::new(7)),
    )
    .unwrap();

    assert_eq!(selected.domain, CpuDomainId::new(7));
    assert_eq!(
        selected.reason,
        CpuAffinitySelectionReason::ExplicitOverride
    );
}

#[test]
fn require_single_domain_selects_the_only_known_domain_even_when_zero_bytes() {
    let selected = resolve_cpu_affinity(
        CpuAffinityPolicy::RequireSingleDomain,
        &[input(None, 20), input(Some(4), 0), input(Some(4), 8)],
        CpuDomainId::new(1),
    )
    .unwrap();

    assert_eq!(selected.domain, CpuDomainId::new(4));
    assert_eq!(
        selected.reason,
        CpuAffinitySelectionReason::SingleInputDomain
    );
}

#[test]
fn require_single_domain_uses_default_when_all_affinities_are_unknown() {
    let selected = resolve_cpu_affinity(
        CpuAffinityPolicy::RequireSingleDomain,
        &[input(None, 20)],
        CpuDomainId::new(6),
    )
    .unwrap();

    assert_eq!(selected.domain, CpuDomainId::new(6));
    assert_eq!(selected.reason, CpuAffinitySelectionReason::DefaultDomain);
}

#[test]
fn require_single_domain_reports_the_two_smallest_domains_deterministically() {
    let forward = [input(Some(9), 0), input(Some(4), 8), input(Some(2), 1)];
    let reverse = [input(Some(2), 1), input(Some(4), 8), input(Some(9), 0)];
    let expected = CpuAffinityResolutionError::MultipleKnownDomains {
        first: CpuDomainId::new(2),
        second: CpuDomainId::new(4),
    };

    for inputs in [&forward[..], &reverse[..]] {
        let error = resolve_cpu_affinity(
            CpuAffinityPolicy::RequireSingleDomain,
            inputs,
            CpuDomainId::new(1),
        )
        .unwrap_err();
        assert_eq!(error, expected);
    }
}

#[test]
fn checked_accounting_reports_the_smallest_overflowing_domain_deterministically() {
    let forward = [
        input(Some(9), usize::MAX),
        input(Some(2), usize::MAX),
        input(Some(9), 1),
        input(Some(2), 1),
    ];
    let reverse = [
        input(Some(2), 1),
        input(Some(9), 1),
        input(Some(2), usize::MAX),
        input(Some(9), usize::MAX),
    ];
    let expected = CpuAffinityResolutionError::LogicalByteCountOverflow {
        domain: CpuDomainId::new(2),
    };

    for inputs in [&forward[..], &reverse[..]] {
        let error = resolve_cpu_affinity(
            CpuAffinityPolicy::DominantInputBytes,
            inputs,
            CpuDomainId::new(1),
        )
        .unwrap_err();
        assert_eq!(error, expected);
    }
}

#[test]
fn more_than_eight_distinct_domains_keep_aggregation_semantics() {
    let inputs = [
        input(Some(1), 1),
        input(Some(2), 1),
        input(Some(3), 1),
        input(Some(4), 1),
        input(Some(5), 1),
        input(Some(6), 1),
        input(Some(7), 1),
        input(Some(8), 1),
        input(Some(9), 1),
        input(Some(9), 1),
    ];

    let selected = resolve_cpu_affinity(
        CpuAffinityPolicy::DominantInputBytes,
        &inputs,
        CpuDomainId::new(20),
    )
    .unwrap();

    assert_eq!(selected.domain, CpuDomainId::new(9));
}

#[test]
fn checked_overflow_remains_typed_after_many_distinct_domains() {
    let inputs = [
        input(Some(1), 1),
        input(Some(2), 1),
        input(Some(3), 1),
        input(Some(4), 1),
        input(Some(5), 1),
        input(Some(6), 1),
        input(Some(7), 1),
        input(Some(8), 1),
        input(Some(9), usize::MAX),
        input(Some(9), 1),
    ];

    let error = resolve_cpu_affinity(
        CpuAffinityPolicy::DominantInputBytes,
        &inputs,
        CpuDomainId::new(20),
    )
    .unwrap_err();

    assert_eq!(
        error,
        CpuAffinityResolutionError::LogicalByteCountOverflow {
            domain: CpuDomainId::new(9),
        }
    );
}
