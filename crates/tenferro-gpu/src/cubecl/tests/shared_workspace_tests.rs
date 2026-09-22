use super::super::gemm::{plan_workspace, shared_workspace_capacity, WorkspacePlan};
use super::super::DEFAULT_CUTENSOR_WORKSPACE_MAX_RETAINED_BYTES;

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

#[test]
fn shared_workspace_capacity_rounds_to_power_of_two_with_a_floor() {
    for (request, expected) in [
        (0, Some(0)),
        (1, Some(MIB)),
        (MIB - 1, Some(MIB)),
        (MIB, Some(MIB)),
        (MIB + 1, Some(2 * MIB)),
        (3 * MIB, Some(4 * MIB)),
        (1 << 63, Some(1 << 63)),
    ] {
        assert_eq!(shared_workspace_capacity(request), expected);
    }
    // Not representable as a power of two: the caller falls back to the
    // exact-size path instead of failing.
    assert_eq!(shared_workspace_capacity((1 << 63) + 1), None);
    assert_eq!(shared_workspace_capacity(u64::MAX), None);
}

#[test]
fn plan_workspace_reuses_retains_exact_and_falls_back_to_temporary() {
    // A slot that already holds enough capacity is reused, including the
    // zero-request case where no buffer exists at all.
    assert_eq!(plan_workspace(0, 0, 0, GIB), WorkspacePlan::Reuse);
    assert_eq!(
        plan_workspace(MIB, 2 * MIB, 2 * MIB, GIB),
        WorkspacePlan::Reuse
    );

    // Rounded capacity inside the headroom is retained.
    assert_eq!(plan_workspace(MIB, 0, 0, GIB), WorkspacePlan::Retain(MIB));
    assert_eq!(
        plan_workspace(MIB + 1, 0, 0, GIB),
        WorkspacePlan::Retain(2 * MIB)
    );

    // The rounded capacity does not fit but the exact request does.
    assert_eq!(
        plan_workspace(3 * MIB, 0, 0, 3 * MIB),
        WorkspacePlan::Retain(3 * MIB)
    );

    // Neither fits: run in a temporary workspace and keep other slots' buffers.
    assert_eq!(
        plan_workspace(3 * MIB, 0, GIB, GIB),
        WorkspacePlan::Temporary(3 * MIB)
    );

    // A slot's own retained capacity does not count against its headroom.
    assert_eq!(
        plan_workspace(4 * MIB, 4 * MIB, GIB, 4 * MIB),
        WorkspacePlan::Reuse
    );

    // Zero retention disables retention but still executes the contraction.
    assert_eq!(plan_workspace(MIB, 0, 0, 0), WorkspacePlan::Temporary(MIB));

    // A request whose rounding is unrepresentable uses the exact-size path,
    // and is only served when the exact size fits the remaining headroom.
    assert_eq!(
        plan_workspace(u64::MAX, 0, 0, u64::MAX),
        WorkspacePlan::Retain(u64::MAX)
    );
    assert_eq!(
        plan_workspace(u64::MAX, 0, 1, u64::MAX),
        WorkspacePlan::Temporary(u64::MAX)
    );
}

#[test]
fn default_retention_cap_is_one_gib() {
    assert_eq!(DEFAULT_CUTENSOR_WORKSPACE_MAX_RETAINED_BYTES, GIB);
}
