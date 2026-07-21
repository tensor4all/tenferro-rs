use std::alloc::{GlobalAlloc, Layout, System};
use std::ptr::null_mut;

use super::{
    case_by_name, case_inventory_json, consume, record_json, shape_token, Case, CaseInputs,
    CounterState, Mode, Operation, Snapshot, CASES, MEASURED_REPETITIONS,
    TRIANGULAR_REPETITION_FACTOR,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Failure {
    Alloc,
    AllocZeroed,
    Realloc,
}

#[derive(Default)]
struct TestDelegate {
    failure: Option<Failure>,
    alloc_calls: usize,
    alloc_zeroed_calls: usize,
    realloc_calls: usize,
    dealloc_calls: usize,
}

impl TestDelegate {
    unsafe fn alloc(&mut self, layout: Layout) -> *mut u8 {
        self.alloc_calls += 1;
        if self.failure == Some(Failure::Alloc) {
            return null_mut();
        }
        // SAFETY: Tests pass a non-zero valid Layout and retain the returned
        // pointer for exactly one matching realloc or dealloc call.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&mut self, layout: Layout) -> *mut u8 {
        self.alloc_zeroed_calls += 1;
        if self.failure == Some(Failure::AllocZeroed) {
            return null_mut();
        }
        // SAFETY: Tests pass a non-zero valid Layout and retain the returned
        // pointer for exactly one matching dealloc call.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&mut self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        self.realloc_calls += 1;
        if self.failure == Some(Failure::Realloc) {
            return null_mut();
        }
        // SAFETY: Tests forward the live pointer and its original Layout, and
        // new_size is non-zero. The returned pointer replaces the old one.
        unsafe { System.realloc(pointer, layout, new_size) }
    }

    unsafe fn dealloc(&mut self, pointer: *mut u8, layout: Layout) {
        self.dealloc_calls += 1;
        // SAFETY: Tests deallocate each live System pointer exactly once with
        // the Layout matching its current allocation.
        unsafe { System.dealloc(pointer, layout) };
    }
}

struct TestAllocator {
    counters: CounterState,
    delegate: TestDelegate,
}

impl TestAllocator {
    fn new() -> Self {
        Self {
            counters: CounterState::new(),
            delegate: TestDelegate::default(),
        }
    }

    fn reset(&self) {
        self.counters.reset();
    }

    fn snapshot(&self) -> Snapshot {
        self.counters.snapshot()
    }

    unsafe fn alloc(&mut self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller supplies a non-zero valid Layout and owns the
        // returned pointer lifecycle through this delegate.
        let pointer = unsafe { self.delegate.alloc(layout) };
        if pointer.is_null() {
            self.counters.record_failure();
        } else {
            self.counters.record_success(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&mut self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller supplies a non-zero valid Layout and owns the
        // returned pointer lifecycle through this delegate.
        let pointer = unsafe { self.delegate.alloc_zeroed(layout) };
        if pointer.is_null() {
            self.counters.record_failure();
        } else {
            self.counters.record_success(layout.size());
        }
        pointer
    }

    unsafe fn realloc(&mut self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: The caller supplies the live pointer/original Layout pair and
        // a non-zero new size; the delegate is called exactly once.
        let result = unsafe { self.delegate.realloc(pointer, layout, new_size) };
        if result.is_null() {
            self.counters.record_failure();
        } else {
            self.counters.record_success(new_size);
        }
        result
    }

    unsafe fn dealloc(&mut self, pointer: *mut u8, layout: Layout) {
        // SAFETY: The caller supplies a live pointer and its current Layout.
        unsafe { self.delegate.dealloc(pointer, layout) };
    }
}

fn layout(size: usize) -> Layout {
    assert_ne!(size, 0, "allocator boundary tests forbid zero-size layouts");
    Layout::from_size_align(size, 8).expect("non-zero test layout must be valid")
}

#[test]
fn valid_lifecycle_covers_all_four_allocator_methods() {
    let mut probe = TestAllocator::new();
    probe.reset();
    // SAFETY: Both layouts are non-zero and valid; each returned pointer is
    // kept live until its one matching realloc or dealloc below.
    let pointer = unsafe { probe.alloc(layout(8)) };
    // SAFETY: layout(16) is non-zero and valid and is deallocated below.
    let zeroed = unsafe { probe.alloc_zeroed(layout(16)) };
    assert!(!pointer.is_null());
    assert!(!zeroed.is_null());
    // SAFETY: pointer/layout describe the live 8-byte allocation and 32 > 0.
    let pointer = unsafe { probe.realloc(pointer, layout(8), 32) };
    assert!(!pointer.is_null());
    // SAFETY: Both pointers are live and paired with their current layouts.
    unsafe {
        probe.dealloc(pointer, layout(32));
        probe.dealloc(zeroed, layout(16));
    }
    assert_eq!(probe.snapshot(), Snapshot::valid(3, 56));
    assert_eq!(probe.delegate.alloc_calls, 1);
    assert_eq!(probe.delegate.alloc_zeroed_calls, 1);
    assert_eq!(probe.delegate.realloc_calls, 1);
    assert_eq!(probe.delegate.dealloc_calls, 2);
}

#[test]
fn alloc_zeroed_returns_zeroed_storage() {
    let mut probe = TestAllocator::new();
    // SAFETY: layout(16) is non-zero and valid; pointer is deallocated once.
    let pointer = unsafe { probe.alloc_zeroed(layout(16)) };
    assert!(!pointer.is_null());
    // SAFETY: pointer denotes a live 16-byte allocation returned by alloc_zeroed.
    let bytes = unsafe { std::slice::from_raw_parts(pointer, 16) };
    assert!(bytes.iter().all(|byte| *byte == 0));
    // SAFETY: pointer remains live and layout(16) is its allocation layout.
    unsafe { probe.dealloc(pointer, layout(16)) };
}

#[test]
fn failed_alloc_and_alloc_zeroed_increment_only_failures() {
    for failure in [Failure::Alloc, Failure::AllocZeroed] {
        let mut probe = TestAllocator::new();
        probe.delegate.failure = Some(failure);
        let pointer = match failure {
            // SAFETY: The injected delegate returns null without calling System.
            Failure::Alloc => unsafe { probe.alloc(layout(8)) },
            // SAFETY: The injected delegate returns null without calling System.
            Failure::AllocZeroed => unsafe { probe.alloc_zeroed(layout(8)) },
            Failure::Realloc => unreachable!(),
        };
        assert!(pointer.is_null());
        assert_eq!(
            probe.snapshot(),
            Snapshot {
                allocations: 0,
                bytes: 0,
                failures: 1,
                overflow: false,
            }
        );
    }
}

#[test]
fn failed_realloc_keeps_original_pointer_live_for_one_deallocation() {
    let mut probe = TestAllocator::new();
    // SAFETY: layout(8) is non-zero and valid; the pointer remains live.
    let original = unsafe { probe.alloc(layout(8)) };
    assert!(!original.is_null());
    probe.reset();
    probe.delegate.failure = Some(Failure::Realloc);
    // SAFETY: original/layout describe a live allocation and 32 > 0. The
    // injected delegate returns null without consuming original.
    let result = unsafe { probe.realloc(original, layout(8), 32) };
    assert!(result.is_null());
    assert_eq!(probe.snapshot().failures, 1);
    // SAFETY: Failed realloc leaves original live with its original layout.
    unsafe { probe.dealloc(original, layout(8)) };
    assert_eq!(probe.delegate.dealloc_calls, 1);
}

#[test]
fn each_counter_overflow_holds_that_counter_and_marks_invalid() {
    let mut allocation_probe = TestAllocator::new();
    allocation_probe.counters.seed(u64::MAX, 7, 0);
    // SAFETY: layout(8) is non-zero and valid; pointer is deallocated once.
    let pointer = unsafe { allocation_probe.alloc(layout(8)) };
    assert!(!pointer.is_null());
    assert_eq!(
        allocation_probe.snapshot(),
        Snapshot {
            allocations: u64::MAX,
            bytes: 15,
            failures: 0,
            overflow: true,
        }
    );
    // SAFETY: pointer is live and layout(8) matches its allocation.
    unsafe { allocation_probe.dealloc(pointer, layout(8)) };

    let mut byte_probe = TestAllocator::new();
    byte_probe.counters.seed(4, u64::MAX - 3, 0);
    // SAFETY: layout(8) is non-zero and valid; pointer is deallocated once.
    let pointer = unsafe { byte_probe.alloc(layout(8)) };
    assert_eq!(
        byte_probe.snapshot(),
        Snapshot {
            allocations: 5,
            bytes: u64::MAX - 3,
            failures: 0,
            overflow: true,
        }
    );
    // SAFETY: pointer is live and layout(8) matches its allocation.
    unsafe { byte_probe.dealloc(pointer, layout(8)) };

    let mut failure_probe = TestAllocator::new();
    failure_probe.counters.seed(2, 3, u64::MAX);
    failure_probe.delegate.failure = Some(Failure::Alloc);
    // SAFETY: The injected delegate returns null without calling System.
    let pointer = unsafe { failure_probe.alloc(layout(8)) };
    assert!(pointer.is_null());
    assert_eq!(
        failure_probe.snapshot(),
        Snapshot {
            allocations: 2,
            bytes: 3,
            failures: u64::MAX,
            overflow: true,
        }
    );
}

#[test]
fn zero_byte_delta_uses_safe_state_machine_without_overflow() {
    let counters = CounterState::new();
    counters.record_success(0);
    assert_eq!(counters.snapshot(), Snapshot::valid(1, 0));
}

#[test]
fn reset_and_finish_snapshot_observe_quiescent_protocol() {
    let counters = CounterState::new();
    counters.record_success(8);
    counters.reset();
    counters.record_success(16);
    let snapshot = counters.finish_and_snapshot();
    counters.record_success(32);
    assert_eq!(snapshot, Snapshot::valid(1, 16));
    assert_eq!(counters.snapshot(), snapshot);
}

#[test]
fn shape_token_values_are_frozen() {
    assert_eq!(shape_token(&[]), 0);
    assert_eq!(shape_token(&[1]), 258);
    assert_eq!(shape_token(&[8]), 265);
    assert_eq!(shape_token(&[64]), 321);
    assert_eq!(shape_token(&[1, 1]), 132_356);
    assert_eq!(shape_token(&[2, 2]), 132_614);
    assert_eq!(shape_token(&[usize::MAX]), 256);
}

#[test]
fn canonical_case_inventory_freezes_all_exact_mappings_and_order() {
    let expected = [
        ("lazy_neg_1", Mode::Lazy, Operation::Neg, 1),
        ("lazy_neg_8", Mode::Lazy, Operation::Neg, 8),
        ("lazy_neg_64", Mode::Lazy, Operation::Neg, 64),
        ("lazy_add_1", Mode::Lazy, Operation::Add, 1),
        ("lazy_add_8", Mode::Lazy, Operation::Add, 8),
        ("lazy_add_64", Mode::Lazy, Operation::Add, 64),
        ("lazy_reduce_1", Mode::Lazy, Operation::Reduce, 1),
        ("lazy_reduce_8", Mode::Lazy, Operation::Reduce, 8),
        ("lazy_reduce_64", Mode::Lazy, Operation::Reduce, 64),
        ("lazy_slice_1", Mode::Lazy, Operation::Slice, 1),
        ("lazy_slice_8", Mode::Lazy, Operation::Slice, 8),
        ("lazy_slice_64", Mode::Lazy, Operation::Slice, 64),
        ("lazy_dot_1", Mode::Lazy, Operation::Dot, 1),
        ("lazy_dot_2", Mode::Lazy, Operation::Dot, 2),
        ("materialized_neg_1", Mode::Materialized, Operation::Neg, 1),
        ("materialized_neg_8", Mode::Materialized, Operation::Neg, 8),
        (
            "materialized_neg_64",
            Mode::Materialized,
            Operation::Neg,
            64,
        ),
        ("materialized_add_1", Mode::Materialized, Operation::Add, 1),
        ("materialized_add_8", Mode::Materialized, Operation::Add, 8),
        (
            "materialized_add_64",
            Mode::Materialized,
            Operation::Add,
            64,
        ),
        (
            "materialized_reduce_1",
            Mode::Materialized,
            Operation::Reduce,
            1,
        ),
        (
            "materialized_reduce_8",
            Mode::Materialized,
            Operation::Reduce,
            8,
        ),
        (
            "materialized_reduce_64",
            Mode::Materialized,
            Operation::Reduce,
            64,
        ),
        (
            "materialized_slice_1",
            Mode::Materialized,
            Operation::Slice,
            1,
        ),
        (
            "materialized_slice_8",
            Mode::Materialized,
            Operation::Slice,
            8,
        ),
        (
            "materialized_slice_64",
            Mode::Materialized,
            Operation::Slice,
            64,
        ),
        ("materialized_dot_1", Mode::Materialized, Operation::Dot, 1),
        ("materialized_dot_2", Mode::Materialized, Operation::Dot, 2),
    ];
    let actual: Vec<_> = CASES
        .iter()
        .map(|case| (case.name, case.mode, case.operation, case.size))
        .collect();
    assert_eq!(actual.as_slice(), expected.as_slice());
    assert!(expected
        .iter()
        .all(|(name, mode, operation, size)| case_by_name(name)
            == Some(Case {
                name,
                mode: *mode,
                operation: *operation,
                size: *size,
            })));
    assert!(case_by_name("lazy_neg_0").is_none());
}

#[test]
fn list_cases_json_is_compact_complete_and_newline_terminated() {
    let expected_names = CASES
        .iter()
        .map(|case| format!("\"{}\"", case.name))
        .collect::<Vec<_>>()
        .join(",");
    assert_eq!(case_inventory_json(), format!("[{expected_names}]\n"));
}

#[test]
fn all_case_tokens_and_checksums_match_frozen_outputs_and_tags() {
    let expected_checksums = [
        2_173_179_904.0,
        2_231_914_496.0,
        2_701_791_232.0,
        2_173_179_904.0,
        2_231_914_496.0,
        2_701_791_232.0,
        8_390_656.0,
        8_390_656.0,
        8_390_656.0,
        2_181_570_560.0,
        2_240_305_152.0,
        2_710_181_888.0,
        1_110_562_056_192.0,
        1_112_726_845_440.0,
        2_156_398_592.0,
        2_215_133_184.0,
        2_685_009_920.0,
        2_181_570_560.0,
        2_240_305_152.0,
        2_710_181_888.0,
        8_390_656.0,
        67_125_248.0,
        537_001_984.0,
        2_173_179_904.0,
        2_231_914_496.0,
        2_701_791_232.0,
        1_110_562_056_192.0,
        1_112_735_236_096.0,
    ];

    for (case, expected_checksum) in CASES.into_iter().zip(expected_checksums) {
        let inputs = CaseInputs::new(case).expect("case setup should succeed");
        let output = inputs
            .execute(case.operation)
            .expect("case operation should succeed");
        let token = consume(output, case.mode).expect("case consumption should succeed");
        assert!(token.is_finite(), "{} token must be finite", case.name);
        assert_eq!(
            token * TRIANGULAR_REPETITION_FACTOR,
            expected_checksum,
            "{} checksum",
            case.name
        );
    }
}

#[test]
fn record_json_has_exact_sorted_schema_and_floating_checksum() {
    let case = CASES[0];
    let record = record_json(case, Snapshot::valid(3, 5), 7.0).expect("finite record");
    assert_eq!(
        record,
        concat!(
            "{\"allocated_bytes\":5,\"allocation_count\":3,",
            "\"allocation_failures\":0,\"case\":\"lazy_neg_1\",",
            "\"checksum\":7e0,\"counter_overflow\":false,",
            "\"repetitions\":4096}\n"
        )
    );
    assert!(record_json(case, Snapshot::valid(0, 0), f64::NAN).is_err());
    assert_eq!(MEASURED_REPETITIONS, 4_096);
}
