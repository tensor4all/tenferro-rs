//! Test-only component probes for the concrete einsum preparation seams.
//!
//! The ignored entry point is intentionally a thin driver for benchmark #95. It
//! reads `TENFERRO_PROBE_MODE`, `TENFERRO_PROBE_CASE`, `TENFERRO_PROBE_STAGE`,
//! `TENFERRO_PROBE_ITERATIONS`, `TENFERRO_PROBE_SAMPLES`, and
//! `TENFERRO_PROBE_MIN_AGGREGATE_NS`. Timed and allocation modes are separate;
//! each invocation selects one stage, and ordinary tests never collect timings.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

use num_complex::Complex64;
use tenferro_tensor::{DType, Tensor, ValidationError};

use super::{input_specs, ConcreteEinsumInputSpec, ConcreteEinsumPlan};
use crate::{ContractionTree, Error, Subscripts};

const OUTPUT_PREFIX: &str = "TENFERRO_EINSUM_PROBE_JSON ";
const VALIDATION_OP: &str = "einsum_component_probe";
const BINARY_NOTATION: &str = "ab,bc->ac";

thread_local! {
    // const initialization is required: allocator calls must not initialize TLS by allocating.
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATION_CALLS: Cell<usize> = const { Cell::new(0) };
    static REQUESTED_BYTES: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

// SAFETY: every operation forwards the original pointer/layout to `System`; only
// caller-thread TLS counters are updated, and the disabled fast path is allocation-free.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation(layout.size());
        // SAFETY: the unchanged layout is forwarded to the standard allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation(layout.size());
        // SAFETY: the unchanged layout is forwarded to the standard allocator.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the pointer and layout came from the corresponding System allocation.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_allocation(new_size);
        // SAFETY: the pointer/layout came from System and the requested size is unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn record_allocation(bytes: usize) {
    let Ok(enabled) = COUNTING.try_with(Cell::get) else {
        return;
    };
    if enabled {
        let _ = ALLOCATION_CALLS.try_with(|calls| calls.set(calls.get().saturating_add(1)));
        let _ = REQUESTED_BYTES.try_with(|total| total.set(total.get().saturating_add(bytes)));
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AllocationSnapshot {
    calls: usize,
    requested_bytes: usize,
}

struct CountingGuard;

impl CountingGuard {
    fn try_new() -> Result<Self, &'static str> {
        if COUNTING
            .try_with(Cell::get)
            .map_err(|_| "allocator_tls_unavailable")?
        {
            return Err("nested_allocator_guard");
        }
        ALLOCATION_CALLS
            .try_with(|calls| calls.set(0))
            .map_err(|_| "allocator_tls_unavailable")?;
        REQUESTED_BYTES
            .try_with(|bytes| bytes.set(0))
            .map_err(|_| "allocator_tls_unavailable")?;
        COUNTING
            .try_with(|enabled| enabled.set(true))
            .map_err(|_| "allocator_tls_unavailable")?;
        Ok(Self)
    }

    fn snapshot(&self) -> AllocationSnapshot {
        AllocationSnapshot {
            calls: ALLOCATION_CALLS.try_with(Cell::get).unwrap_or(0),
            requested_bytes: REQUESTED_BYTES.try_with(Cell::get).unwrap_or(0),
        }
    }
}

impl Drop for CountingGuard {
    fn drop(&mut self) {
        let _ = COUNTING.try_with(|enabled| enabled.set(false));
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InvalidRevalidation {
    Count,
    Dtype,
    Shape,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CaseId {
    Rank2UnaryF64,
    Rank2BinaryF64,
    Rank4BinaryC64,
    Rank4FourF64Alternating,
    Rank8EightF64,
    Rank2BinaryF64CountInvalid,
    Rank2BinaryF64DtypeInvalid,
    Rank2BinaryF64ShapeInvalid,
}

const CASES: [CaseId; 8] = [
    CaseId::Rank2UnaryF64,
    CaseId::Rank2BinaryF64,
    CaseId::Rank4BinaryC64,
    CaseId::Rank4FourF64Alternating,
    CaseId::Rank8EightF64,
    CaseId::Rank2BinaryF64CountInvalid,
    CaseId::Rank2BinaryF64DtypeInvalid,
    CaseId::Rank2BinaryF64ShapeInvalid,
];

impl CaseId {
    fn id(self) -> &'static str {
        match self {
            Self::Rank2UnaryF64 => "rank2-unary-f64",
            Self::Rank2BinaryF64 => "rank2-binary-f64",
            Self::Rank4BinaryC64 => "rank4-binary-c64",
            Self::Rank4FourF64Alternating => "rank4-four-f64-alternating",
            Self::Rank8EightF64 => "rank8-eight-f64",
            Self::Rank2BinaryF64CountInvalid => "rank2-binary-f64-count-invalid",
            Self::Rank2BinaryF64DtypeInvalid => "rank2-binary-f64-dtype-invalid",
            Self::Rank2BinaryF64ShapeInvalid => "rank2-binary-f64-shape-invalid",
        }
    }

    fn dtype(self) -> DType {
        match self {
            Self::Rank4BinaryC64 => DType::C64,
            _ => DType::F64,
        }
    }

    fn notation(self) -> &'static str {
        match self {
            Self::Rank2UnaryF64 => "ab->ab",
            Self::Rank2BinaryF64 => BINARY_NOTATION,
            Self::Rank4BinaryC64 => "abcd,cdef->abef",
            Self::Rank4FourF64Alternating => "abcd,abcd,abcd,abcd->",
            Self::Rank8EightF64 => {
                "abcdefgh,abcdefgh,abcdefgh,abcdefgh,abcdefgh,abcdefgh,abcdefgh,abcdefgh->abcdefgh"
            }
            Self::Rank2BinaryF64CountInvalid
            | Self::Rank2BinaryF64DtypeInvalid
            | Self::Rank2BinaryF64ShapeInvalid => BINARY_NOTATION,
        }
    }

    fn rank(self) -> usize {
        match self {
            Self::Rank2UnaryF64 | Self::Rank2BinaryF64 => 2,
            Self::Rank4BinaryC64 | Self::Rank4FourF64Alternating => 4,
            Self::Rank8EightF64 => 8,
            Self::Rank2BinaryF64CountInvalid
            | Self::Rank2BinaryF64DtypeInvalid
            | Self::Rank2BinaryF64ShapeInvalid => 2,
        }
    }

    fn input_count(self) -> usize {
        match self {
            Self::Rank2UnaryF64 => 1,
            Self::Rank2BinaryF64
            | Self::Rank4BinaryC64
            | Self::Rank2BinaryF64CountInvalid
            | Self::Rank2BinaryF64DtypeInvalid
            | Self::Rank2BinaryF64ShapeInvalid => 2,
            Self::Rank4FourF64Alternating => 4,
            Self::Rank8EightF64 => 8,
        }
    }

    fn alternating(self) -> bool {
        matches!(self, Self::Rank4FourF64Alternating)
    }

    fn invalid_revalidation(self) -> Option<InvalidRevalidation> {
        match self {
            Self::Rank2BinaryF64CountInvalid => Some(InvalidRevalidation::Count),
            Self::Rank2BinaryF64DtypeInvalid => Some(InvalidRevalidation::Dtype),
            Self::Rank2BinaryF64ShapeInvalid => Some(InvalidRevalidation::Shape),
            _ => None,
        }
    }

    fn parse(value: &str) -> Option<Self> {
        CASES.into_iter().find(|case| case.id() == value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Parse,
    InputMetadata,
    PreparedSpecRevalidation,
    PrepareCombined,
    FixedPairPrepareCombined,
    EmptyControl,
}

const STAGES: [Stage; 6] = [
    Stage::Parse,
    Stage::InputMetadata,
    Stage::PreparedSpecRevalidation,
    Stage::PrepareCombined,
    Stage::FixedPairPrepareCombined,
    Stage::EmptyControl,
];

impl Stage {
    fn parse(value: &str) -> Option<Self> {
        STAGES.into_iter().find(|stage| stage.name() == value)
    }

    fn name(self) -> &'static str {
        match self {
            Self::Parse => "parse",
            Self::InputMetadata => "input_metadata",
            Self::PreparedSpecRevalidation => "prepared_spec_revalidation",
            Self::PrepareCombined => "prepare_combined",
            Self::FixedPairPrepareCombined => "fixed_pair_prepare_combined",
            Self::EmptyControl => "empty_control",
        }
    }

    fn phase(self) -> &'static str {
        match self {
            Self::Parse => "syntax",
            Self::InputMetadata => "concrete.input_specs",
            Self::PreparedSpecRevalidation => "concrete.validate_inputs",
            Self::PrepareCombined => "concrete.prepare_subscripts_internal",
            Self::FixedPairPrepareCombined => "planning.from_pairs",
            Self::EmptyControl => "control",
        }
    }

    fn applies(self, case: CaseId) -> bool {
        if case.invalid_revalidation().is_some() {
            return self == Self::PreparedSpecRevalidation;
        }
        !matches!(self, Self::FixedPairPrepareCombined) || matches!(case.input_count(), 2)
    }
}

struct Fixture {
    case: CaseId,
    notation: Subscripts,
    inputs: Vec<Tensor>,
    alternate_inputs: Vec<Tensor>,
    specs: Vec<ConcreteEinsumInputSpec>,
    alternate_specs: Vec<ConcreteEinsumInputSpec>,
    prepared: ConcreteEinsumPlan,
    invalid_specs: Option<Vec<ConcreteEinsumInputSpec>>,
}

fn checked_element_count(shape: &[usize]) -> usize {
    shape
        .iter()
        .try_fold(1usize, |count, &dim| count.checked_mul(dim))
        .unwrap()
}

fn shapes(case: CaseId, alternate: bool) -> Vec<Vec<usize>> {
    match case {
        CaseId::Rank2UnaryF64 => vec![vec![2, 3]],
        CaseId::Rank2BinaryF64 => vec![vec![2, 3], vec![3, 2]],
        CaseId::Rank4BinaryC64 => vec![vec![1, 2, 1, 2], vec![1, 2, 1, 2]],
        CaseId::Rank4FourF64Alternating => {
            let shape = if alternate {
                vec![2, 1, 1, 1]
            } else {
                vec![1, 2, 1, 1]
            };
            vec![shape; 4]
        }
        CaseId::Rank8EightF64 => vec![vec![1; 8]; 8],
        CaseId::Rank2BinaryF64CountInvalid
        | CaseId::Rank2BinaryF64DtypeInvalid
        | CaseId::Rank2BinaryF64ShapeInvalid => vec![vec![2, 3], vec![3, 2]],
    }
}

fn make_tensor(dtype: DType, shape: &[usize], seed: usize) -> Tensor {
    let count = checked_element_count(shape);
    match dtype {
        DType::F64 => Tensor::from_vec_col_major(
            shape.to_vec(),
            (0..count)
                .map(|index| seed as f64 + index as f64 + 1.0)
                .collect(),
        )
        .unwrap(),
        DType::C64 => Tensor::from_vec_col_major(
            shape.to_vec(),
            (0..count)
                .map(|index| Complex64::new(seed as f64 + index as f64 + 1.0, -0.5))
                .collect(),
        )
        .unwrap(),
        dtype => panic!("unsupported probe fixture dtype: {dtype:?}"),
    }
}

fn make_fixture(case: CaseId) -> Fixture {
    let notation = Subscripts::parse(case.notation()).unwrap();
    let fixed_shapes = shapes(case, false);
    let alternate_shapes = shapes(case, true);
    let inputs: Vec<_> = fixed_shapes
        .iter()
        .enumerate()
        .map(|(index, shape)| make_tensor(case.dtype(), shape, index))
        .collect();
    let alternate_inputs: Vec<_> = alternate_shapes
        .iter()
        .enumerate()
        .map(|(index, shape)| make_tensor(case.dtype(), shape, index + 10))
        .collect();
    let fixed_refs: Vec<_> = inputs.iter().collect();
    let specs = input_specs(&fixed_refs);
    let alternate_refs: Vec<_> = alternate_inputs.iter().collect();
    let alternate_specs = input_specs(&alternate_refs);
    let prepared =
        ConcreteEinsumPlan::prepare_subscripts_internal(specs.clone(), &notation).unwrap();
    let invalid_specs = case.invalid_revalidation().map(|invalid| match invalid {
        InvalidRevalidation::Count => specs[..1].to_vec(),
        InvalidRevalidation::Dtype => vec![
            ConcreteEinsumInputSpec {
                dtype: DType::C64,
                shape: specs[0].shape.clone(),
            },
            specs[1].clone(),
        ],
        InvalidRevalidation::Shape => vec![
            ConcreteEinsumInputSpec {
                dtype: DType::F64,
                shape: vec![4, 3],
            },
            specs[1].clone(),
        ],
    });
    Fixture {
        case,
        notation,
        inputs,
        alternate_inputs,
        specs,
        alternate_specs,
        prepared,
        invalid_specs,
    }
}

fn execute_stage(
    fixture: &Fixture,
    fixed_inputs: &[&Tensor],
    alternate_inputs: &[&Tensor],
    iteration: usize,
    stage: Stage,
) -> crate::Result<usize> {
    let use_alternate = fixture.case.alternating() && iteration % 2 == 1;
    let inputs = if use_alternate {
        alternate_inputs
    } else {
        fixed_inputs
    };
    let specs = if let Some(invalid_specs) = &fixture.invalid_specs {
        invalid_specs
    } else if use_alternate {
        &fixture.alternate_specs
    } else {
        &fixture.specs
    };
    match stage {
        Stage::Parse => Ok(Subscripts::parse(fixture.case.notation())?.inputs.len()),
        Stage::InputMetadata => {
            let metadata = input_specs(inputs);
            let count = metadata.len();
            black_box(metadata);
            Ok(count)
        }
        Stage::PreparedSpecRevalidation => {
            fixture.prepared.validate_inputs(specs, VALIDATION_OP)?;
            Ok(specs.len())
        }
        Stage::PrepareCombined => {
            let plan =
                ConcreteEinsumPlan::prepare_subscripts_internal(specs.clone(), &fixture.notation)?;
            let steps = plan.tree.step_count();
            black_box(plan);
            Ok(steps)
        }
        Stage::FixedPairPrepareCombined => {
            let shapes = [
                fixture.specs[0].shape.as_slice(),
                fixture.specs[1].shape.as_slice(),
            ];
            let tree = ContractionTree::from_pairs(&fixture.notation, &shapes, &[(0, 1)])?;
            let steps = tree.step_count();
            black_box(tree);
            Ok(steps)
        }
        Stage::EmptyControl => {
            black_box((fixture.case.id(), fixture.case.input_count(), iteration));
            Ok(0)
        }
    }
}

fn expected_stage_result(case: CaseId, stage: Stage, iteration: usize) -> ExpectedStageResult {
    if stage == Stage::PreparedSpecRevalidation {
        if let Some(invalid) = case.invalid_revalidation() {
            return ExpectedStageResult::Invalid(invalid);
        }
        if case.alternating() && iteration % 2 == 1 {
            return ExpectedStageResult::ShapeMismatch;
        }
    }
    {
        ExpectedStageResult::Value(match stage {
            Stage::Parse | Stage::InputMetadata | Stage::PreparedSpecRevalidation => {
                case.input_count()
            }
            Stage::PrepareCombined => case.input_count().saturating_sub(1),
            Stage::FixedPairPrepareCombined => 1,
            Stage::EmptyControl => 0,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExpectedStageResult {
    Value(usize),
    Invalid(InvalidRevalidation),
    ShapeMismatch,
}

fn check_stage_result(
    result: crate::Result<usize>,
    expected: ExpectedStageResult,
) -> Result<(), &'static str> {
    match (expected, result) {
        (ExpectedStageResult::Value(expected), Ok(actual)) if actual == expected => Ok(()),
        (ExpectedStageResult::Value(_), Ok(_)) => Err("unexpected_value"),
        (ExpectedStageResult::Value(_), Err(_)) => Err("unexpected_error"),
        (
            ExpectedStageResult::Invalid(InvalidRevalidation::Count),
            Err(Error::Validation {
                source:
                    ValidationError::InvalidArgument {
                        argument: "inputs", ..
                    },
                ..
            }),
        ) => Ok(()),
        (
            ExpectedStageResult::Invalid(InvalidRevalidation::Dtype),
            Err(Error::Tensor(tenferro_tensor::Error::Validation {
                source: ValidationError::DTypeMismatch { .. },
                ..
            })),
        ) => Ok(()),
        (
            ExpectedStageResult::Invalid(InvalidRevalidation::Shape),
            Err(Error::Validation {
                source: ValidationError::ShapeMismatch(_),
                ..
            }),
        ) => Ok(()),
        (
            ExpectedStageResult::ShapeMismatch,
            Err(Error::Validation {
                source: ValidationError::ShapeMismatch(_),
                ..
            }),
        ) => Ok(()),
        (ExpectedStageResult::Invalid(_), Ok(_)) => Err("expected_validation_error"),
        (ExpectedStageResult::Invalid(_), Err(_)) => Err("unexpected_error"),
        (ExpectedStageResult::ShapeMismatch, Ok(_)) => Err("expected_shape_mismatch"),
        (ExpectedStageResult::ShapeMismatch, Err(_)) => Err("unexpected_error"),
    }
}

fn fixed_input_refs(fixture: &Fixture) -> Vec<&Tensor> {
    fixture.inputs.iter().collect()
}

fn alternate_input_refs(fixture: &Fixture) -> Vec<&Tensor> {
    fixture.alternate_inputs.iter().collect()
}

fn assert_stage_contract(fixture: &Fixture) {
    if fixture.case.invalid_revalidation().is_some() {
        let fixed_inputs = fixed_input_refs(fixture);
        let result = execute_stage(
            fixture,
            &fixed_inputs,
            &[],
            0,
            Stage::PreparedSpecRevalidation,
        );
        assert!(check_stage_result(
            result,
            expected_stage_result(fixture.case, Stage::PreparedSpecRevalidation, 0),
        )
        .is_ok());
        return;
    }
    let fixed_inputs = fixed_input_refs(fixture);
    let alternate_inputs = alternate_input_refs(fixture);
    assert_eq!(
        execute_stage(fixture, &fixed_inputs, &alternate_inputs, 0, Stage::Parse).unwrap(),
        fixture.case.input_count()
    );
    assert_eq!(
        execute_stage(
            fixture,
            &fixed_inputs,
            &alternate_inputs,
            0,
            Stage::InputMetadata
        )
        .unwrap(),
        fixture.case.input_count()
    );
    assert_eq!(
        execute_stage(
            fixture,
            &fixed_inputs,
            &alternate_inputs,
            0,
            Stage::PreparedSpecRevalidation
        )
        .unwrap(),
        fixture.case.input_count()
    );
    assert_eq!(
        execute_stage(
            fixture,
            &fixed_inputs,
            &alternate_inputs,
            0,
            Stage::PrepareCombined
        )
        .unwrap(),
        fixture.case.input_count().saturating_sub(1)
    );
    assert_eq!(
        execute_stage(
            fixture,
            &fixed_inputs,
            &alternate_inputs,
            0,
            Stage::EmptyControl
        )
        .unwrap(),
        0
    );
    if fixture.case.input_count() == 2 {
        assert_eq!(
            execute_stage(
                fixture,
                &fixed_inputs,
                &alternate_inputs,
                0,
                Stage::FixedPairPrepareCombined
            )
            .unwrap(),
            1
        );
    }
    if fixture.case.alternating() {
        let error = execute_stage(
            fixture,
            &fixed_inputs,
            &alternate_inputs,
            1,
            Stage::PreparedSpecRevalidation,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            Error::Validation {
                source: ValidationError::ShapeMismatch(_),
                ..
            }
        ));
    }
}

fn json_string(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn contract_record(case: CaseId, stage: Stage) -> String {
    let dtype = match case.dtype() {
        DType::F64 => "f64",
        DType::C64 => "c64",
        _ => unreachable!(),
    };
    let setup = match stage {
        Stage::Parse => "notation_string_only",
        Stage::InputMetadata => "preconstructed_tensors;input_specs_output_drop_included",
        Stage::PreparedSpecRevalidation => {
            "preconstructed_expected_and_actual_specs;count_dtype_shape_only"
        }
        Stage::PrepareCombined => "preconstructed_subscripts;owned_spec_clone_included",
        Stage::FixedPairPrepareCombined => {
            "preconstructed_specs;stack_shape_refs_per_call;step_compilation_included"
        }
        Stage::EmptyControl => "same_driver_loop_and_output_scaffolding;no_component_work",
    };
    let expected_outcomes = match (case.invalid_revalidation(), stage) {
        (Some(InvalidRevalidation::Count), Stage::PreparedSpecRevalidation) => {
            "count-invalid=validation.invalid_argument"
        }
        (Some(InvalidRevalidation::Dtype), Stage::PreparedSpecRevalidation) => {
            "dtype-invalid=tensor.validation.dtype_mismatch"
        }
        (Some(InvalidRevalidation::Shape), Stage::PreparedSpecRevalidation) => {
            "shape-invalid=validation.shape_mismatch"
        }
        _ => "ok",
    };
    let revalidation_cases = match (case.invalid_revalidation(), stage) {
        (Some(InvalidRevalidation::Count), Stage::PreparedSpecRevalidation) => "count-invalid",
        (Some(InvalidRevalidation::Dtype), Stage::PreparedSpecRevalidation) => "dtype-invalid",
        (Some(InvalidRevalidation::Shape), Stage::PreparedSpecRevalidation) => "shape-invalid",
        _ => "none",
    };
    let accounting = match stage {
        Stage::InputMetadata => "owned_metadata_collection_and_destruction",
        Stage::PreparedSpecRevalidation => {
            "validation_only;no_layout_storage_backend_checks;invalid_count_dtype_shape_contracts"
        }
        Stage::PrepareCombined | Stage::FixedPairPrepareCombined => {
            "combined_validation_and_planning"
        }
        Stage::Parse => "parser_only;preparation_excluded",
        Stage::EmptyControl => "driver_control_only",
    };
    format!(
        "{OUTPUT_PREFIX}{{\"kind\":\"contract\",\"schema\":\"tenferro.einsum.component-probe.v1\",\"binary\":\"{}\",\"source\":\"{}\",\"case_id\":\"{}\",\"stage\":\"{}\",\"phase\":\"{}\",\"dtype\":\"{}\",\"rank\":{},\"input_count\":{},\"metadata\":\"{}\",\"setup\":\"{}\",\"calls_per_workflow\":1,\"revalidation_cases\":\"{}\",\"expected_outcomes\":\"{}\",\"accounting\":\"{}\"}}",
        env!("CARGO_PKG_NAME"),
        concat!(module_path!(), "::", file!()),
        json_string(case.id()),
        stage.name(),
        stage.phase(),
        dtype,
        case.rank(),
        case.input_count(),
        if case.alternating() { "alternating" } else { "fixed" },
        setup,
        revalidation_cases,
        expected_outcomes,
        accounting,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProbeMode {
    Timed,
    Alloc,
    Correctness,
    Contract,
}

impl ProbeMode {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "timed" => Some(Self::Timed),
            "alloc" => Some(Self::Alloc),
            "correctness" => Some(Self::Correctness),
            "contract" => Some(Self::Contract),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ProbeConfig {
    mode: ProbeMode,
    case: Option<CaseId>,
    stage: Option<Stage>,
    iterations: Option<usize>,
    samples: Option<usize>,
    min_aggregate_ns: Option<u128>,
}

impl ProbeConfig {
    fn parse(
        mode: &str,
        case: Option<&str>,
        stage: Option<&str>,
        iterations: Option<&str>,
        samples: Option<&str>,
        min_aggregate_ns: Option<&str>,
    ) -> Result<Self, String> {
        let mode = ProbeMode::parse(mode)
            .ok_or_else(|| format!("invalid TENFERRO_PROBE_MODE={mode:?}"))?;
        let case = match case {
            Some("all") if mode == ProbeMode::Contract => None,
            Some(value) => Some(
                CaseId::parse(value)
                    .ok_or_else(|| format!("invalid TENFERRO_PROBE_CASE={value:?}"))?,
            ),
            None if mode == ProbeMode::Contract => None,
            None => return Err("TENFERRO_PROBE_CASE is required for this mode".into()),
        };
        let stage = stage
            .map(|value| {
                Stage::parse(value).ok_or_else(|| format!("invalid TENFERRO_PROBE_STAGE={value:?}"))
            })
            .transpose()?;
        if let (Some(case), Some(stage)) = (case, stage) {
            if !stage.applies(case) {
                return Err("TENFERRO_PROBE_STAGE does not apply to TENFERRO_PROBE_CASE".into());
            }
        }
        let iterations = parse_optional_usize("TENFERRO_PROBE_ITERATIONS", iterations)?;
        let samples = parse_optional_usize("TENFERRO_PROBE_SAMPLES", samples)?;
        let min_aggregate_ns = min_aggregate_ns
            .map(|value| {
                value.parse::<u128>().map_err(|_| {
                    format!("TENFERRO_PROBE_MIN_AGGREGATE_NS must be an integer, got {value:?}")
                })
            })
            .transpose()?;
        if matches!(mode, ProbeMode::Timed | ProbeMode::Alloc) && stage.is_none() {
            return Err("timed and alloc modes require TENFERRO_PROBE_STAGE".into());
        }
        if mode == ProbeMode::Timed {
            let (Some(iterations), Some(samples), Some(minimum)) =
                (iterations, samples, min_aggregate_ns)
            else {
                return Err(
                    "timed mode requires iterations, samples, and min aggregate duration".into(),
                );
            };
            if iterations == 0 || samples == 0 || minimum == 0 {
                return Err(
                    "timed mode requires nonzero iterations, samples, and min aggregate duration"
                        .into(),
                );
            }
        }
        if mode == ProbeMode::Alloc && iterations == Some(0) {
            return Err("alloc mode requires nonzero iterations".into());
        }
        Ok(Self {
            mode,
            case,
            stage,
            iterations,
            samples,
            min_aggregate_ns,
        })
    }

    fn from_env() -> Result<Self, String> {
        Self::parse(
            &std::env::var("TENFERRO_PROBE_MODE")
                .map_err(|_| "TENFERRO_PROBE_MODE is required".to_string())?,
            std::env::var("TENFERRO_PROBE_CASE").ok().as_deref(),
            std::env::var("TENFERRO_PROBE_STAGE").ok().as_deref(),
            std::env::var("TENFERRO_PROBE_ITERATIONS").ok().as_deref(),
            std::env::var("TENFERRO_PROBE_SAMPLES").ok().as_deref(),
            std::env::var("TENFERRO_PROBE_MIN_AGGREGATE_NS")
                .ok()
                .as_deref(),
        )
    }
}

fn parse_optional_usize(name: &str, value: Option<&str>) -> Result<Option<usize>, String> {
    value
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| format!("{name} must be an integer, got {value:?}"))
        })
        .transpose()
}

fn aggregate_meets_minimum(elapsed_ns: u128, minimum_ns: u128) -> bool {
    elapsed_ns >= minimum_ns
}

fn selected_stage(config: ProbeConfig, case: CaseId) -> Result<Stage, String> {
    let stage = config
        .stage
        .ok_or_else(|| "stage is required for this mode".to_string())?;
    if !stage.applies(case) {
        return Err(format!("stage {} does not apply to case", stage.name()));
    }
    Ok(stage)
}

fn run_timed(config: ProbeConfig, fixture: Fixture) -> Result<(), String> {
    let iterations = config.iterations.unwrap();
    let samples = config.samples.unwrap();
    let minimum_ns = config.min_aggregate_ns.unwrap();
    let stage = selected_stage(config, fixture.case)?;
    let fixed_inputs = fixed_input_refs(&fixture);
    let alternate_inputs = alternate_input_refs(&fixture);
    for sample in 0..samples {
        let start = Instant::now();
        let mut invalid_reason = None;
        let mut completed_iterations = 0;
        for iteration in 0..iterations {
            let result = black_box(execute_stage(
                &fixture,
                &fixed_inputs,
                &alternate_inputs,
                iteration,
                stage,
            ));
            if let Err(reason) = check_stage_result(
                result,
                expected_stage_result(fixture.case, stage, iteration),
            ) {
                invalid_reason = Some(reason);
                break;
            }
            completed_iterations += 1;
        }
        let elapsed_ns = start.elapsed().as_nanos();
        let invalid_reason = invalid_reason.or_else(|| {
            (!aggregate_meets_minimum(elapsed_ns, minimum_ns)).then_some("under_duration")
        });
        println!(
            "{OUTPUT_PREFIX}{{\"kind\":\"timing\",\"binary\":\"{}\",\"source\":\"{}\",\"case_id\":\"{}\",\"stage\":\"{}\",\"sample\":{},\"iterations\":{},\"completed_iterations\":{},\"elapsed_ns\":{},\"valid\":{},\"invalid_reason\":{}}}",
            env!("CARGO_PKG_NAME"),
            concat!(module_path!(), "::", file!()),
            fixture.case.id(),
            stage.name(),
            sample,
            iterations,
            completed_iterations,
            elapsed_ns,
            invalid_reason.is_none(),
            invalid_reason.map_or_else(|| "null".to_string(), |reason| format!("\"{reason}\"")),
        );
        if let Some(reason) = invalid_reason {
            return Err(format!("invalid timing sample: {reason}"));
        }
    }
    Ok(())
}

fn run_allocations(config: ProbeConfig, fixture: Fixture) -> Result<(), String> {
    let iterations = config.iterations.unwrap_or(1);
    let stage = selected_stage(config, fixture.case)?;
    let fixed_inputs = fixed_input_refs(&fixture);
    let alternate_inputs = alternate_input_refs(&fixture);
    let guard = CountingGuard::try_new().map_err(str::to_string)?;
    let mut invalid_reason = None;
    let mut completed_iterations = 0;
    for iteration in 0..iterations {
        let result = black_box(execute_stage(
            &fixture,
            &fixed_inputs,
            &alternate_inputs,
            iteration,
            stage,
        ));
        if let Err(reason) = check_stage_result(
            result,
            expected_stage_result(fixture.case, stage, iteration),
        ) {
            invalid_reason = Some(reason);
            break;
        }
        completed_iterations += 1;
    }
    let snapshot = guard.snapshot();
    drop(guard);
    println!(
        "{OUTPUT_PREFIX}{{\"kind\":\"allocation\",\"binary\":\"{}\",\"source\":\"{}\",\"case_id\":\"{}\",\"stage\":\"{}\",\"iterations\":{},\"completed_iterations\":{},\"allocation_calls\":{},\"requested_bytes\":{},\"valid\":{},\"invalid_reason\":{}}}",
        env!("CARGO_PKG_NAME"),
        concat!(module_path!(), "::", file!()),
        fixture.case.id(),
        stage.name(),
        iterations,
        completed_iterations,
        snapshot.calls,
        snapshot.requested_bytes,
        invalid_reason.is_none(),
        invalid_reason.map_or_else(|| "null".to_string(), |reason| format!("\"{reason}\"")),
    );
    invalid_reason.map_or(Ok(()), |reason| {
        Err(format!("invalid allocation sample: {reason}"))
    })
}

fn assert_fixture_output(
    fixture: &Fixture,
    inputs: &[&Tensor],
    alternate: bool,
) -> crate::Result<()> {
    use crate::TensorEinsumExt;
    use tenferro_cpu::CpuBackend;
    use tenferro_tensor::BackendSessionHost;

    let mut backend = CpuBackend::new();
    let output =
        backend.with_backend_session(|session| inputs.einsum(fixture.case.notation(), session))?;
    match (fixture.case, alternate) {
        (CaseId::Rank2UnaryF64, false) => {
            assert_eq!(output.shape(), &[2, 3]);
            assert_eq!(output.dtype(), DType::F64);
            assert_eq!(output.as_slice::<f64>()?, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        }
        (CaseId::Rank2BinaryF64, false) => {
            assert_eq!(output.shape(), &[2, 2]);
            assert_eq!(output.dtype(), DType::F64);
            assert_eq!(output.as_slice::<f64>()?, &[31.0, 40.0, 58.0, 76.0]);
        }
        (CaseId::Rank4BinaryC64, false) => {
            assert_eq!(output.shape(), &[1, 2, 1, 2]);
            assert_eq!(output.dtype(), DType::C64);
            assert_eq!(
                output.as_slice::<Complex64>()?,
                &[
                    Complex64::new(10.5, -4.5),
                    Complex64::new(15.5, -5.5),
                    Complex64::new(18.5, -6.5),
                    Complex64::new(27.5, -7.5),
                ]
            );
        }
        (CaseId::Rank4FourF64Alternating, false) => {
            assert_eq!(output.shape(), &[] as &[usize]);
            assert_eq!(output.dtype(), DType::F64);
            assert_eq!(output.as_slice::<f64>()?, &[144.0]);
        }
        (CaseId::Rank4FourF64Alternating, true) => {
            assert_eq!(output.shape(), &[] as &[usize]);
            assert_eq!(output.dtype(), DType::F64);
            assert_eq!(output.as_slice::<f64>()?, &[56_784.0]);
        }
        (CaseId::Rank8EightF64, false) => {
            assert_eq!(output.shape(), &[1; 8]);
            assert_eq!(output.dtype(), DType::F64);
            assert_eq!(output.as_slice::<f64>()?, &[40_320.0]);
        }
        (_, true) => unreachable!("only the alternating fixture has alternate correctness data"),
        (CaseId::Rank2BinaryF64CountInvalid, false)
        | (CaseId::Rank2BinaryF64DtypeInvalid, false)
        | (CaseId::Rank2BinaryF64ShapeInvalid, false) => {
            unreachable!("invalid revalidation cases skip numerical correctness")
        }
    }
    Ok(())
}

fn run_correctness(fixture: Fixture) -> Result<(), String> {
    assert_stage_contract(&fixture);
    if fixture.case.invalid_revalidation().is_some() {
        println!(
            "{OUTPUT_PREFIX}{{\"kind\":\"correctness\",\"case_id\":\"{}\",\"stage\":\"prepared_spec_revalidation\",\"ok\":true}}",
            fixture.case.id(),
        );
        return Ok(());
    }
    let fixed_inputs = fixed_input_refs(&fixture);
    assert_fixture_output(&fixture, &fixed_inputs, false)
        .map_err(|_| "execution_failed".to_string())?;
    if fixture.case.alternating() {
        let alternate_inputs = alternate_input_refs(&fixture);
        assert_fixture_output(&fixture, &alternate_inputs, true)
            .map_err(|_| "alternating_execution_failed".to_string())?;
    }
    for stage in STAGES {
        if stage.applies(fixture.case) {
            println!(
                "{OUTPUT_PREFIX}{{\"kind\":\"correctness\",\"case_id\":\"{}\",\"stage\":\"{}\",\"ok\":true}}",
                fixture.case.id(),
                stage.name(),
            );
        }
    }
    Ok(())
}

fn run_probe(config: ProbeConfig) -> Result<(), String> {
    if config.mode == ProbeMode::Contract {
        for case in CASES {
            if config.case.is_some_and(|selected| selected != case) {
                continue;
            }
            for stage in STAGES {
                if stage.applies(case) {
                    println!("{}", contract_record(case, stage));
                }
            }
        }
        return Ok(());
    }
    let case = config
        .case
        .ok_or_else(|| "a case is required".to_string())?;
    let fixture = make_fixture(case);
    match config.mode {
        ProbeMode::Timed => run_timed(config, fixture),
        ProbeMode::Alloc => run_allocations(config, fixture),
        ProbeMode::Correctness => run_correctness(fixture),
        ProbeMode::Contract => unreachable!(),
    }
}

#[test]
#[ignore = "benchmark #95 invokes this entry point in an isolated process"]
fn component_probe_entrypoint() {
    run_probe(ProbeConfig::from_env().unwrap()).unwrap();
}

#[test]
fn probe_stage_contracts_cover_real_private_seams() {
    for case in CASES {
        assert_stage_contract(&make_fixture(case));
    }
}

#[test]
fn probe_contract_export_is_stable_and_timing_free() {
    let records: Vec<_> = CASES
        .into_iter()
        .flat_map(|case| {
            STAGES
                .into_iter()
                .filter(move |stage| stage.applies(case))
                .map(move |stage| contract_record(case, stage))
        })
        .collect();
    assert_eq!(records.len(), 30);
    assert!(records
        .iter()
        .all(|record| record.starts_with(OUTPUT_PREFIX)));
    assert!(records
        .iter()
        .all(|record| record.contains("\"calls_per_workflow\":1")));
    assert!(records.iter().all(|record| !record.contains("elapsed_ns")));
    assert!(records
        .iter()
        .any(|record| record.contains("\"metadata\":\"alternating\"")));
    for (case_id, outcome) in [
        (
            "rank2-binary-f64-count-invalid",
            "count-invalid=validation.invalid_argument",
        ),
        (
            "rank2-binary-f64-dtype-invalid",
            "dtype-invalid=tensor.validation.dtype_mismatch",
        ),
        (
            "rank2-binary-f64-shape-invalid",
            "shape-invalid=validation.shape_mismatch",
        ),
    ] {
        let record = records
            .iter()
            .find(|record| record.contains(&format!("\"case_id\":\"{case_id}\"")))
            .unwrap();
        assert!(record.contains("\"stage\":\"prepared_spec_revalidation\""));
        assert!(record.contains(&format!(
            "\"revalidation_cases\":\"{}\"",
            outcome.split('=').next().unwrap()
        )));
        assert!(record.contains(&format!("\"expected_outcomes\":\"{outcome}\"")));
    }
    assert!(records
        .iter()
        .filter(|record| record.contains("\"case_id\":\"rank2-binary-f64\""))
        .all(|record| record.contains("\"revalidation_cases\":\"none\"")));
}

#[test]
fn probe_configuration_rejects_invalid_and_zero_timing_values() {
    assert!(ProbeConfig::parse(
        "not-a-mode",
        Some("rank2-unary-f64"),
        Some("parse"),
        Some("1"),
        Some("1"),
        Some("1")
    )
    .is_err());
    assert!(ProbeConfig::parse(
        "timed",
        None,
        Some("parse"),
        Some("1"),
        Some("1"),
        Some("1")
    )
    .is_err());
    assert!(ProbeConfig::parse(
        "timed",
        Some("rank2-unary-f64"),
        None,
        Some("1"),
        Some("1"),
        Some("1")
    )
    .is_err());
    assert!(ProbeConfig::parse(
        "timed",
        Some("rank2-unary-f64"),
        Some("parse"),
        Some("0"),
        Some("1"),
        Some("1")
    )
    .is_err());
    assert!(ProbeConfig::parse(
        "timed",
        Some("rank2-unary-f64"),
        Some("parse"),
        Some("1"),
        Some("0"),
        Some("1")
    )
    .is_err());
    assert!(ProbeConfig::parse(
        "timed",
        Some("rank2-unary-f64"),
        Some("parse"),
        Some("1"),
        Some("1"),
        Some("0")
    )
    .is_err());
    assert!(ProbeConfig::parse(
        "alloc",
        Some("rank2-unary-f64"),
        Some("parse"),
        Some("not-an-integer"),
        Some("1"),
        Some("1")
    )
    .is_err());
    assert!(
        ProbeConfig::parse("contract", Some("all"), None, None, None, None)
            .unwrap()
            .case
            .is_none()
    );
    assert!(ProbeConfig::parse(
        "alloc",
        Some("rank2-unary-f64"),
        Some("parse"),
        Some("0"),
        None,
        None
    )
    .is_err());
    assert!(ProbeConfig::parse(
        "alloc",
        Some("rank2-unary-f64"),
        None,
        Some("1"),
        None,
        None
    )
    .is_err());
}

#[test]
fn probe_under_duration_is_rejected() {
    assert!(!aggregate_meets_minimum(9, 10));
    assert!(aggregate_meets_minimum(10, 10));
}

#[test]
fn probe_allocator_counts_caller_allocations_and_restores_on_unwind() {
    let no_allocations = {
        let guard = CountingGuard::try_new().unwrap();
        black_box(42usize);
        let snapshot = guard.snapshot();
        drop(guard);
        snapshot
    };
    assert_eq!(no_allocations.calls, 0);
    assert_eq!(no_allocations.requested_bytes, 0);

    let allocations = {
        let guard = CountingGuard::try_new().unwrap();
        black_box(vec![1usize, 2, 3, 4]);
        let snapshot = guard.snapshot();
        drop(guard);
        snapshot
    };
    assert!(allocations.calls > 0);
    assert!(allocations.requested_bytes >= 4 * std::mem::size_of::<usize>());

    let result = catch_unwind(AssertUnwindSafe(|| {
        let _guard = CountingGuard::try_new().unwrap();
        assert!(CountingGuard::try_new().is_err());
        panic!("probe guard test");
    }));
    assert!(result.is_err());
    assert!(!COUNTING.try_with(Cell::get).unwrap_or(false));
}

#[test]
fn probe_allocator_forwards_zeroed_realloc_and_dealloc() {
    let guard = CountingGuard::try_new().unwrap();
    // SAFETY: the layout is valid, and each pointer is handled by the matching allocator call.
    unsafe {
        let layout = Layout::from_size_align(8, 8).unwrap();
        let ptr = std::alloc::alloc_zeroed(layout);
        assert!(!ptr.is_null());
        let new_layout = Layout::from_size_align(16, 8).unwrap();
        let ptr = std::alloc::realloc(ptr, layout, new_layout.size());
        assert!(!ptr.is_null());
        std::alloc::dealloc(ptr, new_layout);
    }
    let snapshot = guard.snapshot();
    drop(guard);
    assert!(snapshot.calls >= 2);
    assert!(snapshot.requested_bytes >= 24);
}

#[test]
fn probe_revalidation_reports_typed_count_dtype_and_shape_errors() {
    let fixture = make_fixture(CaseId::Rank2BinaryF64);
    let count_error = fixture
        .prepared
        .validate_inputs(&fixture.specs[..1], VALIDATION_OP)
        .unwrap_err();
    assert!(matches!(
        count_error,
        Error::Validation {
            source: ValidationError::InvalidArgument {
                argument: "inputs",
                ..
            },
            ..
        }
    ));

    let dtype_error = fixture
        .prepared
        .validate_inputs(
            &[
                ConcreteEinsumInputSpec {
                    dtype: DType::C64,
                    shape: fixture.specs[0].shape.clone(),
                },
                fixture.specs[1].clone(),
            ],
            VALIDATION_OP,
        )
        .unwrap_err();
    assert!(matches!(
        dtype_error,
        Error::Tensor(tenferro_tensor::Error::Validation {
            source: ValidationError::DTypeMismatch { .. },
            ..
        })
    ));

    let shape_error = fixture
        .prepared
        .validate_inputs(
            &[
                ConcreteEinsumInputSpec {
                    dtype: DType::F64,
                    shape: vec![4, 3],
                },
                fixture.specs[1].clone(),
            ],
            VALIDATION_OP,
        )
        .unwrap_err();
    assert!(matches!(
        shape_error,
        Error::Validation {
            source: ValidationError::ShapeMismatch(_),
            ..
        }
    ));
}

#[test]
fn probe_numerical_sanity_checks_known_real_and_complex_values() {
    use crate::TensorEinsumExt;
    use tenferro_cpu::CpuBackend;
    use tenferro_tensor::BackendSessionHost;

    let lhs =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs =
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let complex_lhs = Tensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0)],
    )
    .unwrap();
    let complex_rhs = Tensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(2.0, -1.0), Complex64::new(4.0, 2.0)],
    )
    .unwrap();
    let mut backend = CpuBackend::new();
    let (real, complex) = backend
        .with_backend_session(|session| {
            Ok::<_, Error>((
                [&lhs, &rhs].einsum(BINARY_NOTATION, session)?,
                [&complex_lhs, &complex_rhs].einsum("i,i->", session)?,
            ))
        })
        .unwrap();
    assert_eq!(real.shape(), &[2, 2]);
    assert_eq!(real.as_slice::<f64>().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
    assert_eq!(
        complex.as_slice::<Complex64>().unwrap(),
        &[Complex64::new(18.0, 5.0)]
    );
}

#[test]
fn probe_allocator_is_disabled_after_normal_use() {
    assert!(!COUNTING.with(Cell::get));
    black_box(vec![1usize, 2, 3]);
    assert!(!COUNTING.with(Cell::get));
}
