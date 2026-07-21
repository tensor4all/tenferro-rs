use std::alloc::{GlobalAlloc, Layout, System};
use std::fmt::Write as _;
use std::hint::black_box;
use std::io::{self, Write as _};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{DotGeneralConfig, SliceConfig, Tensor, TensorRead};

const WARMUP_REPETITIONS: usize = 256;
const MEASURED_REPETITIONS: usize = 4_096;
#[cfg(test)]
const TRIANGULAR_REPETITION_FACTOR: f64 = 8_390_656.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Snapshot {
    allocations: u64,
    bytes: u64,
    failures: u64,
    overflow: bool,
}

impl Snapshot {
    #[cfg(test)]
    const fn valid(allocations: u64, bytes: u64) -> Self {
        Self {
            allocations,
            bytes,
            failures: 0,
            overflow: false,
        }
    }

    fn is_valid(self) -> bool {
        self.failures == 0 && !self.overflow
    }
}

struct CounterState {
    allocations: AtomicU64,
    bytes: AtomicU64,
    failures: AtomicU64,
    overflow: AtomicBool,
    recording: AtomicBool,
}

impl CounterState {
    const fn disabled() -> Self {
        Self {
            allocations: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            failures: AtomicU64::new(0),
            overflow: AtomicBool::new(false),
            recording: AtomicBool::new(false),
        }
    }

    #[cfg(test)]
    const fn new() -> Self {
        Self {
            recording: AtomicBool::new(true),
            ..Self::disabled()
        }
    }

    fn checked_add(&self, counter: &AtomicU64, delta: u64) {
        if counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(delta)
            })
            .is_err()
        {
            self.overflow.store(true, Ordering::Relaxed);
        }
    }

    fn record_success(&self, bytes: usize) {
        if !self.recording.load(Ordering::Relaxed) {
            return;
        }
        self.checked_add(&self.allocations, 1);
        match u64::try_from(bytes) {
            Ok(bytes) => self.checked_add(&self.bytes, bytes),
            Err(_) => self.overflow.store(true, Ordering::Relaxed),
        }
    }

    fn record_failure(&self) {
        if self.recording.load(Ordering::Relaxed) {
            self.checked_add(&self.failures, 1);
        }
    }

    /// Reset all counters while no allocator call can run concurrently.
    ///
    /// This quiescence requirement is part of the probe protocol. Relaxed
    /// atomics prevent data races but do not make concurrent reset valid.
    fn reset(&self) {
        self.recording.store(false, Ordering::Relaxed);
        self.allocations.store(0, Ordering::Relaxed);
        self.bytes.store(0, Ordering::Relaxed);
        self.failures.store(0, Ordering::Relaxed);
        self.overflow.store(false, Ordering::Relaxed);
        self.recording.store(true, Ordering::Relaxed);
    }

    /// Stop recording and snapshot at a protocol-defined quiescent point.
    fn finish_and_snapshot(&self) -> Snapshot {
        self.recording.store(false, Ordering::Relaxed);
        self.snapshot()
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot {
            allocations: self.allocations.load(Ordering::Relaxed),
            bytes: self.bytes.load(Ordering::Relaxed),
            failures: self.failures.load(Ordering::Relaxed),
            overflow: self.overflow.load(Ordering::Relaxed),
        }
    }

    #[cfg(test)]
    fn seed(&self, allocations: u64, bytes: u64, failures: u64) {
        self.recording.store(false, Ordering::Relaxed);
        self.allocations.store(allocations, Ordering::Relaxed);
        self.bytes.store(bytes, Ordering::Relaxed);
        self.failures.store(failures, Ordering::Relaxed);
        self.overflow.store(false, Ordering::Relaxed);
        self.recording.store(true, Ordering::Relaxed);
    }
}

struct CountingSystem;

static COUNTERS: CounterState = CounterState::disabled();

// SAFETY: Every method delegates the caller's GlobalAlloc contract unchanged
// to System exactly once. The wrapper never dereferences allocation pointers.
unsafe impl GlobalAlloc for CountingSystem {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: GlobalAlloc guarantees that layout is valid; it is forwarded
        // unchanged to the sole delegate, System.
        let pointer = unsafe { System.alloc(layout) };
        if pointer.is_null() {
            COUNTERS.record_failure();
        } else {
            COUNTERS.record_success(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: GlobalAlloc guarantees that layout is valid; it is forwarded
        // unchanged to the sole delegate, System.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if pointer.is_null() {
            COUNTERS.record_failure();
        } else {
            COUNTERS.record_success(layout.size());
        }
        pointer
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: GlobalAlloc guarantees that pointer/layout identify a live
        // System allocation and new_size satisfies the realloc contract.
        let result = unsafe { System.realloc(pointer, layout, new_size) };
        if result.is_null() {
            COUNTERS.record_failure();
        } else {
            COUNTERS.record_success(new_size);
        }
        result
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: GlobalAlloc guarantees that pointer/layout identify a live
        // allocation from this allocator, whose sole delegate is System.
        unsafe { System.dealloc(pointer, layout) };
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingSystem = CountingSystem;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Lazy,
    Materialized,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Operation {
    Neg,
    Add,
    Reduce,
    Slice,
    Dot,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Case {
    name: &'static str,
    mode: Mode,
    operation: Operation,
    size: usize,
}

macro_rules! case {
    ($name:literal, $mode:ident, $operation:ident, $size:literal) => {
        Case {
            name: $name,
            mode: Mode::$mode,
            operation: Operation::$operation,
            size: $size,
        }
    };
}

const CASES: [Case; 28] = [
    case!("lazy_neg_1", Lazy, Neg, 1),
    case!("lazy_neg_8", Lazy, Neg, 8),
    case!("lazy_neg_64", Lazy, Neg, 64),
    case!("lazy_add_1", Lazy, Add, 1),
    case!("lazy_add_8", Lazy, Add, 8),
    case!("lazy_add_64", Lazy, Add, 64),
    case!("lazy_reduce_1", Lazy, Reduce, 1),
    case!("lazy_reduce_8", Lazy, Reduce, 8),
    case!("lazy_reduce_64", Lazy, Reduce, 64),
    case!("lazy_slice_1", Lazy, Slice, 1),
    case!("lazy_slice_8", Lazy, Slice, 8),
    case!("lazy_slice_64", Lazy, Slice, 64),
    case!("lazy_dot_1", Lazy, Dot, 1),
    case!("lazy_dot_2", Lazy, Dot, 2),
    case!("materialized_neg_1", Materialized, Neg, 1),
    case!("materialized_neg_8", Materialized, Neg, 8),
    case!("materialized_neg_64", Materialized, Neg, 64),
    case!("materialized_add_1", Materialized, Add, 1),
    case!("materialized_add_8", Materialized, Add, 8),
    case!("materialized_add_64", Materialized, Add, 64),
    case!("materialized_reduce_1", Materialized, Reduce, 1),
    case!("materialized_reduce_8", Materialized, Reduce, 8),
    case!("materialized_reduce_64", Materialized, Reduce, 64),
    case!("materialized_slice_1", Materialized, Slice, 1),
    case!("materialized_slice_8", Materialized, Slice, 8),
    case!("materialized_slice_64", Materialized, Slice, 64),
    case!("materialized_dot_1", Materialized, Dot, 1),
    case!("materialized_dot_2", Materialized, Dot, 2),
];

struct CaseInputs {
    lhs: EagerTensor,
    rhs: EagerTensor,
    slice: SliceConfig,
    dot: DotGeneralConfig,
}

impl CaseInputs {
    fn new(case: Case) -> Result<Self, String> {
        let runtime = EagerRuntime::with_cpu_backend(
            CpuBackend::with_threads(1).map_err(|error| error.to_string())?,
        );
        let shape = if case.operation == Operation::Dot {
            vec![case.size, case.size]
        } else {
            vec![case.size]
        };
        let len = shape.iter().product();
        let lhs = eager(&runtime, shape.clone(), len)?;
        let rhs = eager(&runtime, shape, len)?;
        Ok(Self {
            lhs,
            rhs,
            slice: SliceConfig {
                starts: vec![0],
                limits: vec![case.size],
                strides: vec![1],
            },
            dot: DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        })
    }

    fn execute(&self, operation: Operation) -> Result<EagerTensor, String> {
        match operation {
            Operation::Neg => black_box(&self.lhs).neg(),
            Operation::Add => black_box(&self.lhs).add(black_box(&self.rhs)),
            Operation::Reduce => black_box(&self.lhs).reduce_sum(Some(black_box(&[0_usize]))),
            Operation::Slice => black_box(&self.lhs).slice(black_box(self.slice.clone())),
            Operation::Dot => {
                black_box(&self.lhs).dot_general(black_box(&self.rhs), black_box(self.dot.clone()))
            }
        }
        .map_err(|error| error.to_string())
    }
}

fn eager(
    runtime: &Arc<EagerRuntime>,
    shape: Vec<usize>,
    len: usize,
) -> Result<EagerTensor, String> {
    let tensor =
        Tensor::from_vec_col_major(shape, vec![1.0_f64; len]).map_err(|error| error.to_string())?;
    EagerTensor::from_tensor_in(tensor, Arc::clone(runtime)).map_err(|error| error.to_string())
}

fn shape_token(shape: &[usize]) -> u64 {
    shape.iter().fold(shape.len() as u64, |token, &dimension| {
        token.wrapping_mul(257).wrapping_add(dimension as u64)
    })
}

fn consume(output: EagerTensor, mode: Mode) -> Result<f64, String> {
    match mode {
        Mode::Lazy => {
            let shape = black_box(output.shape());
            let shape = black_box(shape_token(shape)) as f64;
            let storage_tag = match output.tensor_read() {
                TensorRead::Tensor(tensor) => {
                    black_box(tensor);
                    1.0
                }
                TensorRead::View(view) => {
                    black_box(view);
                    2.0
                }
            };
            Ok(shape + storage_tag)
        }
        Mode::Materialized => {
            let tensor = output.materialized().map_err(|error| error.to_string())?;
            let shape = black_box(shape_token(black_box(tensor.shape()))) as f64;
            let first = *black_box(
                tensor
                    .as_slice::<f64>()
                    .map_err(|error| error.to_string())?,
            )
            .first()
            .ok_or_else(|| "materialized output is empty".to_owned())?;
            Ok(shape + black_box(first))
        }
    }
}

fn run_case(case: Case) -> Result<(Snapshot, f64), String> {
    let inputs = CaseInputs::new(case)?;
    for _ in 0..WARMUP_REPETITIONS {
        let output = inputs.execute(case.operation)?;
        let token = consume(output, case.mode)?;
        black_box(token);
    }

    COUNTERS.reset();
    let measured = (|| {
        let mut checksum = 0.0;
        for index in 0..MEASURED_REPETITIONS {
            let output = inputs.execute(case.operation)?;
            let token = consume(black_box(output), case.mode)?;
            checksum += token * ((index + 1) as f64);
            black_box(checksum);
        }
        Ok::<f64, String>(checksum)
    })();
    let snapshot = COUNTERS.finish_and_snapshot();
    let checksum = measured?;
    if !checksum.is_finite() {
        return Err("measured checksum is not finite".to_owned());
    }
    Ok((snapshot, checksum))
}

fn case_by_name(name: &str) -> Option<Case> {
    CASES.iter().copied().find(|case| case.name == name)
}

fn case_inventory_json() -> String {
    let mut output = String::from("[");
    for (index, case) in CASES.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write!(output, "\"{}\"", case.name).expect("writing to String cannot fail");
    }
    output.push_str("]\n");
    output
}

fn record_json(case: Case, snapshot: Snapshot, checksum: f64) -> Result<String, String> {
    if !checksum.is_finite() {
        return Err("cannot serialize a non-finite checksum".to_owned());
    }
    Ok(format!(
        concat!(
            "{{\"allocated_bytes\":{},\"allocation_count\":{},",
            "\"allocation_failures\":{},\"case\":\"{}\",",
            "\"checksum\":{:e},\"counter_overflow\":{},",
            "\"repetitions\":{}}}\n"
        ),
        snapshot.bytes,
        snapshot.allocations,
        snapshot.failures,
        case.name,
        checksum,
        snapshot.overflow,
        MEASURED_REPETITIONS,
    ))
}

fn write_stdout(bytes: &[u8]) -> Result<(), String> {
    let mut stdout = io::stdout().lock();
    stdout
        .write_all(bytes)
        .and_then(|()| stdout.flush())
        .map_err(|error| format!("cannot write stdout: {error}"))
}

fn run() -> Result<ExitCode, String> {
    let mut arguments = std::env::args_os();
    let _program = arguments.next();
    let argument = arguments
        .next()
        .ok_or_else(|| "expected one canonical case or --list-cases".to_owned())?;
    if arguments.next().is_some() {
        return Err("expected exactly one argument".to_owned());
    }
    let argument = argument
        .into_string()
        .map_err(|_| "argument is not valid UTF-8".to_owned())?;
    if argument == "--list-cases" {
        write_stdout(case_inventory_json().as_bytes())?;
        return Ok(ExitCode::SUCCESS);
    }
    let case = case_by_name(&argument).ok_or_else(|| format!("unknown case: {argument}"))?;
    let (snapshot, checksum) = run_case(case)?;
    let record = record_json(case, snapshot, checksum)?;
    write_stdout(record.as_bytes())?;
    Ok(if snapshot.is_valid() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    })
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("phase2e allocation probe: {error}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests;
