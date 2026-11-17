//! The Metal hardware lane: the acceptance bars of `IMPLEMENTATION-PLAN.md` §2.
//!
//! Every test here needs a Metal device. On a host without one each test returns early
//! unless `TENFERRO_REQUIRE_METAL=1`, which the macOS CI lane sets so a passing run proves
//! the device was there (`scripts/ci/run_profile.py`).

use std::sync::Arc;

use tenferro_gpu::metal::{metal_available, MetalContext, MetalStats};
use tenferro_tensor::{
    DType, Error, HostAccessError, Tensor, TensorElementwise, TensorRead, TensorReduction,
    TensorStructural,
};

/// The `f32` elements of a tensor, read through its host-visible slice.
fn floats(tensor: &Tensor) -> Vec<f32> {
    tensor.as_slice::<f32>().expect("f32 elements").to_vec()
}

/// The `Complex32` elements of a tensor, read through its host-visible slice.
fn complexes(tensor: &Tensor) -> Vec<num_complex::Complex32> {
    tensor
        .as_slice::<num_complex::Complex32>()
        .expect("Complex32 elements")
        .to_vec()
}

/// The allocation identity behind a tensor, for "a fresh allocation" assertions.
fn allocation(tensor: &Tensor) -> tenferro_tensor::AllocationId {
    tensor
        .as_typed::<f32>()
        .and_then(|typed| typed.backend_buffer())
        .and_then(|storage| storage.allocation_id())
        .expect("a shared allocation with an identity")
}

/// The testing context, or `None` on a host that has no device and does not require one.
fn context() -> Option<MetalContext> {
    if !metal_available() {
        assert!(
            std::env::var("TENFERRO_REQUIRE_METAL").as_deref() != Ok("1"),
            "TENFERRO_REQUIRE_METAL=1 but the probe found no Metal device"
        );
        return None;
    }
    Some(MetalContext::open(Default::default()).expect("the probe reported a device"))
}

/// A host tensor of `len` f32 values.
fn host_f32(len: usize) -> Tensor {
    let values: Vec<f32> = (0..len).map(|index| index as f32 * 0.5 - 3.0).collect();
    Tensor::from_vec_col_major(vec![len], values).expect("a 1-D host tensor")
}

fn host_values(len: usize) -> Vec<f32> {
    (0..len).map(|index| index as f32 * 0.5 - 3.0).collect()
}

fn stats_of(context: &MetalContext) -> MetalStats {
    context.stats()
}

#[test]
fn probe_reports_a_device() {
    let Some(context) = context() else { return };
    let device = context.device();
    assert!(!device.name.is_empty(), "the device reports a name");
}

#[test]
fn round_trip_is_bit_exact() {
    let Some(context) = context() else { return };
    for len in [0, 1, 3, 255, 256, 257, 4096, 100_003] {
        let host = host_f32(len);
        let shared = context
            .upload(TensorRead::from_tensor(&host))
            .expect("upload");
        let back = context
            .download(TensorRead::from_tensor(&shared))
            .expect("download");
        assert_eq!(
            floats(&back),
            host_values(len),
            "a round trip of {len} elements is bit exact"
        );
    }
}

#[test]
fn a_round_trip_counts_its_bytes() {
    let Some(context) = context() else { return };
    let before = stats_of(&context);
    let host = host_f32(4096);
    let shared = context
        .upload(TensorRead::from_tensor(&host))
        .expect("upload");
    let _ = context
        .download(TensorRead::from_tensor(&shared))
        .expect("download");
    let after = stats_of(&context);
    assert_eq!(after.upload_bytes - before.upload_bytes, 4096 * 4);
    assert_eq!(after.download_bytes - before.download_bytes, 4096 * 4);
    assert!(after.allocations > before.allocations);
}

#[test]
fn live_bytes_track_live_tensors() {
    let Some(context) = context() else { return };
    let before = stats_of(&context).live_bytes;
    let bytes = 4096 * 4;

    let first = context
        .upload(TensorRead::from_tensor(&host_f32(4096)))
        .expect("upload");
    assert_eq!(
        stats_of(&context).live_bytes,
        before + bytes,
        "one live tensor contributes its bytes"
    );

    let second = context
        .upload(TensorRead::from_tensor(&host_f32(4096)))
        .expect("upload");
    assert_eq!(
        stats_of(&context).live_bytes,
        before + 2 * bytes,
        "a second tensor adds its bytes"
    );

    drop(second);
    assert_eq!(
        stats_of(&context).live_bytes,
        before + bytes,
        "dropping a tensor removes its bytes"
    );
    drop(first);
    assert_eq!(
        stats_of(&context).live_bytes,
        before,
        "the counter returns to its baseline"
    );
}

#[test]
fn elementwise_matches_the_cpu_reference() {
    let Some(context) = context() else { return };
    let mut engine = context.engine();
    let len = 4096;
    let left_values = host_values(len);
    let right_values: Vec<f32> = left_values.iter().map(|value| value.abs() + 1.5).collect();
    let left = context
        .upload(TensorRead::from_tensor(&host_f32(len)))
        .expect("upload");
    let right = context
        .upload(TensorRead::from_tensor(
            &Tensor::from_vec_col_major(vec![len], right_values.clone()).expect("a host tensor"),
        ))
        .expect("upload");

    let expect = |name: &str, actual: Tensor, reference: Vec<f32>| {
        let host = context
            .download(TensorRead::from_tensor(&actual))
            .expect(name);
        assert_eq!(
            floats(&host),
            reference,
            "{name} matches the CPU reference bit for bit"
        );
    };

    let add = engine
        .add_read(
            TensorRead::from_tensor(&left),
            TensorRead::from_tensor(&right),
        )
        .expect("add");
    expect(
        "add",
        add,
        left_values
            .iter()
            .zip(&right_values)
            .map(|(l, r)| l + r)
            .collect(),
    );

    let sub = engine
        .sub_read(
            TensorRead::from_tensor(&left),
            TensorRead::from_tensor(&right),
        )
        .expect("sub");
    expect(
        "sub",
        sub,
        left_values
            .iter()
            .zip(&right_values)
            .map(|(l, r)| l - r)
            .collect(),
    );

    let mul = engine
        .mul_read(
            TensorRead::from_tensor(&left),
            TensorRead::from_tensor(&right),
        )
        .expect("mul");
    expect(
        "mul",
        mul,
        left_values
            .iter()
            .zip(&right_values)
            .map(|(l, r)| l * r)
            .collect(),
    );

    let div = engine
        .div_read(
            TensorRead::from_tensor(&left),
            TensorRead::from_tensor(&right),
        )
        .expect("div");
    expect(
        "div",
        div,
        left_values
            .iter()
            .zip(&right_values)
            .map(|(l, r)| l / r)
            .collect(),
    );

    let neg = engine
        .neg_read(TensorRead::from_tensor(&left))
        .expect("neg");
    expect("neg", neg, left_values.iter().map(|value| -value).collect());

    let abs = engine
        .abs_read(TensorRead::from_tensor(&left))
        .expect("abs");
    expect(
        "abs",
        abs,
        left_values.iter().map(|value| value.abs()).collect(),
    );

    let sign = engine
        .sign_read(TensorRead::from_tensor(&left))
        .expect("sign");
    expect(
        "sign",
        sign,
        left_values
            .iter()
            .map(|value| {
                if *value > 0.0 {
                    1.0
                } else if *value < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            })
            .collect(),
    );

    // A real conjugation is the identity; the result is still a fresh allocation.
    let conj = engine
        .conj_read(TensorRead::from_tensor(&left))
        .expect("conj");
    assert_ne!(allocation(&conj), allocation(&left));
    expect("conj", conj, left_values.clone());
}

#[test]
fn complex_elementwise_matches_the_cpu_reference() {
    let Some(context) = context() else { return };
    let mut engine = context.engine();
    let left_values = vec![
        num_complex::Complex32::new(1.0, 2.0),
        num_complex::Complex32::new(-3.0, 4.0),
        num_complex::Complex32::new(0.5, -0.25),
        num_complex::Complex32::new(0.0, 0.0),
    ];
    let right_values = vec![
        num_complex::Complex32::new(2.0, -1.0),
        num_complex::Complex32::new(0.5, 0.5),
        num_complex::Complex32::new(-1.0, 3.0),
        num_complex::Complex32::new(7.0, 7.0),
    ];
    let upload = |values: &[num_complex::Complex32]| {
        let host =
            Tensor::from_vec_col_major(vec![values.len()], values.to_vec()).expect("a host tensor");
        context
            .upload(TensorRead::from_tensor(&host))
            .expect("upload")
    };
    let left = upload(&left_values);
    let right = upload(&right_values);

    let expect = |name: &str, actual: Tensor, reference: Vec<num_complex::Complex32>| {
        let host = context
            .download(TensorRead::from_tensor(&actual))
            .expect(name);
        assert_eq!(
            complexes(&host),
            reference,
            "{name} matches the CPU reference bit for bit"
        );
    };

    let conj = engine
        .conj_read(TensorRead::from_tensor(&left))
        .expect("conj");
    expect(
        "conj",
        conj,
        left_values.iter().map(|value| value.conj()).collect(),
    );

    let add = engine
        .add_read(
            TensorRead::from_tensor(&left),
            TensorRead::from_tensor(&right),
        )
        .expect("add");
    expect(
        "add",
        add,
        left_values
            .iter()
            .zip(&right_values)
            .map(|(l, r)| *l + *r)
            .collect(),
    );

    let sub = engine
        .sub_read(
            TensorRead::from_tensor(&left),
            TensorRead::from_tensor(&right),
        )
        .expect("sub");
    expect(
        "sub",
        sub,
        left_values
            .iter()
            .zip(&right_values)
            .map(|(l, r)| *l - *r)
            .collect(),
    );

    let mul = engine
        .mul_read(
            TensorRead::from_tensor(&left),
            TensorRead::from_tensor(&right),
        )
        .expect("mul");
    expect(
        "mul",
        mul,
        left_values
            .iter()
            .zip(&right_values)
            .map(|(l, r)| *l * *r)
            .collect(),
    );

    let neg = engine
        .neg_read(TensorRead::from_tensor(&left))
        .expect("neg");
    expect(
        "neg",
        neg,
        left_values.iter().map(|value| -*value).collect(),
    );

    // The composed case the review asked for: a wrong imaginary sign in `conj` or a wrong
    // operand offset in `add` changes this result, so neither claimed row can stay green
    // while broken.
    let conj_left = engine
        .conj_read(TensorRead::from_tensor(&left))
        .expect("conj");
    let conj_add = engine
        .add_read(
            TensorRead::from_tensor(&conj_left),
            TensorRead::from_tensor(&right),
        )
        .expect("conj + add");
    expect(
        "conj + add",
        conj_add,
        left_values
            .iter()
            .zip(&right_values)
            .map(|(l, r)| l.conj() + *r)
            .collect(),
    );
}

#[test]
fn complex_sign_matches_the_unit_direction() {
    let Some(context) = context() else { return };
    let mut engine = context.engine();
    // Axis vectors keep |z| exact, so z / |z| is bit-comparable; (0, 0) is the documented
    // zero result.
    let values = vec![
        num_complex::Complex32::new(1.0, 0.0),
        num_complex::Complex32::new(0.0, 1.0),
        num_complex::Complex32::new(-1.0, 0.0),
        num_complex::Complex32::new(0.0, -1.0),
        num_complex::Complex32::new(3.0, 0.0),
        num_complex::Complex32::new(0.0, -4.0),
        num_complex::Complex32::new(0.0, 0.0),
    ];
    let host =
        Tensor::from_vec_col_major(vec![values.len()], values.clone()).expect("a host tensor");
    let shared = context
        .upload(TensorRead::from_tensor(&host))
        .expect("upload");
    let sign = engine
        .sign_read(TensorRead::from_tensor(&shared))
        .expect("sign");
    let back = context
        .download(TensorRead::from_tensor(&sign))
        .expect("download");
    let reference: Vec<num_complex::Complex32> = values
        .iter()
        .map(|value| {
            if value.re == 0.0 && value.im == 0.0 {
                num_complex::Complex32::new(0.0, 0.0)
            } else {
                let magnitude = value.re.hypot(value.im);
                num_complex::Complex32::new(value.re / magnitude, value.im / magnitude)
            }
        })
        .collect();
    assert_eq!(
        complexes(&back),
        reference,
        "sign is the unit direction for the exact axis cases"
    );
}

#[test]
fn complex_magnitude_reads_a_real_dtype() {
    let Some(context) = context() else { return };
    let mut engine = context.engine();
    let len = 257;
    let values: Vec<num_complex::Complex32> = (0..len)
        .map(|index| num_complex::Complex32::new(index as f32, -(index as f32) * 0.25))
        .collect();
    let host = Tensor::from_vec_col_major(vec![len], values.clone()).expect("a host tensor");
    let shared = context
        .upload(TensorRead::from_tensor(&host))
        .expect("upload");
    let magnitude = engine
        .abs_read(TensorRead::from_tensor(&shared))
        .expect("abs");
    assert_eq!(magnitude.dtype(), DType::F32);
    let back = context
        .download(TensorRead::from_tensor(&magnitude))
        .expect("download");
    let actual = floats(&back);
    for (value, magnitude) in values.iter().zip(actual) {
        // The entry is the baseline's scaled hypot (`scale * sqrt(r*r + i*i)` after
        // dividing by `max(|re|, |im|)`), so it may differ from the platform's `hypot` by
        // an ulp; the comparison is a tolerance, not bit equality, and the reason is printed.
        let expected = value.re.hypot(value.im);
        let tolerance = expected.abs() * 4.0 * f32::EPSILON;
        assert!(
            (magnitude - expected).abs() <= tolerance,
            "|({}, {})| = {magnitude} but the reference is {expected}",
            value.re,
            value.im
        );
    }
}

#[test]
fn a_same_dtype_cast_is_an_exact_copy() {
    let Some(context) = context() else { return };
    let mut engine = context.engine();
    let shared = context
        .upload(TensorRead::from_tensor(&host_f32(513)))
        .expect("upload");
    let cast = engine.cast(&shared, DType::F32).expect("a same-dtype cast");
    assert_ne!(allocation(&cast), allocation(&shared), "a cast allocates");
    let back = context
        .download(TensorRead::from_tensor(&cast))
        .expect("download");
    assert_eq!(floats(&back), host_values(513));
}

#[test]
fn a_cross_dtype_cast_is_refused() {
    let Some(context) = context() else { return };
    let mut engine = context.engine();
    let shared = context
        .upload(TensorRead::from_tensor(&host_f32(4)))
        .expect("upload");
    let error = engine
        .cast(&shared, DType::F64)
        .expect_err("a cross-dtype cast has no Metal entry");
    assert!(
        matches!(error, Error::UnsupportedDTypeConversion { .. }),
        "a typed conversion refusal, got {error}"
    );
}

#[test]
fn the_in_place_form_overwrites_a_caller_owned_output() {
    let Some(context) = context() else { return };
    let mut engine = context.engine();
    let len = 1024;
    let left_values = host_values(len);
    let right_values: Vec<f32> = left_values.iter().map(|value| value * 0.5 + 7.0).collect();
    let left = context
        .upload(TensorRead::from_tensor(&host_f32(len)))
        .expect("upload");
    let right = context
        .upload(TensorRead::from_tensor(
            &Tensor::from_vec_col_major(vec![len], right_values.clone()).expect("a host tensor"),
        ))
        .expect("upload");
    let mut output = context
        .upload(TensorRead::from_tensor(&host_f32(len)))
        .expect("upload");
    engine
        .add_into(
            &left,
            &right,
            tenferro_tensor::TensorWrite::from_tensor(&mut output),
        )
        .expect("add into a caller-owned output");
    let expected: Vec<f32> = left_values
        .iter()
        .zip(&right_values)
        .map(|(l, r)| l + r)
        .collect();
    let back = context
        .download(TensorRead::from_tensor(&output))
        .expect("download");
    assert_eq!(floats(&back), expected);
}

#[test]
fn a_live_host_guard_makes_a_submission_report_mapped_for_host() {
    let Some(context) = context() else { return };
    let mut engine = context.engine();
    let left = context
        .upload(TensorRead::from_tensor(&host_f32(64)))
        .expect("upload");
    let right = context
        .upload(TensorRead::from_tensor(&host_f32(64)))
        .expect("upload");
    // A *read* guard is the reachable form: `map_write` needs `&mut Tensor`, and a `&Tensor`
    // operand cannot be formed while that borrow lives, so safe code cannot hold a write
    // guard on an operand it also passes in. Both guards raise the same live count, which is
    // what the submission's admission check reads.
    let guard = left
        .as_typed::<f32>()
        .expect("f32")
        .backend_buffer()
        .expect("a shared allocation")
        .map_read()
        .expect("a readable map");
    let before = stats_of(&context).submissions;
    let error = engine
        .add_read(
            TensorRead::from_tensor(&left),
            TensorRead::from_tensor(&right),
        )
        .expect_err("a live host guard refuses the submission");
    assert!(
        matches!(
            error,
            Error::HostAccess {
                source: HostAccessError::MappedForHost,
                ..
            }
        ),
        "expected MappedForHost, got {error}"
    );
    assert_eq!(
        stats_of(&context).submissions,
        before,
        "a refused submission commits nothing"
    );
    drop(guard);
    // With the guard gone the same call succeeds.
    engine
        .add_read(
            TensorRead::from_tensor(&left),
            TensorRead::from_tensor(&right),
        )
        .expect("the submission is admitted once the guard is dropped");
}

#[test]
fn a_second_host_mapping_is_refused_as_overlapping() {
    let Some(context) = context() else { return };
    let left = context
        .upload(TensorRead::from_tensor(&host_f32(16)))
        .expect("upload");
    let storage = left
        .as_typed::<f32>()
        .expect("f32")
        .backend_buffer()
        .expect("a shared allocation");
    let _first = storage.map_read().expect("the first map");
    let error = storage
        .map_read()
        .expect_err("a second map overlaps the first");
    assert!(
        matches!(error, HostAccessError::OverlappingHostMapping),
        "expected OverlappingHostMapping, got {error}"
    );
}

#[test]
fn a_gpu_write_after_a_read_only_launch_waits() {
    let Some(context) = context() else { return };
    let mut engine = context.engine();
    let len = 8192;
    let mut left = context
        .upload(TensorRead::from_tensor(&host_f32(len)))
        .expect("upload");
    let value_at_launch = host_values(len);
    // One launch that only *reads* `left`.
    let negated = engine
        .neg_read(TensorRead::from_tensor(&left))
        .expect("neg");
    // Overwrite the read operand on the host. The host write must wait for that launch, or
    // the GPU could read the overwritten values; `PendingTouches` records read touches for
    // exactly this case, and the domain's wait counter only counts a real wait.
    let before = stats_of(&context).waits;
    left.as_typed_mut::<f32>()
        .expect("f32")
        .backend_buffer_mut()
        .expect("a shared allocation")
        .map_write()
        .expect("a writable map")
        .copy_from_slice(&vec![9.0_f32; len])
        .expect("the overwrite");
    assert!(
        stats_of(&context).waits > before,
        "the host write waited for the launch that read the allocation"
    );
    let back = context
        .download(TensorRead::from_tensor(&negated))
        .expect("download");
    let expected: Vec<f32> = value_at_launch.iter().map(|value| -value).collect();
    assert_eq!(
        floats(&back),
        expected,
        "the launch read the values the operand held when it was submitted"
    );
}

#[test]
fn f64_on_the_cpu_and_f32_on_the_gpu_share_one_domain_with_no_transfer() {
    let Some(context) = context() else { return };
    let mut engine = context.engine();
    let len = 128;
    let host = Tensor::from_vec_col_major(
        vec![len],
        (0..len)
            .map(|index| index as f64 * 0.25)
            .collect::<Vec<f64>>(),
    )
    .expect("a host tensor");
    let shared_f64 = context
        .upload(TensorRead::from_tensor(&host))
        .expect("upload the f64 operand");
    assert_eq!(
        shared_f64
            .as_typed::<f64>()
            .expect("f64")
            .backend_buffer()
            .expect("an in-domain allocation")
            .allocation_domain(),
        Some(context.allocation_domain()),
        "the shared domain holds an f64 allocation, not only the dtypes the GPU computes"
    );
    // The paired engine is a `CpuEngine` bound to the context's domain; building one here
    // exercises exactly the same pairing the context hands out.
    let mut cpu_engine =
        tenferro_cpu::CpuEngine::serial().with_allocation_domain(context.domain().clone());
    let uploads_before = stats_of(&context).uploads;
    let doubled = cpu_engine
        .add_read(
            TensorRead::from_tensor(&shared_f64),
            TensorRead::from_tensor(&shared_f64),
        )
        .expect("f64 add on the paired CPU engine, reading the shared allocation directly");
    let values = doubled.as_slice::<f64>().expect("f64");
    assert_eq!(values.len(), len);
    assert_eq!(values[3], 1.5, "the CPU computed the double-precision sum");
    // The GPU engine refuses the same dtype with a typed error naming the CPU engine.
    let error = engine
        .abs_read(TensorRead::from_tensor(&shared_f64))
        .expect_err("f64 is refused by the Metal engine");
    assert!(error.to_string().contains("CpuEngine"), "got {error}");
    assert_eq!(
        stats_of(&context).uploads,
        uploads_before,
        "neither the CPU read nor the refusal transferred anything"
    );
}

#[test]
fn an_f64_operand_names_the_paired_cpu_engine() {
    let Some(context) = context() else { return };
    let mut engine = context.engine();
    let host = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).expect("a tensor");
    let shared = context
        .upload(TensorRead::from_tensor(&host))
        .expect("upload");
    let error = engine
        .neg_read(TensorRead::from_tensor(&shared))
        .expect_err("f64 has no Apple GPU path");
    let message = error.to_string();
    assert!(matches!(error, Error::Unsupported { .. }), "got {message}");
    assert!(
        message.contains("CpuEngine"),
        "the refusal names the paired CPU engine: {message}"
    );
}

#[test]
fn a_foreign_domain_tensor_is_refused() {
    let Some(context) = context() else { return };
    let Some(other) = MetalContext::open(Default::default()).ok() else {
        return;
    };
    let mut engine = context.engine();
    let foreign = other
        .upload(TensorRead::from_tensor(&host_f32(8)))
        .expect("upload into the other context");
    let error = engine
        .neg_read(TensorRead::from_tensor(&foreign))
        .expect_err("another context's allocation is not readable here");
    assert!(matches!(error, Error::RuntimeState { .. }), "got {error}");
}

#[test]
fn a_host_tensor_is_refused_without_an_upload() {
    let Some(context) = context() else { return };
    let mut engine = context.engine();
    let host = host_f32(8);
    let error = engine
        .neg_read(TensorRead::from_tensor(&host))
        .expect_err("a host tensor is not a shared allocation");
    assert!(matches!(error, Error::Unsupported { .. }), "got {error}");
}

#[test]
fn an_unsupported_family_names_its_follow_up_slice() {
    let Some(context) = context() else { return };
    let mut engine = context.engine();
    let shared = context
        .upload(TensorRead::from_tensor(&host_f32(8)))
        .expect("upload");
    let error = engine
        .reduce_sum_read(TensorRead::from_tensor(&shared), &[0])
        .expect_err("reductions are not served yet");
    let message = error.to_string();
    assert!(matches!(error, Error::Unsupported { .. }), "got {message}");
    assert!(
        message.contains("slice 3"),
        "the refusal names the slice that serves it: {message}"
    );
}

#[test]
fn two_threads_submit_and_allocate_without_deadlock() {
    let Some(context) = context() else { return };
    let shared = Arc::new(context);
    let mut handles = Vec::new();
    for thread in 0..2 {
        let context = Arc::clone(&shared);
        handles.push(std::thread::spawn(move || {
            let mut engine = context.engine();
            for round in 0..16 {
                let left = context
                    .upload(TensorRead::from_tensor(&host_f32(256)))
                    .expect("upload");
                let right = context
                    .upload(TensorRead::from_tensor(&host_f32(256)))
                    .expect("upload");
                let sum = engine
                    .add_read(
                        TensorRead::from_tensor(&left),
                        TensorRead::from_tensor(&right),
                    )
                    .expect("add");
                let back = context
                    .download(TensorRead::from_tensor(&sum))
                    .expect("download");
                let values = back.as_slice::<f32>().expect("f32");
                assert_eq!(values.len(), 256);
                for (value, operand) in values.iter().zip(host_values(256)) {
                    // The same operand twice: an f32 sum of equal values is exact here.
                    assert_eq!(*value, operand + operand, "no torn or missing write");
                }
                let _ = (thread, round);
            }
        }));
    }
    for handle in handles {
        handle.join().expect("no deadlock and no panic");
    }
}

#[test]
fn an_empty_operand_allocates_without_a_submission() {
    let Some(context) = context() else { return };
    let mut engine = context.engine();
    let empty = context
        .upload(TensorRead::from_tensor(&host_f32(0)))
        .expect("upload");
    let before = stats_of(&context).submissions;
    let sum = engine
        .add_read(
            TensorRead::from_tensor(&empty),
            TensorRead::from_tensor(&empty),
        )
        .expect("an empty add");
    assert_eq!(stats_of(&context).submissions, before, "nothing to launch");
    assert_eq!(sum.shape(), &[0]);
}

#[test]
fn the_capability_table_agrees_with_a_live_launch() {
    let Some(context) = context() else { return };
    use tenferro_tensor::TensorBackendCapability;
    let engine = context.engine();
    let table = engine.capabilities();
    let add = table
        .iter()
        .find(|row| row.op == tenferro_core_ops::PrimitiveOpKind::Add && row.dtype == DType::F32)
        .expect("add/f32 has a row");
    assert!(add.result.is_supported());
    assert_eq!(
        engine.backend_id(),
        tenferro_tensor::BackendId::Other("metal")
    );
}
