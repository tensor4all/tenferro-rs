#![cfg(feature = "cuda")]

//! Reductions over a zero-length axis on the CUDA engine (#2453).
//!
//! `sum` over nothing is `0` and `prod` over nothing is `1`, on both surfaces and for every
//! dtype. Each case here runs the same input through the CPU engine and the CUDA engine and
//! requires the same values, so a backend that invented its own answer (or kept the old
//! validation error) fails.

use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_gpu::cuda::{download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine};
use tenferro_tensor::{DType, Tensor, TensorRead, TensorReduction};

fn engine() -> Option<CudaEngine> {
    if !gpu_available() {
        return None;
    }
    Some(CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine"))
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Sum,
    Prod,
    Max,
    Min,
}

fn cpu_reduce(op: Op, source: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor> {
    let mut engine = CpuEngine::serial();
    let read = TensorRead::from_tensor(source);
    match op {
        Op::Sum => engine.reduce_sum_read(read, axes),
        Op::Prod => engine.reduce_prod_read(read, axes),
        Op::Max => engine.reduce_max_read(read, axes),
        Op::Min => engine.reduce_min_read(read, axes),
    }
}

fn gpu_reduce(
    engine: &mut CudaEngine,
    op: Op,
    source: &Tensor,
    axes: &[usize],
) -> tenferro_tensor::Result<Tensor> {
    let read = TensorRead::from_tensor(source);
    match op {
        Op::Sum => engine.reduce_sum_read(read, axes),
        Op::Prod => engine.reduce_prod_read(read, axes),
        Op::Max => engine.reduce_max_read(read, axes),
        Op::Min => engine.reduce_min_read(read, axes),
    }
}

/// Upload `host`, reduce it on the device, and hand the result back on the host.
fn device_reduce(
    engine: &mut CudaEngine,
    op: Op,
    host: &Tensor,
    axes: &[usize],
) -> tenferro_tensor::Result<Tensor> {
    let device = upload_tensor(engine.runtime(), host)?;
    let reduced = gpu_reduce(engine, op, &device, axes)?;
    download_tensor(engine.runtime(), &reduced)
}

/// Run `op` over `axes` on both engines and require one shape and one value sequence.
fn assert_cpu_matches_gpu(
    engine: &mut CudaEngine,
    op: Op,
    source: &Tensor,
    axes: &[usize],
    expected: &[f64],
) {
    let cpu = cpu_reduce(op, source, axes).expect("cpu reduce");
    let host = device_reduce(engine, op, source, axes).expect("cuda reduce");

    assert_eq!(host.shape(), cpu.shape(), "{op:?} over {axes:?}: shape");
    assert_eq!(host.dtype(), cpu.dtype(), "{op:?} over {axes:?}: dtype");
    assert_eq!(
        host.as_slice::<f64>().expect("f64 result"),
        expected,
        "{op:?} over {axes:?} did not report the empty fold's unit"
    );
    assert_eq!(
        host.as_slice::<f64>().expect("f64 result"),
        cpu.as_slice::<f64>().expect("f64 result"),
        "{op:?} over {axes:?}: CPU and CUDA disagree"
    );
}

#[test]
#[ignore = "requires CUDA"]
fn empty_fold_identity_matches_cpu_for_every_axis_choice() {
    let Some(mut engine) = engine() else { return };
    // [2, 0, 3]: one empty axis between two kept ones, so the unit is written per output element
    // rather than once.
    let source =
        Tensor::from_vec_col_major(vec![2, 0, 3], Vec::<f64>::new()).expect("zero-extent tensor");

    for (op, unit) in [(Op::Sum, 0.0), (Op::Prod, 1.0)] {
        assert_cpu_matches_gpu(&mut engine, op, &source, &[1], &[unit; 6]);
        // Reducing the empty axis together with a kept one still folds over an empty domain.
        assert_cpu_matches_gpu(&mut engine, op, &source, &[0, 1], &[unit; 3]);
        // The empty axis may also be the only kept one: nothing to write either way.
        assert_cpu_matches_gpu(&mut engine, op, &source, &[0, 2], &[]);
    }

    // A rank-1 empty tensor folds to the scalar unit.
    let scalar_source =
        Tensor::from_vec_col_major(vec![0], Vec::<f64>::new()).expect("zero-extent tensor");
    assert_cpu_matches_gpu(&mut engine, Op::Sum, &scalar_source, &[0], &[0.0]);
    assert_cpu_matches_gpu(&mut engine, Op::Prod, &scalar_source, &[0], &[1.0]);
}

#[test]
#[ignore = "requires CUDA"]
fn empty_fold_identity_covers_f32_i32_i64_and_complex() {
    let Some(mut engine) = engine() else { return };

    let f32_source = Tensor::from_vec_col_major(vec![0, 2], Vec::<f32>::new()).expect("f32");
    for (op, unit) in [(Op::Sum, 0.0_f32), (Op::Prod, 1.0)] {
        let host = device_reduce(&mut engine, op, &f32_source, &[0]).expect("cuda reduce");
        assert_eq!(host.dtype(), DType::F32);
        assert_eq!(host.as_slice::<f32>().expect("f32"), &[unit, unit]);
        assert_eq!(
            cpu_reduce(op, &f32_source, &[0])
                .expect("cpu")
                .as_slice::<f32>()
                .expect("f32"),
            &[unit, unit]
        );
    }

    let i32_source = Tensor::from_vec_col_major(vec![0, 2], Vec::<i32>::new()).expect("i32");
    for (op, unit) in [(Op::Sum, 0_i32), (Op::Prod, 1)] {
        let host = device_reduce(&mut engine, op, &i32_source, &[0]).expect("cuda reduce");
        assert_eq!(host.as_slice::<i32>().expect("i32"), &[unit, unit]);
        assert_eq!(
            cpu_reduce(op, &i32_source, &[0])
                .expect("cpu")
                .as_slice::<i32>()
                .expect("i32"),
            &[unit, unit]
        );
    }

    let i64_source = Tensor::from_vec_col_major(vec![0, 2], Vec::<i64>::new()).expect("i64");
    for (op, unit) in [(Op::Sum, 0_i64), (Op::Prod, 1)] {
        let host = device_reduce(&mut engine, op, &i64_source, &[0]).expect("cuda reduce");
        assert_eq!(host.as_slice::<i64>().expect("i64"), &[unit, unit]);
        assert_eq!(
            cpu_reduce(op, &i64_source, &[0])
                .expect("cpu")
                .as_slice::<i64>()
                .expect("i64"),
            &[unit, unit]
        );
    }

    let c64_source = Tensor::from_vec_col_major(vec![0, 2], Vec::<Complex64>::new()).expect("c64");
    for (op, unit) in [
        (Op::Sum, Complex64::new(0.0, 0.0)),
        (Op::Prod, Complex64::new(1.0, 0.0)),
    ] {
        let host = device_reduce(&mut engine, op, &c64_source, &[0]).expect("cuda reduce");
        assert_eq!(host.as_slice::<Complex64>().expect("c64"), &[unit, unit]);
        assert_eq!(
            cpu_reduce(op, &c64_source, &[0])
                .expect("cpu")
                .as_slice::<Complex64>()
                .expect("c64"),
            &[unit, unit]
        );
    }
}

#[test]
#[ignore = "requires CUDA"]
fn extrema_still_refuse_a_zero_length_axis_on_both_engines() {
    let Some(mut engine) = engine() else { return };
    let source =
        Tensor::from_vec_col_major(vec![0, 3], Vec::<f64>::new()).expect("zero-extent tensor");

    for op in [Op::Max, Op::Min] {
        assert!(
            cpu_reduce(op, &source, &[0]).is_err(),
            "{op:?} must not invent a value for an empty domain on CPU"
        );
        assert!(
            device_reduce(&mut engine, op, &source, &[0]).is_err(),
            "{op:?} must not invent a value for an empty domain on CUDA"
        );
    }

    // A sum over the same operand is still a value, which pins that the refusal above is about
    // the extrema and not about the zero extent itself.
    for op in [Op::Sum, Op::Prod] {
        device_reduce(&mut engine, op, &source, &[0]).expect("identity fold");
    }
}
