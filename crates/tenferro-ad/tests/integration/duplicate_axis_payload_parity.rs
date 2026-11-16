//! #2601: the concrete (`CoreOps`), eager (`EagerSession`) and traced
//! (`TracedTensor`) surfaces report the same validation payload for the same
//! invalid reduction / `reverse` / `transpose` axes.
//!
//! The three layers have distinct error enums, so each extractor below reduces
//! its surface's error to the shared [`ValidationError`] and panics on anything
//! else.

use std::sync::Arc;

use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuEngine;
use tenferro_runtime::{CoreOps, TraceContext};
use tenferro_tensor::{Tensor, TensorRead, TensorStructural, ValidationError};

use crate::support::{cpu_runtime, run_many_traced_with};

fn concrete_payload(err: tenferro_tensor::Error) -> ValidationError {
    match err {
        tenferro_tensor::Error::Validation { source, .. } => source,
        other => panic!("unexpected concrete error: {other:?}"),
    }
}

fn eager_payload(err: tenferro_ad::Error) -> ValidationError {
    match err {
        tenferro_ad::Error::TensorRuntime(tenferro_tensor::Error::Validation {
            source, ..
        }) => source,
        other => panic!("unexpected eager error: {other:?}"),
    }
}

fn traced_payload(err: tenferro_runtime::Error) -> ValidationError {
    match err {
        tenferro_runtime::Error::Validation { source, .. } => source,
        other => panic!("unexpected traced error: {other:?}"),
    }
}

#[derive(Clone, Copy, Debug)]
enum Reduction {
    Sum,
    Prod,
    Max,
    Min,
    SumSquares,
    Mean,
}

const REDUCTIONS: [Reduction; 6] = [
    Reduction::Sum,
    Reduction::Prod,
    Reduction::Max,
    Reduction::Min,
    Reduction::SumSquares,
    Reduction::Mean,
];

fn concrete_reduce(
    engine: &mut CpuEngine,
    x: &Tensor,
    kind: Reduction,
    axes: &[isize],
) -> tenferro_tensor::Result<Tensor> {
    match kind {
        Reduction::Sum => x.reduce_sum(Some(axes), engine),
        Reduction::Prod => x.reduce_prod(Some(axes), engine),
        Reduction::Max => x.reduce_max(Some(axes), engine),
        Reduction::Min => x.reduce_min(Some(axes), engine),
        Reduction::SumSquares => x.reduce_sum_squares(Some(axes), engine),
        Reduction::Mean => x.reduce_mean(Some(axes), engine),
    }
}

fn eager_reduce(
    ctx: &Arc<EagerRuntime>,
    x: &EagerTensor,
    kind: Reduction,
    axes: &[isize],
) -> tenferro_ad::Result<EagerTensor> {
    ctx.with_eager_session(&mut CpuEngine::serial(), |session| match kind {
        Reduction::Sum => session.reduce_sum(x, Some(axes)),
        Reduction::Prod => session.reduce_prod(x, Some(axes)),
        Reduction::Max => session.reduce_max(x, Some(axes)),
        Reduction::Min => session.reduce_min(x, Some(axes)),
        Reduction::SumSquares => session.reduce_sum_squares(x, Some(axes)),
        Reduction::Mean => session.reduce_mean(x, Some(axes)),
    })
}

fn i32_tensor(shape: &[usize]) -> Tensor {
    let size = shape.iter().product();
    Tensor::from_vec_col_major(shape.to_vec(), vec![1_i32; size]).unwrap()
}

fn f64_tensor(shape: &[usize]) -> Tensor {
    let size = shape.iter().product();
    Tensor::from_vec_col_major(shape.to_vec(), vec![1.0_f64; size]).unwrap()
}

#[derive(Clone, Copy, Debug)]
enum Family {
    Reduction(Reduction),
    Reverse,
    Transpose,
}

fn payloads(
    shape: &[usize],
    invalid: &[isize],
    family: Family,
) -> (ValidationError, [ValidationError; 2]) {
    let x = f64_tensor(shape);
    let mut engine = CpuEngine::serial();
    let trace = TraceContext::new();
    let traced = trace
        .from_vec_col_major(shape.to_vec(), vec![1.0_f64; shape.iter().product()])
        .unwrap();
    let ctx = Arc::new(EagerRuntime::new().unwrap());

    match family {
        Family::Reduction(kind) => {
            let concrete =
                concrete_payload(concrete_reduce(&mut engine, &x, kind, invalid).unwrap_err());
            let eager_x = ctx.constant_from(f64_tensor(shape)).unwrap();
            let eager = eager_payload(eager_reduce(&ctx, &eager_x, kind, invalid).unwrap_err());
            let traced = traced_payload(
                match kind {
                    Reduction::Sum => traced.reduce_sum(Some(invalid)),
                    Reduction::Prod => traced.reduce_prod(Some(invalid)),
                    Reduction::Max => traced.reduce_max(Some(invalid)),
                    Reduction::Min => traced.reduce_min(Some(invalid)),
                    Reduction::SumSquares => traced.reduce_sum_squares(Some(invalid)),
                    Reduction::Mean => traced.reduce_mean(Some(invalid)),
                }
                .unwrap_err(),
            );
            (concrete, [eager, traced])
        }
        Family::Reverse => {
            let concrete = concrete_payload(x.reverse(invalid, &mut engine).unwrap_err());
            let eager_x = ctx.constant_from(f64_tensor(shape)).unwrap();
            let eager = eager_payload(
                ctx.with_eager_session(&mut CpuEngine::serial(), |session| {
                    session.reverse(&eager_x, invalid)
                })
                .unwrap_err(),
            );
            let traced = traced_payload(traced.reverse(invalid).unwrap_err());
            (concrete, [eager, traced])
        }
        Family::Transpose => {
            let concrete = concrete_payload(x.transpose(invalid, &mut engine).unwrap_err());
            let eager_x = ctx.constant_from(f64_tensor(shape)).unwrap();
            let eager = eager_payload(
                ctx.with_eager_session(&mut CpuEngine::serial(), |session| {
                    session.transpose(&eager_x, invalid)
                })
                .unwrap_err(),
            );
            let traced = traced_payload(traced.transpose(invalid).unwrap_err());
            (concrete, [eager, traced])
        }
    }
}

fn assert_parity(family: Family, shape: &[usize], invalid: &[isize], expected: ValidationError) {
    let (concrete, [eager, traced]) = payloads(shape, invalid, family);
    assert_eq!(
        concrete, eager,
        "concrete vs eager for {family:?} {invalid:?}"
    );
    assert_eq!(eager, traced, "eager vs traced for {family:?} {invalid:?}");
    assert_eq!(eager, expected, "payload for {family:?} {invalid:?}");
}

fn duplicate(axis: usize, role: &'static str) -> ValidationError {
    ValidationError::DuplicateAxis { axis, role }
}

fn out_of_bounds(axis: isize, rank: usize) -> ValidationError {
    ValidationError::AxisOutOfBounds { axis, rank }
}

fn bad_perm_length(expected: usize, actual: usize) -> ValidationError {
    ValidationError::InvalidPermutationLength { expected, actual }
}

#[test]
fn reduction_duplicate_axis_payload_matches_across_surfaces() {
    // (shape, invalid axes, expected payload)
    let cases: [(Vec<usize>, Vec<isize>, ValidationError); 5] = [
        (vec![2, 3], vec![0, 0], duplicate(0, "axes")),
        // `-2` at rank 3 normalizes to `1`, so the second entry repeats axis 1.
        (vec![2, 3, 4], vec![1, -2], duplicate(1, "axes")),
        (vec![2, 3], vec![2], out_of_bounds(2, 2)),
        // Every entry is range checked before any repeat is reported.
        (vec![2, 3], vec![0, 0, 2], out_of_bounds(2, 2)),
        (vec![2, 3], vec![0, 0, -3], out_of_bounds(-3, 2)),
    ];
    for kind in REDUCTIONS {
        for (shape, invalid, expected) in &cases {
            assert_parity(Family::Reduction(kind), shape, invalid, expected.clone());
        }
    }
}

#[test]
fn reverse_duplicate_axis_payload_matches_across_surfaces() {
    let cases: [(Vec<usize>, Vec<isize>, ValidationError); 5] = [
        (vec![2, 3], vec![0, 0], duplicate(0, "axes")),
        (vec![2, 3, 4], vec![1, -2], duplicate(1, "axes")),
        (vec![2, 3], vec![-3], out_of_bounds(-3, 2)),
        (vec![2, 3], vec![0, 0, 2], out_of_bounds(2, 2)),
        (vec![2, 3], vec![0, 0, -3], out_of_bounds(-3, 2)),
    ];
    for (shape, invalid, expected) in &cases {
        assert_parity(Family::Reverse, shape, invalid, expected.clone());
    }
}

#[test]
fn transpose_duplicate_axis_payload_matches_across_surfaces() {
    let cases: [(Vec<usize>, Vec<isize>, ValidationError); 6] = [
        (vec![2, 3], vec![0, 0], duplicate(0, "permutation")),
        // `-1` at rank 2 normalizes to the same axis as the first entry.
        (vec![2, 3], vec![1, -1], duplicate(1, "permutation")),
        (vec![2, 3], vec![2, 0], out_of_bounds(2, 2)),
        (vec![2, 3], vec![0], bad_perm_length(2, 1)),
        // Range and length are checked before the duplicate, as on the eager
        // and traced surfaces.
        (vec![2, 3, 4], vec![0, 0, 5], out_of_bounds(5, 3)),
        (vec![2, 3, 4], vec![1, -2], bad_perm_length(3, 2)),
    ];
    for (shape, invalid, expected) in &cases {
        assert_parity(Family::Transpose, shape, invalid, expected.clone());
    }
}

/// An unsupported dtype no longer hides an invalid axis list on the composite
/// reduction: the axis payload wins on all three surfaces, as it does on the
/// primitive reductions.
#[test]
fn reduce_mean_reports_invalid_axes_before_the_dtype() {
    let shape = [2_usize, 3];
    let axes = [0_isize, 0];
    let mut engine = CpuEngine::serial();
    let trace = TraceContext::new();
    let traced = trace
        .from_vec_col_major(shape.to_vec(), vec![1_i32; 6])
        .unwrap();
    let ctx = Arc::new(EagerRuntime::new().unwrap());
    let eager_x = ctx.constant_from(i32_tensor(&shape)).unwrap();

    let concrete = concrete_payload(
        i32_tensor(&shape)
            .reduce_mean(Some(&axes), &mut engine)
            .unwrap_err(),
    );
    let eager = eager_payload(
        ctx.with_eager_session(&mut CpuEngine::serial(), |session| {
            session.reduce_mean(&eager_x, Some(&axes))
        })
        .unwrap_err(),
    );
    let traced = traced_payload(traced.reduce_mean(Some(&axes)).unwrap_err());

    assert_eq!(concrete, duplicate(0, "axes"));
    assert_eq!(concrete, eager);
    assert_eq!(eager, traced);
}

#[test]
fn valid_reduction_reverse_transpose_values_match_across_surfaces() {
    let shape = vec![2, 3];
    let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = Tensor::from_vec_col_major(shape.clone(), data.clone()).unwrap();
    let mut engine = CpuEngine::serial();
    let trace = TraceContext::new();
    let traced = trace
        .from_vec_col_major(shape.clone(), data.clone())
        .unwrap();
    let ctx = Arc::new(EagerRuntime::new().unwrap());
    let eager_x = ctx
        .constant_from(Tensor::from_vec_col_major(shape.clone(), data).unwrap())
        .unwrap();
    let runtime = cpu_runtime();

    let values = |concrete: Tensor, eager: EagerTensor, traced: &tenferro_runtime::TracedTensor| {
        let traced = run_many_traced_with(&runtime, &[traced]).unwrap().remove(0);
        let value = eager.value().unwrap();
        let eager = CpuEngine::serial()
            .to_contiguous_read(TensorRead::View(value.as_tensor_view().clone()))
            .unwrap();
        assert_eq!(concrete.shape(), traced.shape());
        assert_eq!(concrete.shape(), eager.shape());
        assert_eq!(
            concrete.as_slice::<f64>().unwrap(),
            traced.as_slice::<f64>().unwrap()
        );
        assert_eq!(
            concrete.as_slice::<f64>().unwrap(),
            eager.as_slice::<f64>().unwrap()
        );
    };

    values(
        x.reduce_sum(Some(&[0]), &mut engine).unwrap(),
        ctx.with_eager_session(&mut CpuEngine::serial(), |session| {
            session.reduce_sum(&eager_x, Some(&[0]))
        })
        .unwrap(),
        &traced.reduce_sum(Some(&[0])).unwrap(),
    );
    values(
        x.reverse(&[0], &mut engine).unwrap(),
        ctx.with_eager_session(&mut CpuEngine::serial(), |session| {
            session.reverse(&eager_x, &[0])
        })
        .unwrap(),
        &traced.reverse(&[0]).unwrap(),
    );
    values(
        x.transpose(&[1, 0], &mut engine).unwrap(),
        ctx.with_eager_session(&mut CpuEngine::serial(), |session| {
            session.transpose(&eager_x, &[1, 0])
        })
        .unwrap(),
        &traced.transpose(&[1, 0]).unwrap(),
    );
}
