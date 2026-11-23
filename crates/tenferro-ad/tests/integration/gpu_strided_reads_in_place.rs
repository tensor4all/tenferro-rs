#![cfg(feature = "cuda")]

//! Strided and broadcast operands are read where they lie on CUDA (#2701).
//!
//! The flat elementwise entries address `len` consecutive elements, so a transposed view used to
//! be gathered into a compact copy inside every operation, on the direct (`TensorView`) route and
//! the eager route alike, and a compact eager value (which reads as a view of its retained
//! allocation) was copied by every allocating operation too. The layout-taking catalogue entries
//! (ABI 15) read such an operand through its own layout instead.
//!
//! This pins the contract with the runtime's own counters, not timings. For every elementwise
//! operation the GPU benchmark measures on a transposed read, the transposed operand costs exactly
//! what a compact one does: one result allocation and one launch, no staging copy and no upload.
//! The result is bit-identical to the same entry on a compact copy of the transposed values.
//!
//! A reduction over a strided view is still staged on purpose: the reduction planner picks its
//! strategy from the operand's layout, so reading a permuted view in place could change the fold
//! association. Its result is checked bit for bit against the compact operand; a flat view (a
//! compact eager value) is read in place.
//!
//! Run with: `cargo test -p tenferro-ad --features cuda --test integration
//! gpu_strided_reads_in_place -- --ignored --include-ignored`

use num_complex::Complex64;
use tenferro_ad::{EagerRuntime, EagerSession, EagerTensor};
use tenferro_gpu::cuda::{download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine};
use tenferro_runtime::{Tensor, TypedTensor};
use tenferro_tensor::{DType, Engine, TensorAnalytic, TensorRead, TensorView};

const M: usize = 96;
const N: usize = 80;

/// A column-major `rows x cols` matrix with strictly positive entries (log, sqrt, rsqrt, pow and
/// rem are all defined on it).
fn f64_matrix(rows: usize, cols: usize, salt: usize) -> Vec<f64> {
    (0..rows * cols)
        .map(|i| ((i * 7 + salt) % 97) as f64 / 8.0 + 0.5)
        .collect()
}

fn c64_matrix(rows: usize, cols: usize) -> Vec<Complex64> {
    (0..rows * cols)
        .map(|i| Complex64::new((i % 11) as f64 - 5.0, (i % 7) as f64 * 0.5))
        .collect()
}

/// The host transpose of a column-major `rows x cols` matrix, as a `cols x rows` matrix.
fn transposed<T: Copy>(values: &[T], rows: usize, cols: usize) -> Vec<T> {
    (0..rows * cols)
        .map(|i| {
            let (c, r) = (i % cols, i / cols);
            values[c * rows + r]
        })
        .collect()
}

fn tensor<T: tenferro_tensor::TensorScalar>(shape: [usize; 2], values: Vec<T>) -> Tensor {
    Tensor::from_typed::<T>(
        TypedTensor::from_vec_col_major(shape.to_vec(), values).expect("host matrix"),
    )
}

/// The runtime's cumulative device allocations and kernel launches.
fn counters(gpu: &CudaEngine) -> (u64, u64) {
    let stats = gpu.runtime().stats().retirement;
    (stats.allocations, stats.launches)
}

fn host_bits(gpu: &CudaEngine, tensor: &Tensor) -> Vec<u64> {
    let host = download_tensor(gpu.runtime(), tensor).expect("download");
    match host.dtype() {
        DType::C64 => host
            .as_slice::<Complex64>()
            .expect("host c64")
            .iter()
            .flat_map(|value| [value.re.to_bits(), value.im.to_bits()])
            .collect(),
        _ => host
            .as_slice::<f64>()
            .expect("host f64")
            .iter()
            .map(|value| value.to_bits())
            .collect(),
    }
}

/// A `[1, 0]` transpose view of an owned device matrix.
fn transpose_view(input: &Tensor) -> TensorView<'_> {
    match input.dtype() {
        DType::F64 => TensorView::F64(
            input
                .as_typed::<f64>()
                .expect("f64")
                .as_view()
                .transpose_view([1, 0])
                .expect("transpose view"),
        ),
        DType::C64 => TensorView::C64(
            input
                .as_typed::<Complex64>()
                .expect("c64")
                .as_view()
                .transpose_view([1, 0])
                .expect("transpose view"),
        ),
        other => panic!("no transpose view for {other:?}"),
    }
}

type Direct =
    fn(&mut dyn Engine, TensorRead<'_>, TensorRead<'_>) -> tenferro_tensor::Result<Tensor>;
type Eager =
    fn(&mut EagerSession<'_>, &EagerTensor, &EagerTensor) -> tenferro_ad::Result<EagerTensor>;

/// Every operation the GPU benchmark measures on a transposed read; unary ones ignore `rhs`.
fn cases() -> Vec<(&'static str, Direct, Eager)> {
    vec![
        ("abs", |s, a, _| s.abs_read(a), |s, a, _| s.abs(a)),
        ("conj", |s, a, _| s.conj_read(a), |s, a, _| s.conj(a)),
        ("cos", |s, a, _| s.cos_read(a), |s, a, _| s.cos(a)),
        ("erf", |s, a, _| s.erf_read(a), |s, a, _| s.erf(a)),
        ("exp", |s, a, _| s.exp_read(a), |s, a, _| s.exp(a)),
        ("expm1", |s, a, _| s.expm1_read(a), |s, a, _| s.expm1(a)),
        ("log", |s, a, _| s.log_read(a), |s, a, _| s.log(a)),
        ("log1p", |s, a, _| s.log1p_read(a), |s, a, _| s.log1p(a)),
        ("neg", |s, a, _| s.neg_read(a), |s, a, _| s.neg(a)),
        ("rsqrt", |s, a, _| s.rsqrt_read(a), |s, a, _| s.rsqrt(a)),
        ("sign", |s, a, _| s.sign_read(a), |s, a, _| s.sign(a)),
        ("sin", |s, a, _| s.sin_read(a), |s, a, _| s.sin(a)),
        ("sqrt", |s, a, _| s.sqrt_read(a), |s, a, _| s.sqrt(a)),
        ("tanh", |s, a, _| s.tanh_read(a), |s, a, _| s.tanh(a)),
        ("add", |s, a, b| s.add_read(a, b), |s, a, b| s.add(a, b)),
        ("div", |s, a, b| s.div_read(a, b), |s, a, b| s.div(a, b)),
        (
            "maximum",
            |s, a, b| s.maximum_read(a, b),
            |s, a, b| s.maximum(a, b),
        ),
        (
            "minimum",
            |s, a, b| s.minimum_read(a, b),
            |s, a, b| s.minimum(a, b),
        ),
        ("mul", |s, a, b| s.mul_read(a, b), |s, a, b| s.mul(a, b)),
        ("pow", |s, a, b| s.pow_read(a, b), |s, a, b| s.pow(a, b)),
        ("rem", |s, a, b| s.rem_read(a, b), |s, a, b| s.rem(a, b)),
        ("sub", |s, a, b| s.sub_read(a, b), |s, a, b| s.sub(a, b)),
        (
            "reduce_sum",
            |s, a, _| s.reduce_sum_read(a, &[1]),
            |s, a, _| s.reduce_sum(a, Some(&[1])),
        ),
        (
            "reduce_prod",
            |s, a, _| s.reduce_prod_read(a, &[1]),
            |s, a, _| s.reduce_prod(a, Some(&[1])),
        ),
        (
            "reduce_max",
            |s, a, _| s.reduce_max_read(a, &[1]),
            |s, a, _| s.reduce_max(a, Some(&[1])),
        ),
        (
            "reduce_min",
            |s, a, _| s.reduce_min_read(a, &[1]),
            |s, a, _| s.reduce_min(a, Some(&[1])),
        ),
        (
            "reduce_sum_squares",
            |s, a, _| s.reduce_sum_squares_read(a, &[1]),
            |s, a, _| s.reduce_sum_squares(a, Some(&[1])),
        ),
    ]
}

#[test]
#[ignore = "requires a CUDA device"]
fn transposed_reads_cost_what_compact_reads_cost_and_match_bit_for_bit() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut gpu = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("open CUDA engine");
    let ctx = EagerRuntime::new().expect("eager runtime");
    let _no_grad = ctx.no_grad();
    let lhs_values = f64_matrix(M, N, 3);
    let rhs_values = f64_matrix(M, N, 41);
    let complex_values = c64_matrix(M, N);

    for (name, direct, eager) in cases() {
        let complex = name == "conj";
        let up =
            |gpu: &CudaEngine, host: Tensor| upload_tensor(gpu.runtime(), &host).expect("upload");
        // The operands as `M x N` device matrices, read through a `[1, 0]` view as `N x M`, and
        // the compact `N x M` matrices holding the same values.
        let (lhs, rhs, lhs_compact, rhs_compact) = if complex {
            (
                up(&gpu, tensor([M, N], complex_values.clone())),
                up(&gpu, tensor([M, N], complex_values.clone())),
                up(&gpu, tensor([N, M], transposed(&complex_values, M, N))),
                up(&gpu, tensor([N, M], transposed(&complex_values, M, N))),
            )
        } else {
            (
                up(&gpu, tensor([M, N], lhs_values.clone())),
                up(&gpu, tensor([M, N], rhs_values.clone())),
                up(&gpu, tensor([N, M], transposed(&lhs_values, M, N))),
                up(&gpu, tensor([N, M], transposed(&rhs_values, M, N))),
            )
        };
        let (lhs_view, rhs_view) = (transpose_view(&lhs), transpose_view(&rhs));
        // The eager route's own copies of the `M x N` operands.
        let (lhs_eager, rhs_eager) = if complex {
            (
                up(&gpu, tensor([M, N], complex_values.clone())),
                up(&gpu, tensor([M, N], complex_values.clone())),
            )
        } else {
            (
                up(&gpu, tensor([M, N], lhs_values.clone())),
                up(&gpu, tensor([M, N], rhs_values.clone())),
            )
        };

        // Reference: the compact operands. Warm the module set first so preparing a translation
        // unit is not counted against either route.
        let compact = |gpu: &mut CudaEngine| {
            direct(
                gpu,
                TensorRead::from_tensor(&lhs_compact),
                TensorRead::from_tensor(&rhs_compact),
            )
            .unwrap_or_else(|error| panic!("{name} on compact operands: {error}"))
        };
        drop(compact(&mut gpu));
        let before = counters(&gpu);
        let reference = compact(&mut gpu);
        let compact_cost = (counters(&gpu).0 - before.0, counters(&gpu).1 - before.1);
        if !name.starts_with("reduce") {
            assert_eq!(
                compact_cost,
                (1, 1),
                "{name} on compact operands is one result allocation and one launch"
            );
        }

        // Direct route: borrowed transposed views.
        let strided = |gpu: &mut CudaEngine| {
            direct(
                gpu,
                TensorRead::from_view(lhs_view.clone()),
                TensorRead::from_view(rhs_view.clone()),
            )
            .unwrap_or_else(|error| panic!("{name} on transposed views: {error}"))
        };
        drop(strided(&mut gpu));
        let before = counters(&gpu);
        let out = strided(&mut gpu);
        let cost = (counters(&gpu).0 - before.0, counters(&gpu).1 - before.1);
        if !name.starts_with("reduce") {
            assert_eq!(
                cost, compact_cost,
                "{name} on a transposed view must cost what a compact operand does (no staging copy)"
            );
        }
        assert_eq!(
            host_bits(&gpu, &out),
            host_bits(&gpu, &reference),
            "{name} on a transposed view must match the compact operand bit for bit"
        );

        // Eager route: transposed eager values. The eager transpose is a metadata view.
        let (lhs_value, rhs_value) = ctx
            .with_eager_session(&mut gpu, |s| {
                let a = s.constant_from(lhs_eager)?;
                let b = s.constant_from(rhs_eager)?;
                Ok::<_, tenferro_ad::Error>((s.transpose(&a, &[1, 0])?, s.transpose(&b, &[1, 0])?))
            })
            .expect("eager transposed values");
        ctx.with_eager_session(&mut gpu, |s| eager(s, &lhs_value, &rhs_value).map(drop))
            .unwrap_or_else(|error| panic!("eager {name} warm-up: {error}"));
        let before = counters(&gpu);
        let out = ctx
            .with_eager_session(&mut gpu, |s| eager(s, &lhs_value, &rhs_value))
            .unwrap_or_else(|error| panic!("eager {name} on transposed values: {error}"));
        let cost = (counters(&gpu).0 - before.0, counters(&gpu).1 - before.1);
        if !name.starts_with("reduce") {
            assert_eq!(
                cost, compact_cost,
                "eager {name} on a transposed value must cost what a compact operand does"
            );
        }
        let value = ctx
            .with_eager_session(&mut gpu, |s| s.duplicate_value(&out))
            .expect("eager value");
        assert_eq!(
            host_bits(&gpu, &value),
            host_bits(&gpu, &reference),
            "eager {name} on a transposed value must match the compact operand bit for bit"
        );
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn allocating_eager_ops_read_compact_eager_values_in_place() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut gpu = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("open CUDA engine");
    let ctx = EagerRuntime::new().expect("eager runtime");
    let _no_grad = ctx.no_grad();
    let host = tensor([M, N], f64_matrix(M, N, 5));
    let owned = upload_tensor(gpu.runtime(), &host).expect("upload");
    let device = upload_tensor(gpu.runtime(), &host).expect("upload");
    let value = ctx
        .with_eager_session(&mut gpu, |s| s.constant_from(device))
        .expect("eager constant");
    assert!(
        matches!(value.tensor_read(), TensorRead::View(_)),
        "an eager value reads as a descriptor view; that is the path under test"
    );
    let reference = gpu
        .exp_read(TensorRead::from_tensor(&owned))
        .expect("owned exp");
    ctx.with_eager_session(&mut gpu, |s| s.exp(&value).map(drop))
        .expect("warm-up");
    let before = counters(&gpu);
    let out = ctx
        .with_eager_session(&mut gpu, |s| s.exp(&value))
        .expect("eager exp");
    let after = counters(&gpu);
    assert_eq!(
        (after.0 - before.0, after.1 - before.1),
        (1, 1),
        "eager exp on a compact eager value is one result allocation and one launch, no copy"
    );
    let out = ctx
        .with_eager_session(&mut gpu, |s| s.duplicate_value(&out))
        .expect("eager value");
    assert_eq!(host_bits(&gpu, &out), host_bits(&gpu, &reference));
}

/// The review counterexamples of the in-place routes (#2721): each keeps the staged route's
/// observable result.
#[test]
#[ignore = "requires a CUDA device"]
fn in_place_routes_keep_the_staged_results_on_edge_cases() {
    use tenferro_tensor::{
        TensorElementwise as _, TensorReduction as _, TensorViewMut, TensorWrite, TypedTensor as T,
    };
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut gpu = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("open CUDA engine");
    let rt = gpu.runtime().clone();
    let up = |host: Tensor| upload_tensor(&rt, &host).expect("upload");
    let down_f64 = |t: &Tensor| {
        download_tensor(&rt, t)
            .expect("download")
            .as_slice::<f64>()
            .expect("f64")
            .to_vec()
    };

    // A product over both axes of a transposed view folds in the staged route's order:
    // [1e200, 1e-200, 1e200, 1e-200] folds to 1, not to NaN (Inf * 0).
    let m = up(tensor([2, 2], vec![1e200, 1e200, 1e-200, 1e-200]));
    let prod = gpu
        .reduce_prod_read(TensorRead::from_view(transpose_view(&m)), &[0, 1])
        .expect("prod over both axes");
    assert_eq!(down_f64(&prod), vec![1.0]);

    // A checked integer division into a strided destination that fails leaves the destination
    // untouched.
    let lhs = up(Tensor::from_typed::<i32>(
        T::from_vec_col_major(vec![2], vec![8, 9]).expect("lhs"),
    ));
    let rhs = up(Tensor::from_typed::<i32>(
        T::from_vec_col_major(vec![2], vec![2, 0]).expect("rhs"),
    ));
    let mut dest = up(Tensor::from_typed::<i32>(
        T::from_vec_col_major(vec![4], vec![77, 77, 77, 77]).expect("dest"),
    ));
    {
        let typed = dest.as_typed_mut::<i32>().expect("i32");
        let mut whole = typed.as_view_mut();
        let view = whole
            .slice_axis_view(0, tenferro_tensor::StridedSliceSpec::new(0, Some(4), 2))
            .expect("stride-2 view");
        let result = gpu.div_read_into(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            TensorWrite::View(TensorViewMut::I32(view)),
        );
        assert!(result.is_err(), "a zero divisor is reported");
    }
    let host = download_tensor(&rt, &dest).expect("download");
    assert_eq!(host.as_slice::<i32>().expect("i32"), &[77, 77, 77, 77]);

    // A product over one axis of a transposed view keeps the compact fold: with 2^600 and 2^-600
    // placed where a different lane split would pair them, the result stays 1.
    let mut values = vec![1.0_f64; 2 * 512];
    // Column-major [512, 2] host data whose [1, 0] view is the [2, 512] operand: logical row 0
    // holds 2^600 at columns 0 and 32 and 2^-600 at columns 1 and 33.
    for (column, value) in [
        (0, 2f64.powi(600)),
        (32, 2f64.powi(600)),
        (1, 2f64.powi(-600)),
        (33, 2f64.powi(-600)),
    ] {
        values[column] = value;
    }
    let wide = up(tensor([512, 2], values));
    let prod = gpu
        .reduce_prod_read(TensorRead::from_view(transpose_view(&wide)), &[1])
        .expect("prod over axis 1");
    assert_eq!(down_f64(&prod), vec![1.0, 1.0]);

    // A rank-9 compact view reduces through the general path, without a panic.
    let rank9 = up(Tensor::from_typed::<f64>(
        T::from_vec_col_major(vec![2, 1, 1, 1, 1, 1, 1, 1, 2], vec![1.0, 2.0, 3.0, 4.0])
            .expect("rank 9"),
    ));
    let rank9_view = TensorView::F64(rank9.as_typed::<f64>().expect("f64").as_view());
    let sum = gpu
        .reduce_sum_read(TensorRead::from_view(rank9_view), &[0])
        .expect("rank-9 view reduction");
    assert_eq!(down_f64(&sum), vec![3.0, 7.0]);

    // A zero-element strided view is an empty result, not a planner refusal.
    let empty = up(tensor([2, 0], Vec::<f64>::new()));
    let out = gpu
        .neg_read(TensorRead::from_view(transpose_view(&empty)))
        .expect("empty view");
    assert_eq!(out.shape(), &[0, 2]);

    // A rank-9 power whose rank-0 exponent is an offset view keeps the flat route.
    let base = up(Tensor::from_typed::<f64>(
        T::from_vec_col_major(vec![2, 1, 1, 1, 1, 1, 1, 1, 1], vec![2.0, 3.0]).expect("base"),
    ));
    let exponents = up(tensor([2, 1], vec![99.0, 2.0]));
    let exponent = TensorView::F64(
        exponents
            .as_typed::<f64>()
            .expect("f64")
            .as_view()
            .slice_axis_view(0, tenferro_tensor::StridedSliceSpec::new(1, Some(2), 1))
            .expect("offset slice")
            .reshape_view(&[])
            .expect("rank-0 view"),
    );
    let out = gpu
        .pow_read(
            TensorRead::from_tensor(&base),
            TensorRead::from_view(exponent),
        )
        .expect("rank-9 power");
    assert_eq!(down_f64(&out), vec![4.0, 9.0]);

    // An empty base with the same offset scalar exponent is an empty result.
    let empty_base = up(tensor([0, 2], Vec::<f64>::new()));
    let exponent = TensorView::F64(
        exponents
            .as_typed::<f64>()
            .expect("f64")
            .as_view()
            .slice_axis_view(0, tenferro_tensor::StridedSliceSpec::new(1, Some(2), 1))
            .expect("offset slice")
            .reshape_view(&[])
            .expect("rank-0 view"),
    );
    let out = gpu
        .pow_read(
            TensorRead::from_tensor(&empty_base),
            TensorRead::from_view(exponent),
        )
        .expect("empty power");
    assert_eq!(out.shape(), &[0, 2]);
}
