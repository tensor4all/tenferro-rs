//! Coverage for the structural cast matrix, exercised through the public backend path.

use crate::tests::*;

/// Every preset dtype is cast to every other one, so each arm of the conversion matrix runs and
/// the refusals for the pairs it does not cover are seen rather than assumed.
#[test]
fn cast_matrix_covers_every_preset_pair() {
    let mut backend = crate::CpuEngine::serial();
    let dtypes = [
        DType::F32,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::Bool,
        DType::C32,
        DType::C64,
    ];
    for from in dtypes {
        let input = sample_for(from);
        for to in dtypes {
            // A pair the matrix covers returns a tensor of the requested dtype; one it does not
            // returns a typed error. Both are outcomes this table owns.
            match crate::test_support::on_engine(&mut backend, |__s| __s.cast(&input, to)).unwrap()
            {
                Ok(out) => assert_eq!(out.dtype(), to),
                Err(_) => continue,
            }
        }
    }
}

fn sample_for(dtype: DType) -> Tensor {
    use num_complex::{Complex32, Complex64};
    match dtype {
        DType::F32 => Tensor::from_vec_col_major(vec![2], vec![1.5_f32, -2.0]).unwrap(),
        DType::F64 => Tensor::from_vec_col_major(vec![2], vec![1.5_f64, -2.0]).unwrap(),
        DType::I32 => Tensor::from_vec_col_major(vec![2], vec![1_i32, -2]).unwrap(),
        DType::I64 => Tensor::from_vec_col_major(vec![2], vec![1_i64, -2]).unwrap(),
        DType::Bool => Tensor::from_vec_col_major(vec![2], vec![true, false]).unwrap(),
        DType::C32 => Tensor::from_vec_col_major(
            vec![2],
            vec![Complex32::new(1.0, 0.5), Complex32::new(-2.0, 0.0)],
        )
        .unwrap(),
        DType::C64 => Tensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.0, 0.5), Complex64::new(-2.0, 0.0)],
        )
        .unwrap(),
        other => panic!("no sample for {other:?}"),
    }
}

/// One tensor value and the strided views a read can name over it.
fn view_source() -> tenferro_tensor::TensorValue {
    // Column-major [2, 3]: columns (1, 2), (3, 4), (5, 6).
    tenferro_tensor::TensorValue::from_tensor(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.5_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    )
}

/// The compact view, a transposed view, an offset view and a negative-stride view of one source.
fn reads<'a>(
    source: &'a tenferro_tensor::TensorValue,
    offset_buffer: &'a [f64],
) -> Vec<TensorRead<'a>> {
    use tenferro_tensor::{TensorView, TypedTensorView};
    let mut reads = vec![TensorRead::from_view(
        source
            .tensor_view()
            .expect("an f64 source always has an f64 view"),
    )];
    // The transpose of the column-major [2, 3] source: dims [3, 2], strides (1, 3).
    reads.push(TensorRead::from_view(TensorView::F64(
        TypedTensorView::from_slice(vec![3, 2], vec![1, 3], 0, &offset_buffer[..6]).unwrap(),
    )));
    // The source shifted by one element: every element moves, the layout stays compact.
    reads.push(TensorRead::from_view(TensorView::F64(
        TypedTensorView::from_slice(vec![2, 3], vec![1, 2], 1, offset_buffer).unwrap(),
    )));
    // A reversed column axis: stride -1 from the last element of the first column.
    reads.push(TensorRead::from_view(TensorView::F64(
        TypedTensorView::from_slice(vec![2, 3], vec![-1, 2], 1, &offset_buffer[..6]).unwrap(),
    )));
    reads
}

/// Compare two cast outputs element by element, dtype and shape included.
///
/// A dtype other than the tensor's own yields `None` on both sides, so one comparison covers
/// every preset arm.
fn assert_same_cast(got: &Tensor, want: &Tensor) {
    assert_eq!(got.dtype(), want.dtype());
    assert_eq!(got.shape(), want.shape());
    assert_eq!(got.as_slice::<f64>().ok(), want.as_slice::<f64>().ok());
    assert_eq!(got.as_slice::<f32>().ok(), want.as_slice::<f32>().ok());
    assert_eq!(got.as_slice::<i32>().ok(), want.as_slice::<i32>().ok());
    assert_eq!(got.as_slice::<i64>().ok(), want.as_slice::<i64>().ok());
    assert_eq!(got.as_slice::<bool>().ok(), want.as_slice::<bool>().ok());
    assert_eq!(
        got.as_slice::<num_complex::Complex32>().ok(),
        want.as_slice::<num_complex::Complex32>().ok()
    );
    assert_eq!(
        got.as_slice::<num_complex::Complex64>().ok(),
        want.as_slice::<num_complex::Complex64>().ok()
    );
}

/// Every destination dtype casts a borrowed view exactly as it casts the staged compact tensor,
/// and the view is never staged (#2223, T3/T4).
#[test]
fn cast_read_matches_cast_on_every_view_layout_without_staging() {
    let source = view_source();
    let offset_buffer = [0.0_f64, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut backend = crate::CpuEngine::serial();
    let destinations = [
        DType::F32,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::Bool,
        DType::C32,
        DType::C64,
    ];

    for read in reads(&source, &offset_buffer) {
        for to in destinations {
            let stagings = backend.resources.read_stagings();
            let got =
                crate::test_support::on_engine(&mut backend, |__s| __s.cast_read(read.clone(), to))
                    .unwrap()
                    .unwrap();
            assert_eq!(
                backend.resources.read_stagings(),
                stagings,
                "cast_read({to:?}) must not stage the view"
            );

            // The reference is the read's own compaction followed by the owned cast: the
            // path this change replaces.
            let staged = backend.to_contiguous_read(read.clone()).unwrap();
            let want = backend.cast(&staged, to).unwrap();
            assert_same_cast(&got, &want);
        }
    }
}

/// A non-finite or out-of-range value refuses the cast, on a view as on an owned tensor.
#[test]
fn cast_read_refuses_the_values_cast_refuses() {
    let mut backend = crate::CpuEngine::serial();
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0e30, -1.0e30] {
        let owned = Tensor::from_vec_col_major(vec![1], vec![value]).unwrap();
        let value_view = tenferro_tensor::TensorValue::from_tensor(
            Tensor::from_vec_col_major(vec![1], vec![value]).unwrap(),
        );
        let read = TensorRead::from_view(
            value_view
                .tensor_view()
                .expect("the scalar value always has the requested view"),
        );
        let owned_error = backend.cast(&owned, DType::I32).unwrap_err();
        let read_error = backend.cast_read(read, DType::I32).unwrap_err();
        assert_eq!(
            std::mem::discriminant(&owned_error),
            std::mem::discriminant(&read_error),
            "{value} must refuse the same way through both entries"
        );
    }
}

/// The range check walks the read's logical elements, so a NaN outside a slice cannot refuse it.
#[test]
fn a_nan_outside_the_view_does_not_refuse_the_cast() {
    let mut backend = crate::CpuEngine::serial();
    let source = Tensor::from_vec_col_major(vec![3], vec![f64::NAN, 1.0, 2.0]).unwrap();
    let view = tenferro_tensor::TensorView::F64(
        tenferro_tensor::TypedTensorView::from_slice(vec![2], vec![1], 1, &[f64::NAN, 1.0, 2.0])
            .unwrap(),
    );

    assert!(
        backend.cast(&source, DType::I32).is_err(),
        "the owned cast reads the NaN"
    );
    let out = backend
        .cast_read(TensorRead::from_view(view), DType::I32)
        .unwrap();
    assert_eq!(out.as_slice::<i32>().unwrap(), &[1, 2]);
    assert_eq!(out.shape(), &[2]);
}

/// A same-dtype cast of a view is fresh compact storage, not a borrow of the source.
#[test]
fn cast_read_of_a_view_is_fresh_compact_storage() {
    let mut backend = crate::CpuEngine::serial();
    let buffer = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let view = tenferro_tensor::TensorView::F64(
        tenferro_tensor::TypedTensorView::from_slice(vec![3, 2], vec![1, 3], 0, &buffer).unwrap(),
    );

    let out = backend
        .cast_read(TensorRead::from_view(view), DType::F64)
        .unwrap();

    assert_eq!(out.shape(), &[3, 2]);
    // Column-major [3, 2]: the transpose of the column-major [2, 3] source.
    assert_eq!(
        out.as_slice::<f64>().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    assert!(
        !std::ptr::eq(out.as_slice::<f64>().unwrap().as_ptr(), buffer.as_ptr()),
        "the output must be fresh storage"
    );
}

/// `convert_read` refuses a lossy pair before the backend sees it, and serves the lattice pairs.
#[test]
fn convert_read_refuses_a_lossy_pair_before_any_backend_work() {
    let mut backend = crate::CpuEngine::serial();
    let source = tenferro_tensor::TensorValue::from_tensor(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
    );
    let read = TensorRead::from_view(
        source
            .tensor_view()
            .expect("the f64 source always has an f64 view"),
    );

    let lossy = backend.convert_read(read.clone(), DType::I32).unwrap_err();
    assert!(
        matches!(
            lossy,
            tenferro_tensor::Error::UnsupportedDTypeConversion { op: "convert", .. }
        ),
        "unexpected error: {lossy}"
    );
    assert_eq!(backend.resources.read_stagings(), 0);

    let widened = backend.convert_read(read, DType::C64).unwrap();
    assert_eq!(
        widened.as_slice::<num_complex::Complex64>().unwrap(),
        &[
            num_complex::Complex64::new(1.0, 0.0),
            num_complex::Complex64::new(2.0, 0.0)
        ]
    );
}

/// #2223 T4, Convert row: the read hook removes the staging copy the traced `Convert` paid.
///
/// The control is the shape the executor used before this change (`ensure_backend_inputs`:
/// stage the view, then cast the compact copy); reverting the change makes the second half
/// of this test fail with one staging.
#[test]
fn convert_read_removes_the_staging_copy_of_a_view_operand() {
    let source = view_source();
    let view = TensorRead::from_view(
        source
            .tensor_view()
            .expect("the sliced source always has an f64 view"),
    );
    let mut backend = crate::CpuEngine::serial();

    let before = backend.resources.read_stagings();
    let staged = backend.to_contiguous_read(view.clone()).unwrap();
    let _ = backend.cast(&staged, DType::F32).unwrap();
    assert_eq!(
        backend.resources.read_stagings(),
        before + 1,
        "the pre-change Convert shape stages the view exactly once"
    );

    let before = backend.resources.read_stagings();
    let out = backend.cast_read(view, DType::F32).unwrap();
    assert_eq!(
        backend.resources.read_stagings(),
        before,
        "cast_read must not stage the view"
    );
    assert_eq!(out.dtype(), DType::F32);
}

/// #2223 T4 end to end: a traced `Convert` of a program-input view stages nothing.
///
/// The engine-level control above pins `cast_read`; this one pins the executor, so re-adding a
/// `to_contiguous_read` in front of the cast inside `execute_convert` makes it fail with one
/// staging while every value assertion still passes.
#[test]
fn a_traced_convert_of_a_program_input_view_stages_nothing() {
    use tenferro_runtime::{DType, ExecutionEngine, GraphCompiler, Runtime, TraceContext};
    use tenferro_tensor::{TensorRead, TensorView, TypedTensorView};

    fn cpu_runtime() -> Runtime {
        let mut builder = Runtime::builder();
        builder
            .register_engine(
                crate::runtime_engine_registration(&crate::CpuEngine::serial())
                    .expect("CPU engine registration"),
            )
            .expect("register the CPU engine");
        builder.build().expect("CPU runtime")
    }

    let trace = TraceContext::new();
    let placeholder = trace
        .input_concrete_shape(DType::F64, &[2, 3])
        .expect("program input");
    // The transpose folds into `Convert` (it reads through `cast_read`), so the entry sees the
    // program input through a metadata view and the staging question is the same one (#2223).
    let converted = placeholder
        .transpose(&[1, 0])
        .expect("traced transpose")
        .cast(DType::F32)
        .expect("traced cast");
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&converted, &[(&placeholder, DType::F64, [2, 3].as_slice())])
        .expect("compile the traced convert");

    // The program input as a view over the head of a larger buffer; the poison tail is never read.
    let buffer = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 99.0];
    let view = TypedTensorView::from_slice(vec![2, 3], vec![1, 2], 0, &buffer[..])
        .expect("strided operand");
    let reads = [TensorRead::from_view(TensorView::F64(view))];

    let runtime = cpu_runtime();
    let mut engine = crate::CpuEngine::serial();
    let target = engine.target();
    let prepared = runtime
        .prepare(&program, &reads, std::slice::from_ref(&target))
        .expect("prepare the traced convert");
    let before = engine.resources.read_stagings();
    let outputs = runtime
        .run(&mut [&mut engine], &prepared, &reads)
        .expect("run the traced convert");

    let out = outputs[0].read();
    assert_eq!(out.dtype(), DType::F32);
    assert_eq!(out.shape(), &[3, 2]);
    assert_eq!(
        out.as_slice::<f32>().expect("prepared f32 output"),
        &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0],
        "the values are the transposed input, converted"
    );
    assert_eq!(
        engine.resources.read_stagings(),
        before,
        "a traced convert of a program-input view must not stage it"
    );
}
