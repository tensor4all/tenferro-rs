use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{
    BackendSessionHost, ContractionScalar, DType, DotGeneralAccumulation, Tensor, TensorRead,
    TensorView, TensorViewMut, TensorWrite, TypedTensor, TypedTensorView, TypedTensorViewMut,
    TypedTensorWrite,
};

use crate::{
    parse_einsum_subscripts, ConcreteEinsumPlan, Error, TensorEinsumExt, TensorEinsumIntoExt,
    TensorReadEinsumExt, TensorReadEinsumIntoExt, TensorTensordotExt, TypedTensorEinsumExt,
    TypedTensorEinsumIntoExt, TypedTensorReadEinsumExt, TypedTensorReadEinsumIntoExt,
    TypedTensorTensordotExt,
};

fn assert_f64_tensor(tensor: &Tensor, shape: &[usize], expected: &[f64]) {
    assert_eq!(tensor.shape(), shape);
    assert_eq!(tensor.as_slice::<f64>().unwrap(), expected);
}

#[test]
fn public_tensor_einsum_ext_executes_dtype_erased_inputs() {
    let mut backend = CpuBackend::new();
    let lhs =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs =
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let result = [&lhs, &rhs].einsum("ij,jk->ik", &mut backend).unwrap();

    assert_f64_tensor(&result, &[2, 2], &[22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn concrete_tensordot_matches_einsum_for_erased_and_typed_tensors() {
    let mut backend = CpuBackend::new();
    let lhs = Tensor::from_vec_col_major([2], vec![1.0_f64, 2.0]).unwrap();
    let rhs = Tensor::from_vec_col_major([2], vec![3.0_f64, 4.0]).unwrap();
    let erased = lhs
        .tensordot(&rhs, crate::TensorDotAxes::Count(1), &mut backend)
        .unwrap();
    assert_eq!(erased.as_slice::<f64>().unwrap(), &[11.0]);

    let lhs = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    let rhs = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![3.0, 4.0]).unwrap();
    let typed = lhs
        .tensordot(&rhs, crate::TensorDotAxes::Count(1), &mut backend)
        .unwrap();
    assert_eq!(typed.as_slice().unwrap(), &[11.0]);
}

#[test]
fn public_tensor_einsum_ext_accepts_slice_and_integer_subscripts() {
    let mut backend = CpuBackend::new();
    let lhs =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs =
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let subscripts = parse_einsum_subscripts("ij,jk->ik").unwrap();
    let inputs = vec![&lhs, &rhs];

    let slice_result = inputs
        .as_slice()
        .einsum_subscripts(&subscripts, &mut backend)
        .unwrap();
    let array_result = [&lhs, &rhs]
        .einsum_subscripts(&subscripts, &mut backend)
        .unwrap();

    assert_f64_tensor(&slice_result, &[2, 2], &[22.0, 28.0, 49.0, 64.0]);
    assert_f64_tensor(&array_result, &[2, 2], &[22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn public_typed_tensor_einsum_ext_preserves_complex_dtype() {
    let mut backend = CpuBackend::new();
    let lhs = TypedTensor::<Complex64>::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 2.0),
        ],
    )
    .unwrap();
    let rhs = TypedTensor::<Complex64>::from_vec_col_major(
        vec![2, 1],
        vec![Complex64::new(5.0, 0.0), Complex64::new(6.0, -1.0)],
    )
    .unwrap();

    let result = [&lhs, &rhs].einsum("ij,jk->ik", &mut backend).unwrap();
    let subscripts = parse_einsum_subscripts("ij,jk->ik").unwrap();
    let integer_result = [&lhs, &rhs]
        .einsum_subscripts(&subscripts, &mut backend)
        .unwrap();
    let plan = ConcreteEinsumPlan::prepare_typed_subscripts([&lhs, &rhs], &subscripts).unwrap();
    let planned_result = backend
        .with_backend_session(|session| plan.execute_typed([&lhs, &rhs], session))
        .unwrap();

    assert_eq!(result.shape(), &[2, 1]);
    assert_eq!(
        result.as_slice().unwrap(),
        &[Complex64::new(23.0, 2.0), Complex64::new(36.0, 3.0)]
    );
    assert_eq!(
        integer_result.as_slice().unwrap(),
        &[Complex64::new(23.0, 2.0), Complex64::new(36.0, 3.0)]
    );
    assert_eq!(
        planned_result.as_slice().unwrap(),
        &[Complex64::new(23.0, 2.0), Complex64::new(36.0, 3.0)]
    );
}

#[test]
fn public_typed_tensor_einsum_ext_accepts_slice_and_integer_subscripts() {
    let mut backend = CpuBackend::new();
    let lhs =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let rhs =
        TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let subscripts = parse_einsum_subscripts("ij,jk->ik").unwrap();
    let inputs = vec![&lhs, &rhs];

    let slice_result = inputs
        .as_slice()
        .einsum_subscripts(&subscripts, &mut backend)
        .unwrap();
    let array_result = [&lhs, &rhs]
        .einsum_subscripts(&subscripts, &mut backend)
        .unwrap();

    assert_eq!(slice_result.as_slice().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
    assert_eq!(array_result.as_slice().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn public_typed_tensor_read_einsum_ext_accepts_borrowed_strided_complex_views() {
    let mut backend = CpuBackend::new();
    let matrix_data = [
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
        Complex64::new(5.0, 0.0),
        Complex64::new(6.0, 0.0),
    ];
    let vector_data = [
        Complex64::new(10.0, 0.0),
        Complex64::new(20.0, 0.0),
        Complex64::new(30.0, 0.0),
    ];
    let matrix_view = TypedTensorView::from_slice([2, 3], [3, 1], 0, &matrix_data).unwrap();
    let vector_view = TypedTensorView::from_col_major(&[3], &vector_data).unwrap();
    let mut out =
        TypedTensor::<Complex64>::from_vec_col_major(vec![2], vec![Complex64::new(0.0, 0.0); 2])
            .unwrap();

    let result = [matrix_view.clone(), vector_view.clone()]
        .einsum_read("ij,j->i", &mut backend)
        .unwrap();
    [matrix_view, vector_view]
        .einsum_read_into("ij,j->i", &mut backend, &mut out)
        .unwrap();

    assert_eq!(
        result.as_slice().unwrap(),
        &[Complex64::new(140.0, 0.0), Complex64::new(320.0, 0.0)]
    );
    assert_eq!(
        out.as_slice().unwrap(),
        &[Complex64::new(140.0, 0.0), Complex64::new(320.0, 0.0)]
    );
}

#[test]
fn public_tensor_read_einsum_ext_accepts_strided_views() {
    let mut backend = CpuBackend::new();
    let matrix_data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let vector = Tensor::from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0]).unwrap();
    let row_major_view = TypedTensorView::from_slice([2, 3], [3, 1], 0, &matrix_data).unwrap();
    let inputs = [
        TensorRead::from_view(TensorView::F64(row_major_view)),
        TensorRead::from_tensor(&vector),
    ];

    let result = inputs.einsum_read("ij,j->i", &mut backend).unwrap();

    assert_f64_tensor(&result, &[2], &[140.0, 320.0]);
}

#[test]
fn public_tensor_read_einsum_ext_accepts_slice_and_integer_subscripts() {
    let mut backend = CpuBackend::new();
    let matrix_data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let vector = Tensor::from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0]).unwrap();
    let row_major_view = TypedTensorView::from_slice([2, 3], [3, 1], 0, &matrix_data).unwrap();
    let inputs = [
        TensorRead::from_view(TensorView::F64(row_major_view)),
        TensorRead::from_tensor(&vector),
    ];
    let subscripts = parse_einsum_subscripts("ij,j->i").unwrap();

    let slice_result = inputs
        .as_slice()
        .einsum_read_subscripts(&subscripts, &mut backend)
        .unwrap();
    let array_result = inputs
        .einsum_read_subscripts(&subscripts, &mut backend)
        .unwrap();

    assert_f64_tensor(&slice_result, &[2], &[140.0, 320.0]);
    assert_f64_tensor(&array_result, &[2], &[140.0, 320.0]);
}

#[test]
fn public_einsum_into_writes_dynamic_typed_and_read_outputs() {
    let mut backend = CpuBackend::new();
    let lhs =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs =
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let mut dynamic_out = Tensor::from_vec_col_major(vec![2, 2], vec![-1.0_f64; 4]).unwrap();
    [&lhs, &rhs]
        .einsum_into(
            "ij,jk->ik",
            &mut backend,
            TensorWrite::from_tensor(&mut dynamic_out),
        )
        .unwrap();
    assert_f64_tensor(&dynamic_out, &[2, 2], &[22.0, 28.0, 49.0, 64.0]);

    let typed_lhs =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let typed_rhs =
        TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let mut typed_out = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![-1.0; 4]).unwrap();
    [&typed_lhs, &typed_rhs]
        .einsum_into("ij,jk->ik", &mut backend, &mut typed_out)
        .unwrap();
    assert_eq!(typed_out.as_slice().unwrap(), &[22.0, 28.0, 49.0, 64.0]);

    let mut typed_strided_data = [-1.0_f64; 8];
    {
        let out_view =
            TypedTensorViewMut::from_slice([2, 2], [1, 3], 1, &mut typed_strided_data).unwrap();
        [&typed_lhs, &typed_rhs]
            .einsum_into(
                "ij,jk->ik",
                &mut backend,
                TypedTensorWrite::from_view(out_view),
            )
            .unwrap();
    }
    assert_eq!(
        typed_strided_data,
        [-1.0, 22.0, 28.0, -1.0, 49.0, 64.0, -1.0, -1.0]
    );

    let mut strided_data = [-1.0_f64; 8];
    {
        let out_view = TensorViewMut::F64(
            TypedTensorViewMut::from_slice([2, 2], [1, 3], 1, &mut strided_data).unwrap(),
        );
        let inputs = [TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs)];
        inputs
            .einsum_read_into("ij,jk->ik", &mut backend, TensorWrite::from_view(out_view))
            .unwrap();
    }
    assert_eq!(
        strided_data,
        [-1.0, 22.0, 28.0, -1.0, 49.0, 64.0, -1.0, -1.0]
    );
}

#[test]
fn public_einsum_into_preserves_complex_dtype() {
    let mut backend = CpuBackend::new();
    let lhs = TypedTensor::<Complex64>::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 2.0),
        ],
    )
    .unwrap();
    let rhs = TypedTensor::<Complex64>::from_vec_col_major(
        vec![2, 1],
        vec![Complex64::new(5.0, 0.0), Complex64::new(6.0, -1.0)],
    )
    .unwrap();
    let mut out =
        TypedTensor::<Complex64>::from_vec_col_major(vec![2, 1], vec![Complex64::new(0.0, 0.0); 2])
            .unwrap();

    [&lhs, &rhs]
        .einsum_into("ij,jk->ik", &mut backend, &mut out)
        .unwrap();

    assert_eq!(
        out.as_slice().unwrap(),
        &[Complex64::new(23.0, 2.0), Complex64::new(36.0, 3.0)]
    );
}

#[test]
fn public_einsum_into_rejects_output_shape_and_dtype_mismatch() {
    let mut backend = CpuBackend::new();
    let lhs =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs =
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let mut wrong_shape = Tensor::from_vec_col_major(vec![4], vec![0.0_f64; 4]).unwrap();
    let shape_err = [&lhs, &rhs]
        .einsum_into(
            "ij,jk->ik",
            &mut backend,
            TensorWrite::from_tensor(&mut wrong_shape),
        )
        .unwrap_err();
    assert!(matches!(
        shape_err,
        Error::Validation {
            op: "TensorEinsumIntoExt::einsum_into",
            source: tenferro_tensor::ValidationError::ShapeMismatch(_),
        }
    ));

    let mut wrong_dtype = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f32; 4]).unwrap();
    let dtype_err = [&lhs, &rhs]
        .einsum_into(
            "ij,jk->ik",
            &mut backend,
            TensorWrite::from_tensor(&mut wrong_dtype),
        )
        .unwrap_err();
    assert!(matches!(
        dtype_err,
        Error::Tensor(tenferro_tensor::Error::Validation {
            op: "TensorEinsumIntoExt::einsum_into",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        })
    ));
}

#[test]
fn einsum_into_gemm_fast_path_dispatches_to_backend_into_before_owned_fallback() {
    let source = include_str!("eager.rs");
    let start = source
        .find("pub(crate) fn eager_einsum_exec_read_into")
        .expect("missing eager_einsum_exec_read_into");
    let tail = &source[start..];
    let end = tail
        .find("pub(crate) fn eager_einsum_exec_read_into_accum")
        .expect("missing following function");
    let body = &tail[..end];

    let into_call = body
        .find("exec.dot_general_read_into")
        .expect("GEMM-compatible einsum_into must call backend read-into");
    let fallback = body
        .find("eager_einsum_exec_read(exec, inputs, tree)")
        .expect("general einsum_into fallback should keep using owned execution");

    assert!(
        into_call < fallback,
        "GEMM-compatible einsum_into should try backend read-into before owned fallback"
    );
}

#[test]
fn concrete_einsum_plan_executes_without_replanning_contract() {
    let lhs =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs =
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs2 =
        Tensor::from_vec_col_major(vec![3, 2], vec![2.0_f64, 0.0, 1.0, 3.0, 4.0, 5.0]).unwrap();
    let plan = ConcreteEinsumPlan::prepare([&lhs, &rhs], "ij,jk->ik").unwrap();

    let mut backend = CpuBackend::new();
    let (first, second) = backend
        .with_backend_session(|session| -> crate::Result<(Tensor, Tensor)> {
            let first = plan.execute([&lhs, &rhs], session)?;
            let second = plan.execute([&lhs, &rhs2], session)?;
            Ok((first, second))
        })
        .unwrap();

    assert_f64_tensor(&first, &[2, 2], &[22.0, 28.0, 49.0, 64.0]);
    assert_f64_tensor(&second, &[2, 2], &[7.0, 10.0, 40.0, 52.0]);
}

#[test]
fn concrete_einsum_plan_execute_into_writes_reused_outputs() {
    let lhs =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs =
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs2 =
        Tensor::from_vec_col_major(vec![3, 2], vec![2.0_f64, 0.0, 1.0, 3.0, 4.0, 5.0]).unwrap();
    let plan = ConcreteEinsumPlan::prepare([&lhs, &rhs], "ij,jk->ik").unwrap();
    let mut backend = CpuBackend::new();
    let mut out = Tensor::from_vec_col_major(vec![2, 2], vec![-1.0_f64; 4]).unwrap();

    let first = backend
        .with_backend_session(|session| -> crate::Result<Tensor> {
            plan.execute_into([&lhs, &rhs], session, TensorWrite::from_tensor(&mut out))?;
            let first = out.duplicate().unwrap();
            plan.execute_into([&lhs, &rhs2], session, TensorWrite::from_tensor(&mut out))?;
            Ok(first)
        })
        .unwrap();
    assert_f64_tensor(&first, &[2, 2], &[22.0, 28.0, 49.0, 64.0]);
    assert_f64_tensor(&out, &[2, 2], &[7.0, 10.0, 40.0, 52.0]);
}

#[test]
fn concrete_einsum_plan_execute_read_into_accum_updates_outputs() {
    let lhs =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs =
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let mut backend = CpuBackend::new();

    let plan = ConcreteEinsumPlan::prepare([&lhs, &rhs], "ij,jk->ik").unwrap();
    let mut out = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();
    let fallback_plan = ConcreteEinsumPlan::prepare([&lhs], "ij->").unwrap();
    let mut scalar_out = Tensor::from_vec_col_major(vec![], vec![2.0_f64]).unwrap();
    let accumulation =
        DotGeneralAccumulation::scaled(ContractionScalar::F64(0.5), ContractionScalar::F64(2.0))
            .unwrap();
    backend
        .with_backend_session(|session| -> crate::Result<()> {
            plan.execute_read_into_accum(
                [TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs)],
                session,
                DotGeneralAccumulation::add_to(DType::F64).unwrap(),
                TensorWrite::from_tensor(&mut out),
            )?;
            fallback_plan.execute_read_into_accum(
                [TensorRead::from_tensor(&lhs)],
                session,
                accumulation,
                TensorWrite::from_tensor(&mut scalar_out),
            )
        })
        .unwrap();
    assert_f64_tensor(&out, &[2, 2], &[23.0, 29.0, 50.0, 65.0]);
    assert_f64_tensor(&scalar_out, &[], &[14.5]);
}

#[test]
fn concrete_einsum_plan_execute_typed_and_read_into_outputs() {
    let lhs =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let rhs =
        TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let plan = ConcreteEinsumPlan::prepare_typed([&lhs, &rhs], "ij,jk->ik").unwrap();
    let mut backend = CpuBackend::new();

    let mut typed_out = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![-1.0; 4]).unwrap();
    let mut typed_strided_data = [-1.0_f64; 8];
    backend
        .with_backend_session(|session| -> crate::Result<()> {
            plan.execute_typed_into([&lhs, &rhs], session, &mut typed_out)?;
            let out_view =
                TypedTensorViewMut::from_slice([2, 2], [1, 3], 1, &mut typed_strided_data).unwrap();
            plan.execute_typed_into([&lhs, &rhs], session, TypedTensorWrite::from_view(out_view))
        })
        .unwrap();
    assert_eq!(typed_out.as_slice().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
    assert_eq!(
        typed_strided_data,
        [-1.0, 22.0, 28.0, -1.0, 49.0, 64.0, -1.0, -1.0]
    );

    let lhs_erased = Tensor::from(lhs.duplicate().unwrap());
    let rhs_erased = Tensor::from(rhs.duplicate().unwrap());
    let mut strided_data = [-1.0_f64; 8];
    backend
        .with_backend_session(|session| {
            let out_view = TensorViewMut::F64(
                TypedTensorViewMut::from_slice([2, 2], [1, 3], 1, &mut strided_data).unwrap(),
            );
            plan.execute_read_into(
                [
                    TensorRead::from_tensor(&lhs_erased),
                    TensorRead::from_tensor(&rhs_erased),
                ],
                session,
                TensorWrite::from_view(out_view),
            )
        })
        .unwrap();
    assert_eq!(
        strided_data,
        [-1.0, 22.0, 28.0, -1.0, 49.0, 64.0, -1.0, -1.0]
    );
}

#[test]
fn concrete_einsum_plan_execute_into_rejects_incompatible_output() {
    let lhs =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs =
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let plan = ConcreteEinsumPlan::prepare([&lhs, &rhs], "ij,jk->ik").unwrap();
    let mut backend = CpuBackend::new();
    let mut wrong_shape = Tensor::from_vec_col_major(vec![4], vec![0.0_f64; 4]).unwrap();

    let err = backend
        .with_backend_session(|session| {
            plan.execute_into(
                [&lhs, &rhs],
                session,
                TensorWrite::from_tensor(&mut wrong_shape),
            )
        })
        .unwrap_err();

    assert!(matches!(
        err,
        Error::Validation {
            op: "ConcreteEinsumPlan::execute",
            source: tenferro_tensor::ValidationError::ShapeMismatch(_),
        }
    ));
}

#[test]
fn concrete_einsum_plan_prepares_integer_and_read_variants() {
    let lhs =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs =
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let typed_lhs =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let typed_rhs =
        TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let read_inputs = [TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs)];
    let subscripts = parse_einsum_subscripts("ij,jk->ik").unwrap();

    let tensor_plan = ConcreteEinsumPlan::prepare_subscripts([&lhs, &rhs], &subscripts).unwrap();
    let typed_plan =
        ConcreteEinsumPlan::prepare_typed_subscripts([&typed_lhs, &typed_rhs], &subscripts)
            .unwrap();
    let read_string_plan =
        ConcreteEinsumPlan::prepare_read(read_inputs.as_slice(), "ij,jk->ik").unwrap();
    let read_integer_plan =
        ConcreteEinsumPlan::prepare_read_subscripts(read_inputs.as_slice(), &subscripts).unwrap();
    let debug = format!("{tensor_plan:?}");

    let mut backend = CpuBackend::new();
    let tensor_result = backend
        .with_backend_session(|session| tensor_plan.execute([&lhs, &rhs], session))
        .unwrap();
    let (typed_result, read_string_result, read_integer_result) = backend
        .with_backend_session(
            |session| -> crate::Result<(TypedTensor<f64>, Tensor, Tensor)> {
                let typed_result = typed_plan.execute_typed([&typed_lhs, &typed_rhs], session)?;
                let read_string_result =
                    read_string_plan.execute_read(read_inputs.as_slice(), session)?;
                let read_integer_result =
                    read_integer_plan.execute_read(read_inputs.as_slice(), session)?;
                Ok((typed_result, read_string_result, read_integer_result))
            },
        )
        .unwrap();

    assert!(debug.contains("ConcreteEinsumPlan"));
    assert!(debug.contains("F64"));
    assert_f64_tensor(&tensor_result, &[2, 2], &[22.0, 28.0, 49.0, 64.0]);
    assert_eq!(typed_result.as_slice().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
    assert_f64_tensor(&read_string_result, &[2, 2], &[22.0, 28.0, 49.0, 64.0]);
    assert_f64_tensor(&read_integer_result, &[2, 2], &[22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn concrete_einsum_plan_executes_read_and_typed_inputs() {
    let lhs =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let rhs =
        TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let plan = ConcreteEinsumPlan::prepare_typed([&lhs, &rhs], "ij,jk->ik").unwrap();
    let lhs_erased = Tensor::from(lhs.duplicate().unwrap());
    let rhs_erased = Tensor::from(rhs.duplicate().unwrap());

    let mut backend = CpuBackend::new();
    let (typed, read) = backend
        .with_backend_session(|session| -> crate::Result<(TypedTensor<f64>, Tensor)> {
            let typed = plan.execute_typed([&lhs, &rhs], session)?;
            let read = plan.execute_read(
                [
                    TensorRead::from_tensor(&lhs_erased),
                    TensorRead::from_tensor(&rhs_erased),
                ],
                session,
            )?;
            Ok((typed, read))
        })
        .unwrap();

    assert_eq!(typed.as_slice().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
    assert_f64_tensor(&read, &[2, 2], &[22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn concrete_einsum_plan_rejects_shape_and_dtype_mismatches() {
    let lhs =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs =
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let wrong_shape = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    let wrong_dtype = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f32; 6]).unwrap();
    let plan = ConcreteEinsumPlan::prepare([&lhs, &rhs], "ij,jk->ik").unwrap();

    let mut backend = CpuBackend::new();
    let (shape_err, dtype_err) = backend.with_backend_session(|session| {
        let shape_err = plan.execute([&lhs, &wrong_shape], session).unwrap_err();
        let dtype_err = plan.execute([&lhs, &wrong_dtype], session).unwrap_err();
        (shape_err, dtype_err)
    });

    assert!(matches!(
        shape_err,
        Error::Validation {
            op: "ConcreteEinsumPlan::execute",
            source: tenferro_tensor::ValidationError::ShapeMismatch(_),
        }
    ));
    assert!(matches!(
        dtype_err,
        Error::Tensor(tenferro_tensor::Error::Validation {
            op: "ConcreteEinsumPlan::execute",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        })
    ));
}

#[test]
fn concrete_einsum_public_api_reports_parse_and_input_count_errors() {
    let lhs =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs =
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let mut backend = CpuBackend::new();

    let parse_err = [&lhs, &rhs]
        .einsum("ij,(jk)->ik", &mut backend)
        .unwrap_err();
    let plan = ConcreteEinsumPlan::prepare([&lhs, &rhs], "ij,jk->ik").unwrap();
    let count_err = backend
        .with_backend_session(|session| plan.execute([&lhs], session))
        .unwrap_err();

    assert!(matches!(parse_err, Error::InvalidSubscripts { .. }));
    assert!(matches!(
        count_err,
        Error::Validation {
            op: "ConcreteEinsumPlan::execute",
            ..
        }
    ));
}

#[test]
fn concrete_einsum_typed_result_reports_defensive_dtype_mismatch() {
    let tensor = Tensor::from_vec_col_major(vec![1], vec![1.0_f32]).unwrap();

    let err = crate::concrete::into_typed_result::<f64>(tensor, "test typed result").unwrap_err();

    assert!(matches!(
        err,
        Error::Tensor(tenferro_tensor::Error::Validation {
            op: "test typed result",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        })
    ));
}

// ---------------------------------------------------------------------------
// Direct session tests for the ConcreteEinsumPlan execute surface
// ---------------------------------------------------------------------------

fn f64_plan_fixture() -> (Tensor, Tensor, ConcreteEinsumPlan) {
    let lhs =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs =
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let plan = ConcreteEinsumPlan::prepare([&lhs, &rhs], "ij,jk->ik").unwrap();
    (lhs, rhs, plan)
}

#[test]
fn concrete_einsum_plan_execute_in_session_matches_expected_values() {
    let (lhs, rhs, plan) = f64_plan_fixture();
    let mut backend = CpuBackend::new();

    let in_session = backend
        .with_backend_session(|session| plan.execute([&lhs, &rhs], session))
        .unwrap();
    assert_f64_tensor(&in_session, &[2, 2], &[22.0, 28.0, 49.0, 64.0]);

    let reads = [TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs)];
    let in_session = backend
        .with_backend_session(|session| plan.execute_read(reads.clone(), session))
        .unwrap();
    assert_f64_tensor(&in_session, &[2, 2], &[22.0, 28.0, 49.0, 64.0]);

    let typed_lhs =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let typed_rhs =
        TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let typed_plan =
        ConcreteEinsumPlan::prepare_typed([&typed_lhs, &typed_rhs], "ij,jk->ik").unwrap();
    let in_session = backend
        .with_backend_session(|session| typed_plan.execute_typed([&typed_lhs, &typed_rhs], session))
        .unwrap();
    assert_eq!(in_session.as_slice().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn concrete_einsum_plan_execute_in_session_rejects_bad_inputs() {
    let (lhs, _rhs, plan) = f64_plan_fixture();
    let wrong_shape = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    let wrong_dtype = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f32; 6]).unwrap();
    let mut backend = CpuBackend::new();

    let in_session = backend
        .with_backend_session(|session| plan.execute([&lhs, &wrong_shape], session))
        .unwrap_err();
    assert!(matches!(
        in_session,
        Error::Validation {
            op: "ConcreteEinsumPlan::execute",
            source: tenferro_tensor::ValidationError::ShapeMismatch(_),
        }
    ));

    let in_session = backend
        .with_backend_session(|session| plan.execute([&lhs, &wrong_dtype], session))
        .unwrap_err();
    assert!(matches!(
        in_session,
        Error::Tensor(tenferro_tensor::Error::Validation {
            op: "ConcreteEinsumPlan::execute",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        })
    ));

    let in_session = backend
        .with_backend_session(|session| plan.execute([&lhs], session))
        .unwrap_err();
    assert!(matches!(
        in_session,
        Error::Validation {
            op: "ConcreteEinsumPlan::execute",
            ..
        }
    ));
}

#[test]
fn concrete_einsum_plan_execute_typed_in_session_validates_dtype() {
    let typed_lhs =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let typed_rhs =
        TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let plan = ConcreteEinsumPlan::prepare_typed([&typed_lhs, &typed_rhs], "ij,jk->ik").unwrap();
    let mut backend = CpuBackend::new();

    let in_session = backend
        .with_backend_session(|session| plan.execute_typed([&typed_lhs, &typed_rhs], session))
        .unwrap();
    assert_eq!(in_session.as_slice().unwrap(), &[22.0, 28.0, 49.0, 64.0]);

    // Requesting a scalar type different from the prepared dtype must fail
    // through validate_inputs' dtype contract.
    let f32_lhs = TypedTensor::<f32>::from_vec_col_major(vec![2, 3], vec![1.0_f32; 6]).unwrap();
    let f32_rhs = TypedTensor::<f32>::from_vec_col_major(vec![3, 2], vec![1.0_f32; 6]).unwrap();
    let in_session = backend
        .with_backend_session(|session| plan.execute_typed([&f32_lhs, &f32_rhs], session))
        .unwrap_err();
    assert!(matches!(
        in_session,
        Error::Tensor(tenferro_tensor::Error::Validation {
            op: "ConcreteEinsumPlan::execute",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        })
    ));
}

#[test]
fn concrete_einsum_plan_execute_into_in_session_validates_output() {
    let (lhs, rhs, plan) = f64_plan_fixture();
    let mut backend = CpuBackend::new();

    // execute_into: incompatible output shape.
    let mut wrong_shape = Tensor::from_vec_col_major(vec![4], vec![0.0_f64; 4]).unwrap();
    let in_session = backend
        .with_backend_session(|session| {
            plan.execute_into(
                [&lhs, &rhs],
                session,
                TensorWrite::from_tensor(&mut wrong_shape),
            )
        })
        .unwrap_err();
    assert!(matches!(
        in_session,
        Error::Validation {
            op: "ConcreteEinsumPlan::execute",
            source: tenferro_tensor::ValidationError::ShapeMismatch(_),
        }
    ));

    // execute_read_into: incompatible output dtype.
    let mut wrong_dtype_out = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f32; 4]).unwrap();
    let in_session = backend
        .with_backend_session(|session| {
            plan.execute_read_into(
                [TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs)],
                session,
                TensorWrite::from_tensor(&mut wrong_dtype_out),
            )
        })
        .unwrap_err();
    assert!(matches!(
        in_session,
        Error::Tensor(tenferro_tensor::Error::Validation {
            op: "ConcreteEinsumPlan::execute",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        })
    ));

    // execute_typed_into: incompatible output shape.
    let typed_lhs =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let typed_rhs =
        TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let typed_plan =
        ConcreteEinsumPlan::prepare_typed([&typed_lhs, &typed_rhs], "ij,jk->ik").unwrap();
    let mut typed_out = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![0.0; 3]).unwrap();
    let in_session = backend
        .with_backend_session(|session| {
            typed_plan.execute_typed_into([&typed_lhs, &typed_rhs], session, &mut typed_out)
        })
        .unwrap_err();
    assert!(matches!(
        in_session,
        Error::Validation {
            op: "ConcreteEinsumPlan::execute",
            source: tenferro_tensor::ValidationError::ShapeMismatch(_),
        }
    ));

    // execute_read_into_accum: incompatible output shape.
    let mut accum_out = Tensor::from_vec_col_major(vec![4], vec![0.0_f64; 4]).unwrap();
    let accumulation = DotGeneralAccumulation::add_to(DType::F64).unwrap();
    let in_session = backend
        .with_backend_session(|session| {
            plan.execute_read_into_accum(
                [TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs)],
                session,
                accumulation,
                TensorWrite::from_tensor(&mut accum_out),
            )
        })
        .unwrap_err();
    assert!(matches!(
        in_session,
        Error::Validation {
            op: "ConcreteEinsumPlan::execute",
            source: tenferro_tensor::ValidationError::ShapeMismatch(_),
        }
    ));

    // Valid into execution writes the expected values.
    let mut out = Tensor::from_vec_col_major(vec![2, 2], vec![-1.0_f64; 4]).unwrap();
    backend
        .with_backend_session(|session| {
            plan.execute_into([&lhs, &rhs], session, TensorWrite::from_tensor(&mut out))
        })
        .unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
}

/// A non-`Send` output adapter: `PhantomData<Rc<()>>` makes the type `!Send`
/// (a `Cell` would not — `Cell<T>` is `Send` but `!Sync`). Passing this
/// adapter to `ConcreteEinsumPlan::execute_typed_into` only compiles because
/// the caller converts `out` to `TypedTensorWrite` before the
/// `with_backend_session` closure, so `O` itself never needs `Send`.
struct NonSendIntoWrite<'a, T> {
    write: TypedTensorWrite<'a, T>,
    _not_send: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl<'a, T> From<NonSendIntoWrite<'a, T>> for TypedTensorWrite<'a, T> {
    fn from(adapter: NonSendIntoWrite<'a, T>) -> Self {
        adapter.write
    }
}

#[test]
fn concrete_einsum_plan_execute_typed_into_accepts_non_send_adapter() {
    let typed_lhs =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let typed_rhs =
        TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let plan = ConcreteEinsumPlan::prepare_typed([&typed_lhs, &typed_rhs], "ij,jk->ik").unwrap();

    let mut backend = CpuBackend::new();
    let mut out = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![0.0; 4]).unwrap();
    let write = TypedTensorWrite::from(NonSendIntoWrite {
        write: TypedTensorWrite::from_tensor(&mut out),
        _not_send: std::marker::PhantomData,
    });
    backend
        .with_backend_session(|session| {
            plan.execute_typed_into([&typed_lhs, &typed_rhs], session, write)
        })
        .unwrap();
    assert_eq!(out.as_slice().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
}
