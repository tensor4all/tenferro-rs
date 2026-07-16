use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{
    ContractionScalar, DType, DotGeneralAccumulation, Tensor, TensorRead, TensorView,
    TensorViewMut, TensorWrite, TypedTensor, TypedTensorView, TypedTensorViewMut, TypedTensorWrite,
};

use crate::{
    parse_einsum_subscripts, ConcreteEinsumPlan, TensorEinsumExt, TensorEinsumIntoExt,
    TensorReadEinsumExt, TensorReadEinsumIntoExt, TypedTensorEinsumExt, TypedTensorEinsumIntoExt,
    TypedTensorReadEinsumExt, TypedTensorReadEinsumIntoExt,
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
    let planned_result = plan.execute_typed([&lhs, &rhs], &mut backend).unwrap();

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
        tenferro_tensor::Error::ShapeMismatch {
            op: "TensorEinsumIntoExt::einsum_into",
            ..
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
        tenferro_tensor::Error::DTypeMismatch {
            op: "TensorEinsumIntoExt::einsum_into",
            lhs: DType::F32,
            rhs: DType::F64,
        }
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
        .find("fn tensor_value_from_read")
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
    let first = plan.execute([&lhs, &rhs], &mut backend).unwrap();
    let second = plan.execute([&lhs, &rhs2], &mut backend).unwrap();

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

    plan.execute_into(
        [&lhs, &rhs],
        &mut backend,
        TensorWrite::from_tensor(&mut out),
    )
    .unwrap();
    assert_f64_tensor(&out, &[2, 2], &[22.0, 28.0, 49.0, 64.0]);

    plan.execute_into(
        [&lhs, &rhs2],
        &mut backend,
        TensorWrite::from_tensor(&mut out),
    )
    .unwrap();
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
    plan.execute_read_into_accum(
        [TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs)],
        &mut backend,
        DotGeneralAccumulation::add_to(DType::F64).unwrap(),
        TensorWrite::from_tensor(&mut out),
    )
    .unwrap();
    assert_f64_tensor(&out, &[2, 2], &[23.0, 29.0, 50.0, 65.0]);

    let fallback_plan = ConcreteEinsumPlan::prepare([&lhs], "ij->").unwrap();
    let mut scalar_out = Tensor::from_vec_col_major(vec![], vec![2.0_f64]).unwrap();
    let accumulation =
        DotGeneralAccumulation::scaled(ContractionScalar::F64(0.5), ContractionScalar::F64(2.0))
            .unwrap();
    fallback_plan
        .execute_read_into_accum(
            [TensorRead::from_tensor(&lhs)],
            &mut backend,
            accumulation,
            TensorWrite::from_tensor(&mut scalar_out),
        )
        .unwrap();
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
    plan.execute_typed_into([&lhs, &rhs], &mut backend, &mut typed_out)
        .unwrap();
    assert_eq!(typed_out.as_slice().unwrap(), &[22.0, 28.0, 49.0, 64.0]);

    let mut typed_strided_data = [-1.0_f64; 8];
    {
        let out_view =
            TypedTensorViewMut::from_slice([2, 2], [1, 3], 1, &mut typed_strided_data).unwrap();
        plan.execute_typed_into(
            [&lhs, &rhs],
            &mut backend,
            TypedTensorWrite::from_view(out_view),
        )
        .unwrap();
    }
    assert_eq!(
        typed_strided_data,
        [-1.0, 22.0, 28.0, -1.0, 49.0, 64.0, -1.0, -1.0]
    );

    let lhs_erased = Tensor::from(lhs.clone());
    let rhs_erased = Tensor::from(rhs.clone());
    let mut strided_data = [-1.0_f64; 8];
    {
        let out_view = TensorViewMut::F64(
            TypedTensorViewMut::from_slice([2, 2], [1, 3], 1, &mut strided_data).unwrap(),
        );
        plan.execute_read_into(
            [
                TensorRead::from_tensor(&lhs_erased),
                TensorRead::from_tensor(&rhs_erased),
            ],
            &mut backend,
            TensorWrite::from_view(out_view),
        )
        .unwrap();
    }
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

    let err = plan
        .execute_into(
            [&lhs, &rhs],
            &mut backend,
            TensorWrite::from_tensor(&mut wrong_shape),
        )
        .unwrap_err();

    assert!(matches!(
        err,
        tenferro_tensor::Error::ShapeMismatch {
            op: "ConcreteEinsumPlan::execute",
            ..
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
    let tensor_result = tensor_plan.execute([&lhs, &rhs], &mut backend).unwrap();
    let typed_result = typed_plan
        .execute_typed([&typed_lhs, &typed_rhs], &mut backend)
        .unwrap();
    let read_string_result = read_string_plan
        .execute_read(read_inputs.as_slice(), &mut backend)
        .unwrap();
    let read_integer_result = read_integer_plan
        .execute_read(read_inputs.as_slice(), &mut backend)
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
    let lhs_erased = Tensor::from(lhs.clone());
    let rhs_erased = Tensor::from(rhs.clone());

    let mut backend = CpuBackend::new();
    let typed = plan.execute_typed([&lhs, &rhs], &mut backend).unwrap();
    let read = plan
        .execute_read(
            [
                TensorRead::from_tensor(&lhs_erased),
                TensorRead::from_tensor(&rhs_erased),
            ],
            &mut backend,
        )
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
    let shape_err = plan
        .execute([&lhs, &wrong_shape], &mut backend)
        .unwrap_err();
    let dtype_err = plan
        .execute([&lhs, &wrong_dtype], &mut backend)
        .unwrap_err();

    assert!(matches!(
        shape_err,
        tenferro_tensor::Error::ShapeMismatch {
            op: "ConcreteEinsumPlan::execute",
            ..
        }
    ));
    assert!(matches!(
        dtype_err,
        tenferro_tensor::Error::DTypeMismatch {
            op: "ConcreteEinsumPlan::execute",
            lhs: DType::F64,
            rhs: DType::F32,
        }
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
    let count_err = plan.execute([&lhs], &mut backend).unwrap_err();

    assert!(matches!(
        parse_err,
        tenferro_tensor::Error::InvalidConfig {
            op: "TensorEinsumExt::einsum",
            ..
        }
    ));
    assert!(matches!(
        count_err,
        tenferro_tensor::Error::InvalidConfig {
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
        tenferro_tensor::Error::DTypeMismatch {
            op: "test typed result",
            lhs: DType::F64,
            rhs: DType::F32,
        }
    ));
}
