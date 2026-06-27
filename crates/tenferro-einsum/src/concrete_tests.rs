use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{DType, Tensor, TensorRead, TensorView, TypedTensor, TypedTensorView};

use crate::{
    parse_einsum_subscripts, ConcreteEinsumPlan, TensorEinsumExt, TensorReadEinsumExt,
    TypedTensorEinsumExt,
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
