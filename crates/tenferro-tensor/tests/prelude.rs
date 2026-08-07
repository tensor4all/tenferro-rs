use tenferro_tensor::prelude::*;

#[test]
fn prelude_constructs_owned_typed_and_erased_values() {
    let typed = TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 2], vec![1.0; 4]).unwrap();
    let view = typed.as_view();
    assert_eq!(view.shape(), &[2, 2]);

    let tensor = Tensor::from_vec_col_major([2], vec![1.0_f64, 2.0]).unwrap();
    let read = TensorRead::from_tensor(&tensor);
    assert_eq!(read.dtype(), DType::F64);
}
