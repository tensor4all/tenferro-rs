use tenferro_runtime::{Tensor, TypedTensor};

fn main() {
    let typed =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
            .unwrap();
    assert_eq!(typed.as_slice().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    let dynamic =
        Tensor::from_vec_row_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert_eq!(
        dynamic.as_slice::<f64>().unwrap(),
        typed.as_slice().unwrap()
    );
}
