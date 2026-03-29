use mdarray::{tensor, Array, DynRank};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{mdarray_to_tensor, tensor_to_mdarray};

#[test]
fn mdarray_tensor_roundtrip_preserves_shape_and_values() {
    let original: Array<f64, DynRank> = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();

    let tensor = mdarray_to_tensor(original.clone());
    assert_eq!(tensor.dims(), &[2, 3]);
    assert_eq!(tensor.strides(), &[3, 1]);

    let roundtrip = tensor_to_mdarray(tensor);
    assert_eq!(roundtrip.dims(), &[2, 3]);
    assert_eq!(roundtrip.into_vec(), original.into_vec());
}

#[test]
fn tensor_to_mdarray_materializes_logical_values_from_column_major_input() {
    let tensor = Tensor::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let md = tensor_to_mdarray(tensor);
    assert_eq!(md.dims(), &[2, 3]);
    assert_eq!(md.into_vec(), vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
}

#[test]
fn mdarray_to_tensor_produces_row_major_strides() {
    let md: Array<f64, DynRank> = tensor![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]].into_dyn();

    let tensor = mdarray_to_tensor(md);
    assert_eq!(tensor.dims(), &[3, 2]);
    assert_eq!(tensor.strides(), &[2, 1]);

    let row_major = tensor.into_contiguous(MemoryOrder::RowMajor);
    assert_eq!(
        row_major.try_into_data_vec().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn tensor_to_mdarray_handles_permuted_views() {
    let tensor = Tensor::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        MemoryOrder::RowMajor,
    )
    .unwrap();
    let permuted = tensor.permute(&[1, 0]).unwrap();

    let md = tensor_to_mdarray(permuted);
    assert_eq!(md.dims(), &[3, 2]);
    assert_eq!(md.into_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}
