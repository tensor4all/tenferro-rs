use std::collections::HashMap;

use tenferro_tensor::MemoryOrder;

use super::*;

fn tensor(data: &[f64], dims: &[usize]) -> Tensor<f64> {
    Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn build_size_dict_merges_extra_sizes() {
    let subs = Subscripts::new(&[&['i' as u32, 'j' as u32]], &['i' as u32, 'k' as u32]);
    let extra = HashMap::from([('k' as u32, 5usize)]);

    let sizes = build_size_dict(&subs, &[&[2, 3]], Some(&extra)).unwrap();

    assert_eq!(sizes.get(&('i' as u32)), Some(&2));
    assert_eq!(sizes.get(&('j' as u32)), Some(&3));
    assert_eq!(sizes.get(&('k' as u32)), Some(&5));
}

#[test]
fn compute_output_shape_rejects_unknown_label() {
    let sizes = HashMap::from([('i' as u32, 2usize)]);
    let err = compute_output_shape(&['i' as u32, 'j' as u32], &sizes).unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)));
}

#[test]
fn tensor_get_and_unflatten_index_follow_column_major_order() {
    let t = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    assert_eq!(unflatten_index(4, &[2, 3]), vec![0, 2]);
    assert!((tensor_get(&t, &[0, 2]) - 5.0).abs() < 1e-10);
    assert!((tensor_get(&t, &[1, 1]) - 4.0).abs() < 1e-10);
}
