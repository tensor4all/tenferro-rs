use tenferro_tensor::{Rank, TypedTensor};

#[test]
fn as_view_paths_do_not_allocate_or_clone_storage() {
    let tensor = TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 2], vec![1.0; 4]).unwrap();
    let before = tensor.storage_counters();
    let view = tensor.as_view();
    let after_read = tensor.storage_counters();
    assert_eq!(view.shape(), &[2, 2]);
    assert_eq!(before, after_read);

    let mut tensor = tensor.duplicate().unwrap();
    let before_mut = tensor.storage_counters();
    let view_mut = tensor.as_view_mut().unwrap();
    let after_mut = tensor.storage_counters();
    assert_eq!(view_mut.shape(), &[2, 2]);
    assert_eq!(before_mut, after_mut);
}

