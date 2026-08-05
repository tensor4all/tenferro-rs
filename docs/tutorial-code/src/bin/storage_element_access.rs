use tenferro_runtime::TypedTensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut tensor = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0; 4])?;

    // A view is a metadata-only borrow and sees mutations to its owner.
    {
        let mut view = tensor.as_view_mut();
        *view.get_mut(&[1, 0]).ok_or("missing mutable element")? = 3.0;
    }
    let view = tensor.as_view();
    assert_eq!(view.as_slice()?, &[1.0, 3.0, 1.0, 1.0]);

    // duplicate is the explicit fresh-allocation boundary.
    let duplicate = view.duplicate()?;
    assert_eq!(duplicate.as_slice()?, &[1.0, 3.0, 1.0, 1.0]);
    assert_ne!(view.as_slice()?.as_ptr(), duplicate.as_slice()?.as_ptr());

    Ok(())
}
