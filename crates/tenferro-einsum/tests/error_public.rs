use tenferro_einsum::{Error, Result};
use tenferro_tensor::Error as TensorError;

#[test]
fn local_error_preserves_existing_invalid_argument_text() {
    let result: Result<()> = Err(Error::InvalidArgument("bad labels".into()));

    let Err(err) = result else {
        panic!("expected invalid argument error");
    };

    assert_eq!(err.to_string(), "invalid argument: bad labels");
}

#[test]
fn local_error_maps_to_tensor_backend_failure() {
    let err = Error::ShapeMismatch {
        expected: vec![2, 3],
        got: vec![2, 4],
    };

    let tensor_err = err.to_tensor_error("einsum_extension");

    assert!(matches!(
        tensor_err,
        TensorError::BackendFailure {
            op: "einsum_extension",
            ref message,
        } if message == "shape mismatch: expected [2, 3], got [2, 4]"
    ));
}
