use super::compact_col_major_strides;
use tenferro_tensor::{ErrorKind, ValidationKind};

const OP: &str = "native_permutation_test";

#[test]
fn compact_col_major_strides_validate_shape_metadata_once() {
    assert_eq!(
        compact_col_major_strides(OP, &[]).unwrap(),
        Vec::<isize>::new()
    );
    assert_eq!(compact_col_major_strides(OP, &[2, 3]).unwrap(), vec![1, 2]);

    let dimension = usize::MAX;
    let error = compact_col_major_strides("compact_stride_test", &[dimension]).unwrap_err();
    assert_eq!(
        error.kind(),
        ErrorKind::Validation(ValidationKind::InvalidArgument)
    );
    assert!(matches!(
        error,
        crate::Error::Validation {
            op: "compact_stride_test",
            source: tenferro_tensor::ValidationError::InvalidArgument {
                argument: "shape",
                message,
            },
        } if message == format!("dimension {dimension} cannot be represented as isize")
    ));

    let shape = [isize::MAX as usize, 2];
    let error = compact_col_major_strides("compact_stride_test", &shape).unwrap_err();
    assert_eq!(
        error.kind(),
        ErrorKind::Validation(ValidationKind::InvalidArgument)
    );
    assert!(matches!(
        error,
        crate::Error::Validation {
            op: "compact_stride_test",
            source: tenferro_tensor::ValidationError::InvalidArgument {
                argument: "shape",
                message,
            },
        } if message == format!("column-major stride overflow for shape {shape:?}")
    ));
}
