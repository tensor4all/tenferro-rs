//! tenferro-rs #2424: cuBLAS takes its extents, leading dimensions and batch count as
//! `i32`, so a value that does not fit must be refused instead of wrapped by an `as` cast.
//! These tests exercise the narrowing itself; they neither allocate nor launch anything.

use crate::cuda::vendor::narrow_i32;
use crate::gpu::GpuError;

#[test]
fn the_largest_representable_value_is_accepted() {
    assert_eq!(narrow_i32("gemm", "m", i64::from(i32::MAX)), Ok(i32::MAX));
    assert_eq!(narrow_i32("gemm", "lda", 1), Ok(1));
}

#[test]
fn one_past_the_bound_is_refused_by_name_and_value() {
    let too_large = i64::from(i32::MAX) + 1;
    for field in ["m", "n", "k", "lda", "ldb", "ldc", "batch count"] {
        let error = narrow_i32("gemm", field, too_large)
            .expect_err("a value above i32::MAX must be refused");
        let GpuError::Unsupported { op, detail } = error else {
            panic!("{field}: expected a typed Unsupported error");
        };
        assert_eq!(op, "gemm");
        assert!(
            detail.contains(&format!("{field} = {too_large}")),
            "{field}: the error must name the field and its value, got {detail}"
        );
    }
}

#[test]
fn a_negative_value_still_narrows() {
    // The positivity checks run first in `gemm`; the narrowing itself must not wrap a
    // negative value into a plausible size.
    assert_eq!(narrow_i32("gemm", "m", -1), Ok(-1));
    assert!(narrow_i32("gemm", "m", i64::from(i32::MIN) - 1).is_err());
}

/// The rank bound a cuTENSOR descriptor enforces (tenferro-rs #2424). `MAX_TENSOR_MODES` is
/// `i32::MAX`, so the assertion is on the relation the code depends on, not on an allocation.
#[test]
fn the_descriptor_rank_bound_fits_both_narrowing_types() {
    let modes = crate::cuda::cutensor::MAX_TENSOR_MODES;
    assert_eq!(u32::try_from(modes), Ok(i32::MAX as u32));
    assert_eq!(i32::try_from(modes), Ok(i32::MAX));
}
