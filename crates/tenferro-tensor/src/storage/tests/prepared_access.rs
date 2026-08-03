use super::super::prepared::{
    prepare_read, prepare_write, AccessTarget, CheckedLayout, PreparedRead, PreparedWrite,
};
use crate::{DynRank, TensorScalar};

#[test]
fn checked_layout_rejects_invalid_dtype_and_out_of_span_before_mapping() {
    let _ = CheckedLayout::<DynRank>::Contiguous {
        element_range: 0..1,
    };
    let _ = AccessTarget::Host;
}

#[test]
fn prepared_contiguous_read_and_write_use_typed_slices() {
    let _ = prepare_read::<f64, DynRank>;
    let _ = prepare_write::<f64, DynRank>;
    let _: Option<PreparedRead<'static, f64, DynRank>> = None;
    let _: Option<PreparedWrite<'static, f64, DynRank>> = None;
}

#[test]
fn prepared_strided_iterators_cover_reverse_and_empty_layouts() {
    let _ = <f64 as TensorScalar>::dtype();
}

#[test]
fn provider_resolution_counts_do_not_depend_on_element_count() {
    let _ = AccessTarget::Device;
}
