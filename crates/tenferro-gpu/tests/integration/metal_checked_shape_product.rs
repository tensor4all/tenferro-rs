//! Source contracts for the Metal backend's element counts (issue #2668).
//!
//! Both sites used a raw `shape.iter().product()`, so a wrapped count reached the
//! allocation check and the kernel arguments. These tests keep the checked forms in place.

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start = source
        .find(start)
        .unwrap_or_else(|| panic!("section start {start:?} should exist"));
    let tail = &source[start..];
    let end = tail
        .find(end)
        .unwrap_or_else(|| panic!("section end {end:?} should exist"));
    &tail[..end]
}

#[test]
fn metal_engine_len_uses_checked_shape_product() {
    let engine = include_str!("../../src/metal/engine.rs");
    let len = section(engine, "fn len(tensor: &Tensor", "fn submit(");

    assert!(
        len.contains("checked_shape_product(op, \"shape\", tensor.shape())?"),
        "MetalEngine::len must reject a shape-product overflow before the i64 conversion"
    );
    assert!(
        !len.contains("shape().iter().product"),
        "MetalEngine::len must not use an unchecked shape product"
    );
}

#[test]
fn metal_wrap_uses_checked_element_count_and_byte_length() {
    let context = include_str!("../../src/metal/context.rs");
    let wrap = section(context, "pub(crate) fn wrap<T>", "fn allocate_typed<T>");

    assert!(
        wrap.contains("Self::element_count(&shape, op)?"),
        "MetalContext::wrap must validate the element count through the checked helper"
    );
    assert!(
        wrap.contains(".checked_mul(std::mem::size_of::<T>())"),
        "MetalContext::wrap must compute the expected byte length with checked arithmetic"
    );
    assert!(
        !wrap.contains("shape.iter().product"),
        "MetalContext::wrap must not use an unchecked shape product"
    );
    assert!(
        !wrap.contains("saturating_mul(std::mem::size_of::<T>())"),
        "MetalContext::wrap must not paper over a byte-length overflow by saturating"
    );
}
