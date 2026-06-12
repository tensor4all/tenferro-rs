use std::{fs, path::Path};

fn cpu_lapack_helpers_source() -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("cpu")
            .join("linalg")
            .join("lapack_linalg")
            .join("helpers.rs"),
    )
    .unwrap_or_else(|err| panic!("LAPACK helper source should be readable: {err}"))
}

fn source_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start_idx = source
        .find(start)
        .unwrap_or_else(|| panic!("source should contain section start {start:?}"));
    let remaining = &source[start_idx..];
    let end_idx = remaining
        .find(end)
        .map(|offset| start_idx + offset)
        .unwrap_or(source.len());
    &source[start_idx..end_idx]
}

#[test]
fn lapack_batched_helpers_reuse_input_scratch_instead_of_copying_per_batch() {
    let source = cpu_lapack_helpers_source();
    let batched_helpers = source_section(
        &source,
        "pub(crate) fn batched_single",
        "pub(crate) fn zero_dim_eig_outputs",
    );

    for needle in [
        "input.host_data()[start..end].to_vec()",
        "a.host_data()[a_start..a_end].to_vec()",
        "b.host_data()[b_start..b_end].to_vec()",
    ] {
        assert!(
            !batched_helpers.contains(needle),
            "LAPACK batched helpers should not allocate a fresh input Vec per batch: found {needle}"
        );
    }

    for needle in [
        "tensor_from_pooled_slice_with_template",
        "refill_tensor_from_slice",
    ] {
        assert!(
            batched_helpers.contains(needle),
            "LAPACK batched helpers should reuse pooled input scratch via {needle}"
        );
    }
}
