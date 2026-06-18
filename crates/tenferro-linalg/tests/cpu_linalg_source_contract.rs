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

fn cpu_lapack_full_piv_lu_source() -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("cpu")
            .join("linalg")
            .join("lapack_linalg")
            .join("full_piv_lu.rs"),
    )
    .unwrap_or_else(|err| panic!("LAPACK full_piv_lu source should be readable: {err}"))
}

fn cpu_backend_source() -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("cpu")
            .join("backend.rs"),
    )
    .unwrap_or_else(|err| panic!("CPU linalg backend source should be readable: {err}"))
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
        "input.host_data().unwrap()[start..end].to_vec()",
        "a.host_data().unwrap()[a_start..a_end].to_vec()",
        "b.host_data().unwrap()[b_start..b_end].to_vec()",
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

#[test]
fn lapack_full_piv_lu_rejects_positive_getc2_info() {
    let source = cpu_lapack_full_piv_lu_source();
    let factor = source_section(&source, "fn factor_getc2", "fn full_piv_lu_2d");

    assert!(
        factor.contains("check_lapack_info(op, \"getc2\", info.min(0))?;"),
        "factor_getc2 should still report negative LAPACK argument errors"
    );
    assert!(
        factor.contains("if info > 0"),
        "factor_getc2 should not discard positive getc2 singularity info"
    );
    assert!(
        factor.contains("matrix is singular"),
        "positive getc2 info should be reported as a singular matrix"
    );
}

#[test]
fn cpu_eigh_complex_output_adapters_are_fallible() {
    let source = cpu_backend_source();
    let eigh_impl = source_section(&source, "fn eigh(&mut self", "fn eigh_values");
    let c32_adapter = source_section(
        &source,
        "fn eigh_c32_outputs_to_public_tensors",
        "fn eigh_c64_outputs_to_public_tensors",
    );
    let c64_adapter = source_section(
        &source,
        "fn eigh_c64_outputs_to_public_tensors",
        "fn apply_lu_pivots_cpu",
    );

    assert!(
        eigh_impl.contains(".and_then(eigh_c32_outputs_to_public_tensors)")
            && eigh_impl.contains(".and_then(eigh_c64_outputs_to_public_tensors)"),
        "CPU complex eigh output adapters should propagate malformed output errors"
    );
    for (name, adapter) in [
        ("eigh_c32_outputs_to_public_tensors", c32_adapter),
        ("eigh_c64_outputs_to_public_tensors", c64_adapter),
    ] {
        assert!(
            adapter.contains("tenferro_tensor::Result<Vec<Tensor>>"),
            "{name} should return a typed Result"
        );
        assert!(
            !adapter.contains(".expect("),
            "{name} should not panic on malformed backend output"
        );
    }
}
