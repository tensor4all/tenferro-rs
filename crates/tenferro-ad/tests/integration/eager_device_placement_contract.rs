use std::{fs, path::Path};

fn ad_source(file: &str) -> String {
    fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src").join(file))
        .unwrap_or_else(|err| panic!("tenferro-ad source {file} should be readable: {err}"))
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

fn assert_ordered_needles(source_name: &str, source: &str, needles: &[&str]) {
    let mut offset = 0;
    for needle in needles {
        let remaining = &source[offset..];
        let found = remaining.find(needle).unwrap_or_else(|| {
            panic!("{source_name} should contain {needle:?} after byte offset {offset}")
        });
        offset += found + needle.len();
    }
}

#[test]
fn eager_generated_constant_and_shape_outputs_are_uploaded_before_backend_ops() {
    let eager_exec = ad_source("eager_exec.rs");

    let generated = source_section(
        &eager_exec,
        "fn generated_host_output(",
        "pub(crate) fn exec_standard_op_on_tensor_reads_with_session",
    );
    assert!(generated
        .contains("StdTensorOp::Constant { dtype, bytes } => constant_tensor(*dtype, bytes)"));
    assert!(generated.contains("shape_of_host_tensor(*axis, shape).map(Some)"));

    let read_session_path = source_section(
        &eager_exec,
        "pub(crate) fn exec_standard_op_on_tensor_reads_with_session",
        "pub(crate) fn exec_standard_op_on_tensors_with_session",
    );
    assert_ordered_needles(
        "exec_standard_op_on_tensor_reads_with_session",
        read_session_path,
        &[
            "generated_host_output(op, inputs.first().map(TensorRead::shape))?",
            ".upload_host_tensor(TensorRead::from_tensor(&host))",
            "exec_standard_op_on_tensor_reads_in_session(op, inputs, exec)",
        ],
    );

    let tensor_session_path = source_section(
        &eager_exec,
        "pub(crate) fn exec_standard_op_on_tensors_with_session",
        "pub(crate) fn exec_standard_op_on_tensor_reads_in_session",
    );
    assert_ordered_needles(
        "exec_standard_op_on_tensors_with_session",
        tensor_session_path,
        &[
            "generated_host_output(op, inputs.first().map(|tensor| tensor.shape()))?",
            ".upload_host_tensor(TensorRead::from_tensor(&host))",
            "exec_standard_op_on_tensors_in_session(op, inputs, exec)",
        ],
    );

    // No self-entering standard-op path remains (#1946 F9): standard ops run
    // only on a borrowed session; the owner-context fallback is extension-only.
    assert!(!eager_exec.contains("fn exec_standard_op_on_tensor_reads<B: BackendSessionHost>"));

    // The owned-tensor standard-op path is gone (#1946 F9): owned inputs run
    // through the borrowed session's `exec_standard_op_on_tensors_with_session`.
    assert!(!eager_exec.contains("fn exec_standard_op_on_tensors<B: BackendSessionHost>"));
}

#[test]
fn eager_index_select_imports_hidden_indices_through_borrowed_session() {
    let shape_packing = ad_source("shape_packing.rs");
    let index_select = source_section(
        &shape_packing,
        "pub fn index_select(",
        "    /// Select entries from an axis by host-known positions.",
    );
    assert_ordered_needles(
        "EagerSession::index_select",
        index_select,
        &[
            "self.ensure_runtime(tensor)?",
            "index_select_config(tensor.shape(), axis, positions)?",
            "self.constant_from_host(indices)?",
            "self.gather(tensor, &indices, config)",
        ],
    );
    let eager = ad_source("eager.rs");
    let upload = source_section(
        &eager,
        "pub fn constant_from_host(",
        "/// Import a trainable leaf",
    );
    assert!(upload.contains(".upload_host_tensor(TensorRead::from_tensor(&tensor))"));
    assert!(upload.contains("self.constant_from(uploaded)"));
    let leaf = source_section(&eager, "fn new_leaf_with_session(", "fn new_result(");
    assert_ordered_needles(
        "new_leaf_with_session",
        leaf,
        &["Some(session) => session.to_contiguous_read(read)"],
    );
}

#[test]
fn eager_ad_seed_and_missing_tangent_zeroes_are_uploaded() {
    let eager = ad_source("eager.rs");
    let eager_zero_like = source_section(
        &eager,
        "pub(crate) fn zero_like_tensor(",
        "pub(crate) fn one_like_tensor",
    );
    assert!(eager_zero_like.contains(".upload_host_tensor(TensorRead::from_tensor(&host))"));

    let eager_one_like = source_section(
        &eager,
        "pub(crate) fn one_like_tensor(input: &Tensor, session: &mut dyn Engine)",
        "#[cfg(test)]",
    );
    assert!(eager_one_like.contains("ones_tensor(input.dtype(), input.shape().to_vec())"));
    assert!(eager_one_like.contains(".upload_host_tensor(TensorRead::from_tensor(&host))"));
    assert!(!eager_one_like.contains(".exp("));

    assert!(!eager.contains("tidu::"));
    assert!(!eager.contains("ShapeGuardContext::with_global_metadata()"));
}
