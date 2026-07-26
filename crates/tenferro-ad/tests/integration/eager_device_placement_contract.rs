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

    let helper = source_section(
        &eager_exec,
        "fn upload_generated_host_tensor",
        "fn exec_standard_op_on_tensor_reads",
    );
    assert!(helper.contains("backend.upload_host_tensor(&tensor)"));
    assert!(
        helper.contains("StdTensorOp::Constant { dtype, bytes } => constant_tensor(*dtype, bytes)")
    );
    assert!(helper.contains(
        "StdTensorOp::ShapeOf { axis } => shape_of_host_tensor(*axis, inputs[0].shape())?"
    ));

    let read_path = source_section(
        &eager_exec,
        "fn exec_standard_op_on_tensor_reads",
        "fn exec_standard_op_on_tensors",
    );
    assert_ordered_needles(
        "exec_standard_op_on_tensor_reads",
        read_path,
        &[
            "execute_generated_host_output_on_backend_reads(op, inputs, backend)?",
            ".with_backend_session",
        ],
    );

    let tensor_path = source_section(
        &eager_exec,
        "fn exec_standard_op_on_tensors",
        "pub(crate) fn exec_op_on_tensors_with_runtime",
    );
    assert_ordered_needles(
        "exec_standard_op_on_tensors",
        tensor_path,
        &[
            "execute_generated_host_output_on_backend_tensors(op, inputs, backend)?",
            ".with_backend_session",
        ],
    );
}

#[test]
fn eager_index_select_uploads_hidden_indices_before_importing_constant() {
    let shape_packing = ad_source("shape_packing.rs");
    let index_select = source_section(
        &shape_packing,
        "pub fn index_select(&self, axis: isize, positions: &[usize]) -> Result<Self>",
        "    /// Stack tensors along a newly inserted axis.",
    );

    assert_ordered_needles(
        "EagerTensor::index_select",
        index_select,
        &[
            "backend.upload_host_tensor(&indices)?",
            "self.ctx.constant_from(indices)?",
            "self.gather(&indices, config)",
        ],
    );
}

#[test]
fn eager_ad_seed_and_missing_tangent_zeroes_are_uploaded() {
    let eager = ad_source("eager.rs");
    let eager_zero_like = source_section(
        &eager,
        "pub(crate) fn zero_like_tensor<B: TensorBackend>",
        "pub(crate) fn one_like_tensor",
    );
    assert!(eager_zero_like.contains(".upload_host_tensor(&host)"));

    let eager_one_like = source_section(
        &eager,
        "pub(crate) fn one_like_tensor<B: TensorBackend>",
        "#[cfg(test)]",
    );
    assert!(eager_one_like.contains("ones_tensor(input.dtype(), input.shape().to_vec())"));
    assert!(eager_one_like.contains(".upload_host_tensor(&host)"));
    assert!(!eager_one_like.contains(".exp("));

    assert!(!eager.contains("tidu::"));
    assert!(!eager.contains("ShapeGuardContext::with_global_metadata()"));
}
