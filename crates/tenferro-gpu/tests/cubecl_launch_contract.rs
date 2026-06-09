use std::{fs, path::Path};

fn cubecl_source(file: &str) -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("cubecl")
            .join(file),
    )
    .unwrap_or_else(|err| panic!("CubeCL source {file} should be readable: {err}"))
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
fn cubecl_scatter_does_not_use_single_thread_launch_fallback() {
    let mod_source = cubecl_source("mod.rs");
    let scatter_source = source_section(&mod_source, "    fn scatter(", "    fn slice(");
    let dispatch_source = cubecl_source("dispatch.rs");
    let sources = [
        ("cubecl/mod.rs scatter body", scatter_source),
        ("cubecl/dispatch.rs", dispatch_source.as_str()),
    ];
    let banned = ["single_thread_launch_config", "CubeCount::new_single()"];

    let mut violations = Vec::new();
    for (name, source) in sources {
        for needle in banned {
            if source.contains(needle) {
                violations.push(format!("{name} contains {needle}"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "CubeCL scatter launch must not use a single-thread fallback:\n{}",
        violations.join("\n")
    );
}

#[test]
fn cubecl_zero_length_launches_validate_buffers_before_returning() {
    let dispatch_source = cubecl_source("dispatch.rs");
    let dispatch_contracts = [
        (
            "launch_unary",
            "pub(crate) fn launch_unary<",
            "pub(crate) fn launch_unary_tensor<",
            vec![
                "ensure_resident_on_runtime(rt, input, op)?;",
                "let input_arg = typed_tensor_array_arg(input, op)?;",
                "if len == 0",
            ],
        ),
        (
            "launch_unary_tensor",
            "pub(crate) fn launch_unary_tensor<",
            "pub(crate) fn launch_nullary_into<",
            vec![
                "ensure_resident_on_runtime(rt, input, op)?;",
                "let input_arg = typed_tensor_binding(input, op)?;",
                "if len == 0",
            ],
        ),
        (
            "launch_nullary_into",
            "pub(crate) fn launch_nullary_into<",
            "pub(crate) fn launch_unary_tensor_into<",
            vec![
                "ensure_resident_on_runtime(rt, output, op)?;",
                "let output_arg = typed_tensor_array_arg(output, op)?;",
                "if output.n_elements() == 0",
            ],
        ),
        (
            "launch_unary_tensor_into",
            "pub(crate) fn launch_unary_tensor_into<",
            "pub(crate) fn launch_binary<",
            vec![
                "ensure_resident_on_runtime(rt, output, op)?;",
                "ensure_resident_on_runtime(rt, input, op)?;",
                "let output_arg = typed_tensor_binding(output, op)?;",
                "let input_arg = typed_tensor_binding(input, op)?;",
                "if output.n_elements() == 0",
            ],
        ),
        (
            "launch_binary",
            "pub(crate) fn launch_binary<",
            "pub(crate) fn launch_compare_bool<",
            vec![
                "ensure_resident_on_runtime(rt, lhs, op)?;",
                "ensure_resident_on_runtime(rt, rhs, op)?;",
                "let lhs_arg = typed_tensor_array_arg(lhs, op)?;",
                "let rhs_arg = typed_tensor_array_arg(rhs, op)?;",
                "if len == 0",
            ],
        ),
        (
            "launch_compare_bool",
            "pub(crate) fn launch_compare_bool<",
            "pub(crate) fn launch_binary_tensor<",
            vec![
                "ensure_resident_on_runtime(rt, lhs, op)?;",
                "ensure_resident_on_runtime(rt, rhs, op)?;",
                "let lhs_arg = typed_tensor_array_arg(lhs, op)?;",
                "let rhs_arg = typed_tensor_array_arg(rhs, op)?;",
                "if len == 0",
            ],
        ),
        (
            "launch_binary_tensor",
            "pub(crate) fn launch_binary_tensor<",
            "pub(crate) fn launch_select_bool<",
            vec![
                "ensure_resident_on_runtime(rt, lhs, op)?;",
                "ensure_resident_on_runtime(rt, rhs, op)?;",
                "let lhs_arg = typed_tensor_binding(lhs, op)?;",
                "let rhs_arg = typed_tensor_binding(rhs, op)?;",
                "if len == 0",
            ],
        ),
        (
            "launch_select_bool",
            "pub(crate) fn launch_select_bool<",
            "pub(crate) fn launch_ternary<",
            vec![
                "ensure_resident_on_runtime(rt, pred, op)?;",
                "ensure_resident_on_runtime(rt, on_true, op)?;",
                "ensure_resident_on_runtime(rt, on_false, op)?;",
                "let pred_arg = bool_tensor_array_arg(pred, op)?;",
                "let true_arg = typed_tensor_array_arg(on_true, op)?;",
                "let false_arg = typed_tensor_array_arg(on_false, op)?;",
                "if len == 0",
            ],
        ),
        (
            "launch_ternary",
            "pub(crate) fn launch_ternary<",
            "pub(crate) fn dtype_mismatch(",
            vec![
                "ensure_resident_on_runtime(rt, a, op)?;",
                "ensure_resident_on_runtime(rt, b, op)?;",
                "ensure_resident_on_runtime(rt, c, op)?;",
                "let a_arg = typed_tensor_array_arg(a, op)?;",
                "let b_arg = typed_tensor_array_arg(b, op)?;",
                "let c_arg = typed_tensor_array_arg(c, op)?;",
                "if len == 0",
            ],
        ),
    ];
    for (name, start, end, needles) in dispatch_contracts {
        let section = source_section(&dispatch_source, start, end);
        assert_ordered_needles(name, section, &needles);
    }

    let fusion_source = cubecl_source("fusion/launch.rs");
    assert_ordered_needles(
        "fusion::launch",
        &fusion_source,
        &[
            "ensure_resident_on_runtime(runtime, input, \"fused_elementwise\")?;",
            "typed_tensor_array_arg(input, \"fused_elementwise\")?;",
            "typed_tensor_array_arg(output, \"fused_elementwise\")?;",
            "if classified.n_elements == 0",
        ],
    );
}

#[test]
fn cubecl_i64_index_conversion_does_not_roundtrip_through_host() {
    let mod_source = cubecl_source("mod.rs");
    let banned = [
        "fn i64_indices_as_f64",
        "download_tensor(self.runtime(), &Tensor::I64",
        "upload_tensor(self.runtime(), &converted",
    ];

    let mut violations = Vec::new();
    for needle in banned {
        if mod_source.contains(needle) {
            violations.push(format!("cubecl/mod.rs contains {needle}"));
        }
    }

    assert!(
        violations.is_empty(),
        "CubeCL I64 index conversion must stay on device; host roundtrips in indexing paths are performance regressions:\n{}",
        violations.join("\n")
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cubecl_runtime_exposes_explicit_synchronize() {
    let _sync: fn(&tenferro_gpu::CubeclRuntime) -> tenferro_tensor::Result<()> =
        tenferro_gpu::CubeclRuntime::synchronize;
}
