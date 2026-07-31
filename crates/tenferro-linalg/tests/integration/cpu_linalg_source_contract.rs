use std::{fs, path::Path};

fn lapack_production_sources() -> Vec<(String, String)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("cpu")
        .join("linalg")
        .join("lapack_linalg");
    let mut sources = Vec::new();
    for entry in fs::read_dir(&root)
        .unwrap_or_else(|err| panic!("LAPACK source directory should be readable: {err}"))
    {
        let path = entry
            .unwrap_or_else(|err| panic!("LAPACK source entry should be readable: {err}"))
            .path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_else(|| panic!("LAPACK source path should have a UTF-8 file name"))
            .to_owned();
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("LAPACK source {name} should be readable: {err}"));
        sources.push((name, source));
    }
    sources
}

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

fn cpu_lapack_eig_source() -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("cpu")
            .join("linalg")
            .join("lapack_linalg")
            .join("eig.rs"),
    )
    .unwrap_or_else(|err| panic!("LAPACK eig source should be readable: {err}"))
}

fn cpu_lapack_source(path: &str) -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("cpu")
            .join("linalg")
            .join("lapack_linalg")
            .join(path),
    )
    .unwrap_or_else(|err| panic!("LAPACK source {path} should be readable: {err}"))
}

fn cpu_faer_linalg_source() -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("cpu")
            .join("linalg")
            .join("faer_linalg.rs"),
    )
    .unwrap_or_else(|err| panic!("faer linalg source should be readable: {err}"))
}

fn linalg_extension_source() -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("extension.rs"),
    )
    .unwrap_or_else(|err| panic!("linalg extension source should be readable: {err}"))
}

#[test]
fn cpu_linalg_allocation_helpers_remain_fallible_and_checked() {
    let faer = cpu_faer_linalg_source();
    let lapack = cpu_lapack_helpers_source();

    for source in [&faer, &lapack] {
        let lines: Vec<_> = source.lines().collect();
        for (index, line) in lines.iter().enumerate() {
            if line.contains("rows * cols") || line.contains(".expect(") {
                let invariant = index > 0 && lines[index - 1].contains("// INVARIANT:");
                assert!(
                    invariant,
                    "linalg helpers must propagate errors and use checked allocation sizes unless the immediately preceding line documents an INVARIANT"
                );
            }
        }
    }
}

#[test]
fn lapack_shape_derived_allocation_lengths_remain_checked() {
    let mut violations = Vec::new();
    for (name, source) in lapack_production_sources() {
        let lines: Vec<_> = source.lines().collect();
        for (index, line) in lines.iter().enumerate() {
            let allocation_product = (line.contains("vec![")
                || line.contains("with_capacity(")
                || line.contains("pool_acquire("))
                && line.contains(" * ");
            let unchecked_workspace_formula = line.contains("return 5 * mn * mn")
                || line.contains("(5 * mn * mn")
                || line.contains("8 * min_dim")
                || line.contains("2 * min_dim * min_dim")
                || line.contains("5 * min_dim * min_dim");
            if allocation_product || unchecked_workspace_formula {
                let invariant = index > 0 && lines[index - 1].contains("// INVARIANT:");
                if !invariant {
                    violations.push(format!("{name}:{}: {}", index + 1, line.trim()));
                }
            }
        }
    }
    assert!(
        violations.is_empty(),
        "LAPACK allocation lengths require checked arithmetic:\n{}",
        violations.join("\n")
    );
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

fn cpu_crate_source(path: &str) -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("tenferro-cpu")
            .join("src")
            .join(path),
    )
    .unwrap_or_else(|err| panic!("tenferro-cpu source {path} should be readable: {err}"))
}

fn linalg_backend_trait_source() -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("backend.rs"),
    )
    .unwrap_or_else(|err| panic!("linalg backend trait source should be readable: {err}"))
}

fn assert_unsafe_blocks_have_safety_comments(path: &str, source: &str) {
    let lines: Vec<_> = source.lines().collect();
    let mut missing = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        if !line.contains("unsafe {") {
            continue;
        }
        let window_start = idx.saturating_sub(3);
        let has_safety = lines[window_start..idx]
            .iter()
            .any(|candidate| candidate.trim_start().starts_with("// SAFETY:"));
        if !has_safety {
            missing.push(format!("{}:{}", path, idx + 1));
        }
    }
    assert!(
        missing.is_empty(),
        "LAPACK unsafe blocks need local SAFETY comments:\n{}",
        missing.join("\n")
    );
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

fn rust_function_section<'a>(source: &'a str, name: &str) -> &'a str {
    let signature = format!("fn {name}");
    let code = rust_code_mask(source);
    let start = code
        .windows(signature.len())
        .position(|window| window == signature.as_bytes())
        .unwrap_or_else(|| panic!("source should contain function {name}"));
    let body_start = code[start..]
        .iter()
        .position(|&byte| byte == b'{')
        .map(|offset| start + offset)
        .unwrap_or_else(|| panic!("function {name} should have a body"));
    let mut depth = 0usize;
    for (offset, byte) in code[body_start..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..=body_start + offset];
                }
            }
            _ => {}
        }
    }
    panic!("function {name} should have balanced braces")
}

fn rust_code_mask(source: &str) -> Vec<u8> {
    let bytes = source.as_bytes();
    let mut code = bytes.to_vec();
    let mut index = 0usize;
    while index < bytes.len() {
        let end = if bytes[index..].starts_with(b"//") {
            Some(
                bytes[index..]
                    .iter()
                    .position(|&byte| byte == b'\n')
                    .map_or(bytes.len(), |offset| index + offset),
            )
        } else if bytes[index..].starts_with(b"/*") {
            Some(block_comment_end(bytes, index))
        } else if bytes[index] == b'r' {
            raw_string_end(bytes, index)
        } else if bytes[index] == b'"' {
            Some(quoted_string_end(bytes, index))
        } else if bytes[index] == b'\'' {
            char_literal_end(bytes, index)
        } else {
            None
        };

        if let Some(end) = end {
            code[index..end].fill(b' ');
            index = end;
        } else {
            index += 1;
        }
    }
    code
}

fn block_comment_end(bytes: &[u8], start: usize) -> usize {
    let mut index = start + 2;
    let mut depth = 1usize;
    while index < bytes.len() && depth > 0 {
        if bytes[index..].starts_with(b"/*") {
            depth += 1;
            index += 2;
        } else if bytes[index..].starts_with(b"*/") {
            depth -= 1;
            index += 2;
        } else {
            index += 1;
        }
    }
    index
}

fn raw_string_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut quote = start + 1;
    while bytes.get(quote) == Some(&b'#') {
        quote += 1;
    }
    if bytes.get(quote) != Some(&b'"') {
        return None;
    }
    let hashes = quote - start - 1;
    let mut index = quote + 1;
    while index < bytes.len() {
        if bytes[index] == b'"'
            && bytes
                .get(index + 1..index + 1 + hashes)
                .is_some_and(|suffix| suffix.iter().all(|&byte| byte == b'#'))
        {
            return Some(index + 1 + hashes);
        }
        index += 1;
    }
    Some(bytes.len())
}

fn quoted_string_end(bytes: &[u8], start: usize) -> usize {
    let mut index = start + 1;
    while index < bytes.len() {
        match bytes[index] {
            b'\\' => index = (index + 2).min(bytes.len()),
            b'"' => return index + 1,
            _ => index += 1,
        }
    }
    bytes.len()
}

fn char_literal_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut index = start + 1;
    match *bytes.get(index)? {
        b'\\' => {
            index += 1;
            if bytes.get(index) == Some(&b'u') && bytes.get(index + 1) == Some(&b'{') {
                index += 2;
                index += bytes[index..].iter().position(|&byte| byte == b'}')? + 1;
            } else if bytes.get(index) == Some(&b'x') {
                index += 3;
            } else {
                index += 1;
            }
        }
        byte if byte.is_ascii() => index += 1,
        byte => index += utf8_char_width(byte)?,
    }
    (bytes.get(index) == Some(&b'\'')).then_some(index + 1)
}

fn utf8_char_width(first: u8) -> Option<usize> {
    match first {
        0xC2..=0xDF => Some(2),
        0xE0..=0xEF => Some(3),
        0xF0..=0xF4 => Some(4),
        _ => None,
    }
}

#[test]
fn rust_function_section_ignores_braces_in_comments_and_literals() {
    let source = r###"
fn target<'a>(value: &'a str) {
    let _ = "a string with } and an escaped quote: \" {";
    let _ = r#"a raw string with } { and \" quotes"#;
    let _ = '}';
    let _ = '\u{7b}';
    // a line comment with }
    /* an outer { comment /* with nested } comment */ still open } */
    let _ = value;
}

fn after() { panic!("must not be included"); }
"###;

    let section = rust_function_section(source, "target");
    assert!(section.contains("let _ = value;"));
    assert!(!section.contains("fn after"));
}

#[test]
fn linalg_rustdoc_reuses_backend_in_read_examples() {
    let source = linalg_backend_trait_source();

    assert!(
        !source.contains("CpuBackend::new()."),
        "LinalgBackend rustdoc examples should bind and reuse CpuBackend instead of constructing it inline"
    );
}

#[test]
fn cpu_backend_overrides_solve_read_into_hook() {
    let backend = cpu_backend_source();
    let override_section = rust_function_section(&backend, "solve_read_into");
    assert!(
        override_section.contains("validate_solve_read_into")
            && override_section.contains("with_linalg_pool")
            && override_section.contains("solve_read_into_entered"),
        "CpuBackend must override solve_read_into and inject its entered pool/context"
    );

    let trait_source = linalg_backend_trait_source();
    let default_section = rust_function_section(&trait_source, "solve_read_into");
    assert!(
        default_section.contains("solve_read_into_default(self, a, b, out)"),
        "LinalgBackend must retain the portable solve_read_into fallback"
    );
}

#[test]
fn public_cpu_linalg_read_methods_keep_one_operation_entry() {
    let source = cpu_backend_source();
    let methods = [
        ("triangular_solve_read", "triangular_solve_entered"),
        ("solve_read", "solve_entered"),
        ("svd_read", "svd_entered"),
        ("qr_read", "qr_entered"),
        ("eigh_read", "eigh_entered"),
        ("cholesky_read", "cholesky_entered"),
        ("lu_read", "lu_entered"),
        ("full_piv_lu_read", "full_piv_lu_entered"),
        ("eig_read", "eig_entered"),
    ];
    for (read, entered) in methods {
        let section = rust_function_section(&source, read);
        assert_eq!(
            section.matches("self.with_linalg_pool_fresh(").count(),
            1,
            "{read} must enter CPU resources exactly once"
        );
        assert!(
            section.contains(entered),
            "{read} must delegate to already-entered helper {entered}"
        );
        assert!(
            section.contains("context.with_materialized_tensor_read("),
            "{read} must canonicalize through its entered execution-context proof"
        );
        assert!(
            !section.contains(".to_contiguous"),
            "{read} must not recursively enter through public to_contiguous"
        );
        let owned = read.trim_end_matches("_read");
        assert!(
            !section.contains(&format!("self.{owned}(")),
            "{read} must not recursively enter through its owned operation"
        );

        let entered_section = rust_function_section(&source, entered);
        for forbidden in ["with_linalg_pool", ".to_contiguous", "self."] {
            assert!(
                !entered_section.contains(forbidden),
                "already-entered helper {entered} must not contain {forbidden}"
            );
        }
    }

    for read in ["triangular_solve_read", "solve_read"] {
        assert_eq!(
            rust_function_section(&source, read)
                .matches("context.with_materialized_tensor_read(")
                .count(),
            2,
            "{read} must nest scoped materialization so failure of the second input reclaims the first"
        );
    }

    let cholesky = rust_function_section(&source, "cholesky_read");
    let entry = cholesky
        .find("self.with_linalg_pool_fresh(")
        .expect("cholesky_read must have one operation entry");
    let managed_dispatch = cholesky
        .find("tensor_uses_backend_storage")
        .expect("cholesky_read must retain managed-storage dispatch");
    assert!(
        entry < managed_dispatch,
        "cholesky_read must enter CPU resources before choosing the managed-storage path"
    );
    let before_entry = &cholesky[..entry];
    assert!(
        !before_entry.contains("return ") && !before_entry.contains('?'),
        "cholesky_read must not retain a fallible or early-return path before its sole operation entry"
    );
    let managed = rust_function_section(&source, "managed_cholesky");
    assert!(
        managed.contains("context: &CpuExecutionContext")
            && managed.contains("buffers: &mut BufferPool")
            && !managed.contains("with_linalg_pool")
            && !managed.contains("&mut CpuBackend"),
        "managed Cholesky must consume an already-entered context and pool without recursively entering"
    );
    let managed_typed = rust_function_section(&source, "managed_cholesky_typed");
    for required in [
        "let values = if n == 0",
        "T::factor(context, buffers",
        "domain.allocate(",
        "write_managed_cholesky_output(",
    ] {
        assert!(
            managed_typed.contains(required),
            "managed Cholesky must keep zero-size, provider, allocation, and output adaptation inside its entered helper: missing {required}"
        );
    }

    let solve = rust_function_section(&source, "solve_entered");
    assert!(
        solve.contains("context.reshape_tensor("),
        "vector RHS solve must use its already-entered metadata-only reshape"
    );
    assert!(
        !solve.contains("with_linalg_pool") && !solve.contains("self.reshape("),
        "solve_entered must not reacquire resources while reshaping vector RHS"
    );
}

#[test]
fn cpu_linalg_materialization_requires_an_entered_context_token() {
    let cpu_lib = cpu_crate_source("lib.rs");
    let interop = source_section(
        &cpu_lib,
        "pub mod linalg_interop",
        "pub(crate) fn cpu_backend_buffer_error",
    );
    assert!(
        !interop.contains("pub fn materialize_tensor_read")
            && !interop.contains("pub fn reshape_tensor"),
        "linalg interop must not expose free materialize/reshape entry bypasses"
    );
    assert!(
        cpu_lib.contains("pub(crate) fn materialize_tensor_read"),
        "the raw materializer must remain crate-private"
    );

    let provider = cpu_crate_source("provider.rs");
    let scoped = rust_function_section(&provider, "with_materialized_tensor_read");
    assert!(
        scoped.contains("&self")
            && scoped.contains("crate::materialize_tensor_read")
            && scoped.contains("reclaim_tensor(buffers, materialized)"),
        "entered context receiver must own raw materialization and ordinary-path reclaim"
    );
    assert!(
        rust_function_section(&provider, "reshape_tensor").contains("&self"),
        "metadata-only reshape must also require the entered context receiver"
    );
    let entered = rust_function_section(&provider, "entered");
    assert!(
        !entered.trim_start().starts_with("pub "),
        "external crates must not be able to construct an entered execution-context token"
    );
}

#[test]
fn lapack_ffi_unsafe_blocks_document_safety_invariants() {
    for path in [
        "cholesky.rs",
        "eig.rs",
        "eigh.rs",
        "full_piv_lu.rs",
        "lu.rs",
        "qr.rs",
        "solve.rs",
        "svd.rs",
        "triangular_solve.rs",
    ] {
        assert_unsafe_blocks_have_safety_comments(path, &cpu_lapack_source(path));
    }
}

#[test]
fn lapack_right_triangular_solve_uses_right_side_trsm_without_physical_transposes() {
    let source = cpu_lapack_source("triangular_solve.rs");
    let solve_right = source_section(&source, "fn solve_right", "fn triangular_solve_2d");

    assert!(
        solve_right.contains("T::trsm("),
        "right-side triangular_solve should call BLAS TRSM directly"
    );
    assert!(
        solve_right.contains("CblasRight"),
        "right-side triangular_solve should use BLAS side=Right"
    );
    assert!(
        !solve_right.contains("transpose_col_major_data("),
        "right-side triangular_solve should not physically transpose RHS data"
    );
    assert!(
        !solve_right.contains("T::trtrs("),
        "right-side triangular_solve should not emulate side=Right through LAPACK TRTRS"
    );
    assert!(
        solve_right.contains("validate_non_unit_diagonal"),
        "right-side TRSM path should preserve non-unit singular checks before calling BLAS"
    );
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
        factor.contains("crate::Error::Singular"),
        "positive getc2 info should be reported through the typed singular source"
    );
}

#[test]
fn cpu_eig_real_complex_classification_uses_tolerance() {
    for (name, source) in [
        ("LAPACK eig", cpu_lapack_eig_source()),
        ("faer eig", cpu_faer_linalg_source()),
    ] {
        assert!(
            source.contains("eig_imag_is_effectively_zero"),
            "{name} should classify real-vs-complex eigenvalue pairs with a tolerance helper"
        );
        assert!(
            !source.contains("s_im[col] == 0.0") && !source.contains("s_im[j] == 0.0"),
            "{name} should not compare eigenvalue imaginary parts to exact zero"
        );
    }
}

#[test]
fn real_eig_complex_pair_conversion_guards_unpaired_last_column() {
    for (name, source, loop_var) in [
        ("LAPACK eig", cpu_lapack_eig_source(), "col"),
        ("faer eig", cpu_faer_linalg_source(), "j"),
    ] {
        let converter = source_section(
            &source,
            "macro_rules! impl_real_eig_to_complex_outputs",
            "macro_rules! impl_real_eig_to_complex_values",
        );
        assert!(
            converter.contains(&format!("if {loop_var} + 1 >= n")),
            "{name} real eig conversion should guard an apparent complex pair at the final column"
        );
    }
}

#[test]
fn faer_complex_slice_casts_assert_field_offsets() {
    let source = cpu_faer_linalg_source();
    let casts = source_section(
        &source,
        "macro_rules! impl_complex_faer_casts",
        "impl_complex_faer_casts!(",
    );

    for needle in [
        "std::mem::offset_of!($complex, re)",
        "std::mem::offset_of!($faer_complex, re)",
        "std::mem::offset_of!($complex, im)",
        "std::mem::offset_of!($faer_complex, im)",
    ] {
        assert!(
            casts.contains(needle),
            "faer complex slice casts must assert field-order compatibility: missing {needle}"
        );
    }
}

#[test]
fn linalg_batched_helpers_use_checked_products_and_slice_ranges() {
    let lapack_helpers = cpu_lapack_helpers_source();
    let batched_helpers = source_section(
        &lapack_helpers,
        "pub(crate) fn batched_single",
        "pub(crate) fn zero_dim_eig_outputs",
    );
    assert!(
        !batched_helpers.contains(".iter().product"),
        "LAPACK batched helpers must use checked shape products"
    );
    for needle in [
        "batch_idx * slice_size",
        "batch_idx * a_slice_size",
        "batch_idx * b_slice_size",
    ] {
        assert!(
            !batched_helpers.contains(needle),
            "LAPACK batched helpers must use checked slice ranges instead of {needle}"
        );
    }
    assert!(
        batched_helpers.contains("checked_product(")
            && batched_helpers.contains("checked_slice_range("),
        "LAPACK batched helpers should route products and batch ranges through checked helpers"
    );

    let faer_source = cpu_faer_linalg_source();
    let batch_count = source_section(&faer_source, "fn batch_count", "fn checked_repeated_len");
    assert!(
        !batch_count.contains(".iter().product"),
        "faer batch_count must use checked products"
    );
    let lu_factor = source_section(
        &faer_source,
        "pub(crate) fn lu_factor<T: FaerLinalg>",
        "pub(crate) fn full_piv_lu<T: FaerLinalg>",
    );
    assert!(
        !lu_factor.contains("batch * matrix_len") && !lu_factor.contains("start + matrix_len"),
        "faer LU factor batching must use checked_slice_range for batch windows"
    );
}

#[test]
fn canonical_svd_gauge_uses_checked_layout_without_raw_batch_offsets() {
    let source = linalg_extension_source();
    let gauge = source_section(
        &source,
        "fn apply_canonical_pivot_svd_gauge",
        "fn require_matrix_meta",
    );

    for needle in [
        ".iter().product::<usize>()",
        "batch * m * k",
        "batch * k * n",
    ] {
        assert!(
            !gauge.contains(needle),
            "canonical SVD gauge must use its checked layout instead of {needle}"
        );
    }
    assert!(
        gauge.contains("canonical_svd_gauge_layout(") && gauge.contains("validate_storage("),
        "canonical SVD gauge should prepare and validate one checked layout"
    );
}

#[test]
fn cpu_eigh_complex_output_adapters_are_fallible() {
    let source = cpu_backend_source();
    let eigh_impl = rust_function_section(&source, "eigh_entered");
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
