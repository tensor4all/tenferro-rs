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
        !source.contains("tenferro_cpu::CpuEngine::serial()."),
        "LinalgBackend rustdoc examples should bind and reuse CpuEngine instead of constructing it inline"
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
        "CpuEngine must override solve_read_into and inject its entered pool/exec bundle"
    );

    let trait_source = linalg_backend_trait_source();
    let default_section = rust_function_section(&trait_source, "solve_read_into");
    assert!(
        default_section.contains("solve_read_into_default(self, a, b, out)"),
        "LinalgBackend must retain the portable solve_read_into fallback"
    );
}

#[test]
fn faer_eligible_reads_reach_the_view_path_before_any_materialization() {
    // Evidence for the borrowed-input contract: a read hook that faer can
    // consume as a strided `MatRef` must test eligibility and return through
    // the `*_faer_view_entered` adapter *before* it reaches
    // `with_materialized_tensor_read`, so an eligible view never pays for a
    // front-end copy.
    let source = cpu_backend_source();
    for (read, view_entry) in [
        ("svd_read", "svd_faer_view_entered"),
        ("svd_full_read", "svd_full_faer_view_entered"),
        ("svd_values_read", "svd_values_faer_view_entered"),
        ("qr_read", "qr_faer_view_entered"),
        ("eigh_read", "eigh_faer_view_entered"),
        ("eig_read", "linalg::faer::eig_view"),
        ("eig_values_read", "linalg::faer::eig_values_view"),
        (
            "rank_revealing_qr_read",
            "rank_revealing_qr_faer_view_entered",
        ),
    ] {
        let section = rust_function_section(&source, read);
        let eligibility = section
            .find("faer_strided_read_ok(&input)")
            .unwrap_or_else(|| panic!("{read} must consult faer_strided_read_ok"));
        let view_return = section
            .find(view_entry)
            .unwrap_or_else(|| panic!("{read} must return through {view_entry}"));
        let materialization = section
            .find("tenferro_cpu::with_materialized_tensor_read(")
            .unwrap_or_else(|| panic!("{read} must keep a materializing fallback"));
        assert!(
            eligibility < view_return && view_return < materialization,
            "{read} must take the faer view path before materializing"
        );
    }

    // `triangular_solve_read` has two operands, so its eligibility test names
    // both and the RHS predicate differs: the right-hand side is gathered
    // element by element, so only host placement, rank and dtype matter there.
    let section = rust_function_section(&source, "triangular_solve_read");
    let eligibility = section
        .find("faer_strided_read_ok(&a)")
        .expect("triangular_solve_read must test the coefficient view");
    let rhs_eligibility = section
        .find("faer_rhs_read_ok(&b)")
        .expect("triangular_solve_read must test the right-hand side view");
    let view_return = section
        .find("triangular_solve_faer_view_entered")
        .expect("triangular_solve_read must return through its view adapter");
    let materialization = section
        .find("tenferro_cpu::with_materialized_tensor_read(")
        .expect("triangular_solve_read must keep a materializing fallback");
    assert!(
        eligibility < view_return && rhs_eligibility < view_return && view_return < materialization,
        "triangular_solve_read must take the faer view path before materializing"
    );
}

#[test]
fn public_cpu_linalg_read_methods_keep_one_operation_entry() {
    let source = cpu_backend_source();
    let methods = [
        ("triangular_solve_read", "triangular_solve_entered"),
        ("solve_read", "solve_entered"),
        ("svd_read", "svd_entered"),
        ("svd_full_read", "svd_full_entered"),
        ("qr_read", "qr_entered"),
        ("eigh_read", "eigh_entered"),
        ("cholesky_read", "cholesky_entered"),
        ("lu_read", "lu_entered"),
        ("full_piv_lu_read", "full_piv_lu_entered"),
        ("eig_read", "eig_entered"),
        ("eig_values_read", "eig_values_entered"),
    ];
    for (read, entered) in methods {
        let section = rust_function_section(&source, read);
        assert_eq!(
            section.matches("self.with_linalg_pool_checked(").count(),
            1,
            "{read} must enter CPU resources exactly once"
        );
        assert!(
            section.contains(entered),
            "{read} must delegate to already-entered helper {entered}"
        );
        assert!(
            section.contains("tenferro_cpu::with_materialized_tensor_read("),
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
                .matches("tenferro_cpu::with_materialized_tensor_read(")
                .count(),
            2,
            "{read} must nest scoped materialization so failure of the second input reclaims the first"
        );
    }

    let cholesky = rust_function_section(&source, "cholesky_read");
    let entry = cholesky
        .find("self.with_linalg_pool_checked(")
        .expect("cholesky_read must have one operation entry");
    let managed_dispatch = cholesky
        .find("input.backend_family().is_some()")
        .expect("cholesky_read must dispatch managed owned and borrowed descriptors");
    assert!(
        !cholesky.contains("input.as_tensor()"),
        "managed dispatch must not require an owned tensor"
    );
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
        managed.contains("exec: &mut Exec")
            && managed.contains("buffers: &mut BufferPool")
            && !managed.contains("with_linalg_pool")
            && !managed.contains("&mut CpuEngine"),
        "managed Cholesky must consume an already-entered exec bundle and pool without recursively entering"
    );
    let managed_typed = rust_function_section(&source, "managed_cholesky_typed");
    for required in [
        "let values = if n == 0",
        "T::factor(exec, buffers",
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
        solve.contains("tenferro_cpu::reshape_tensor(exec,"),
        "vector RHS solve must use its already-entered metadata-only reshape"
    );
    assert!(
        !solve.contains("with_linalg_pool") && !solve.contains("self.reshape("),
        "solve_entered must not reacquire resources while reshaping vector RHS"
    );
}

#[test]
fn cpu_linalg_materialization_requires_the_operations_exec_bundle() {
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
    let raw = rust_function_section(&cpu_lib, "materialize_tensor_read");
    assert!(
        raw.contains("exec: Exec<'_>"),
        "the raw materializer must take the placement token, never an ambient pool"
    );

    let provider = cpu_crate_source("provider.rs");
    let scoped = rust_function_section(&provider, "with_materialized_tensor_read");
    assert!(
        scoped.contains("exec: &mut Exec")
            && scoped.contains("crate::materialize_tensor_read")
            && scoped.contains("reclaim_tensor(buffers, materialized)"),
        "the scoped materializer must take the operation's exec bundle and own the ordinary-path reclaim"
    );
    assert!(
        rust_function_section(&provider, "reshape_tensor").contains("exec: &mut Exec"),
        "metadata-only reshape must also take the operation's exec bundle"
    );
}

#[test]
fn lapack_ffi_unsafe_blocks_document_safety_invariants() {
    for path in [
        "cholesky.rs",
        "eig.rs",
        "eigh.rs",
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
fn cpu_linalg_routes_hand_whole_batches_to_the_extracted_providers() {
    // The batch loop belongs to the extracted providers (`tlinalg`, `tlinalg-blas`): a host
    // route makes one provider call per batch and never slices the batch into per-matrix tensors.
    for (name, source) in lapack_production_sources()
        .into_iter()
        .chain(std::iter::once((
            "faer_linalg.rs".to_owned(),
            cpu_faer_linalg_source(),
        )))
    {
        for needle in [
            "fn batched_single",
            "fn batched_multi",
            "refill_tensor_from_slice",
            "for batch_idx in 0..",
            "host_data()?[range]",
        ] {
            assert!(
                !source.contains(needle),
                "{name} should delegate whole batches to its provider instead of `{needle}`"
            );
        }
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
fn linalg_host_shapes_use_checked_products() {
    for (name, source) in lapack_production_sources() {
        assert!(
            !source.contains(".iter().product"),
            "LAPACK host source {name} must use checked shape products"
        );
    }

    let faer_source = cpu_faer_linalg_source();
    let lu_factor = source_section(
        &faer_source,
        "pub(crate) fn lu_factor<T: FaerLinalg>",
        "fn full_piv_lu_impl",
    );
    assert!(
        !lu_factor.contains(".iter().product") && lu_factor.contains("checked_product("),
        "faer LU factor batch counts must use checked products"
    );
    assert!(
        !lu_factor.contains("batch * matrix_len") && !lu_factor.contains("start + matrix_len"),
        "faer LU factor must hand the whole batch to the provider instead of raw batch offsets"
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
