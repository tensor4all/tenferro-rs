use std::{
    fs,
    path::{Path, PathBuf},
};

fn rust_sources_under(dir: &Path, sources: &mut Vec<(PathBuf, String)>) {
    for entry in fs::read_dir(dir).expect("source directory should be readable") {
        let path = entry.expect("source entry should be readable").path();
        if path.is_dir() {
            rust_sources_under(&path, sources);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            let source = fs::read_to_string(&path)
                .unwrap_or_else(|err| panic!("source {path:?} should be readable: {err}"));
            sources.push((path, source));
        }
    }
}

#[test]
fn bool_structural_support_is_copy_only_and_scatter_stays_excluded() {
    let source = std::fs::read_to_string("src/cubecl/mod.rs").unwrap();
    for needle in [
        "Tensor::Bool(t) => self.transpose_bool(t, perm).map(Tensor::Bool)",
        "Tensor::Bool(t) => self.broadcast_bool(t, shape, dims).map(Tensor::Bool)",
        "Tensor::Bool(t) => self.slice_bool(t, config).map(Tensor::Bool)",
        "Tensor::Bool(operand), Tensor::I64(indices)",
        "Tensor::Bool(input), Tensor::I64(starts)",
    ] {
        assert!(
            source.contains(needle),
            "missing Bool copy/index dispatch: {needle}"
        );
    }
    assert!(source.contains("Bool data tensors are not supported by additive scatter"));
    assert!(!source.contains("scatter_bool_typed"));
    assert!(!source.contains("(Tensor::Bool(input), Tensor::F32(starts))"));
    assert!(!source.contains("(Tensor::Bool(input), Tensor::F64(starts))"));
}

#[test]
fn explicit_cast_uses_shared_device_kernel_families_and_keeps_checked_convert() {
    let kernels = std::fs::read_to_string("src/kernels/structural.rs").unwrap();
    for family in [
        "pub fn convert_numeric<",
        "pub fn convert_numeric_to_bool<",
        "pub fn convert_bool_to_numeric<",
        "pub fn convert_numeric_to_complex_raw<",
        "pub fn convert_complex_to_numeric<",
        "pub fn validate_real_cast<",
    ] {
        assert!(
            kernels.contains(family),
            "missing cast kernel family: {family}"
        );
    }

    let backend = std::fs::read_to_string("../tenferro-tensor/src/backend.rs").unwrap();
    let convert = source_section(
        &backend,
        "fn convert(&mut self, input: &Tensor, to: crate::DType)",
        "fn cast(&mut self, input: &Tensor, to: crate::DType)",
    );
    assert!(convert.contains("validate_convert_dtype"));
    assert!(convert.contains("self.cast(input, to)"));

    let cuda = std::fs::read_to_string("src/cubecl/mod.rs").unwrap();
    let validation = source_section(
        &cuda,
        "fn validate_cuda_real_cast",
        "fn checked_integer_domain_error",
    );
    assert!(validation.find("if n == 0").unwrap() < validation.find("alloc_output::<F>").unwrap());
    assert!(!validation.contains("download_tensor(backend.runtime(), input"));
}

#[test]
fn bool_launch_domains_are_checked_before_output_allocation() {
    let dispatch = std::fs::read_to_string("src/cubecl/dispatch.rs").unwrap();
    for (start, end) in [
        (
            "pub(crate) fn launch_unary_bool_tensor(",
            "pub(crate) fn launch_binary_bool_tensor",
        ),
        (
            "pub(crate) fn launch_binary_bool_tensor",
            "pub(crate) fn launch_bool_tensor_into",
        ),
    ] {
        let body = source_section(&dispatch, start, end);
        let checked_len = body.find("checked_shape_product(op, out_shape)?").unwrap();
        let checked_count = body.find("cube_count_for_len(output_len)?").unwrap();
        let allocation = body.find("alloc_bool_output(rt, out_shape)?").unwrap();
        assert!(checked_len < checked_count && checked_count < allocation);
        assert!(body.contains("let Some(launch_count) = launch_count else"));
    }

    let backend = std::fs::read_to_string("src/cubecl/mod.rs").unwrap();
    let embed = source_section(&backend, "fn embed_diagonal_bool(", "pub fn tril_typed");
    assert!(embed.find("checked_dim_product").unwrap() < embed.find("alloc_bool_output").unwrap());
    assert!(
        embed.find("cube_count_for_len(output_len)?").unwrap()
            < embed.find("alloc_bool_output").unwrap()
    );
    assert!(
        embed
            .find("cube_count_for_len(input.n_elements())?")
            .unwrap()
            < embed.find("alloc_bool_output").unwrap()
    );

    let concatenate = source_section(&backend, "fn concatenate_bool(", "fn gather_typed");
    assert!(
        concatenate.find("checked_dim_product").unwrap()
            < concatenate.find("alloc_bool_output").unwrap()
    );
    assert!(
        concatenate.find("let launch_counts").unwrap()
            < concatenate.find("alloc_bool_output").unwrap()
    );
}

fn production_rust_sources() -> Vec<(PathBuf, String)> {
    let mut sources = Vec::new();
    rust_sources_under(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut sources,
    );
    sources
}

#[derive(Debug)]
struct Lexeme {
    text: String,
    offset: usize,
}

fn rust_code_lexemes(source: &str) -> Vec<Lexeme> {
    let bytes = source.as_bytes();
    let mut lexemes = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i..].starts_with(b"//") {
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if bytes[i..].starts_with(b"/*") {
            i += 2;
            let mut depth = 1;
            while i < bytes.len() && depth > 0 {
                if bytes[i..].starts_with(b"/*") {
                    depth += 1;
                    i += 2;
                } else if bytes[i..].starts_with(b"*/") {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            continue;
        }
        let raw = if bytes[i] == b'r' {
            Some(i + 1)
        } else if bytes[i..].starts_with(b"br") {
            Some(i + 2)
        } else {
            None
        };
        if let Some(mut cursor) = raw {
            let mut hashes = 0;
            while cursor < bytes.len() && bytes[cursor] == b'#' {
                hashes += 1;
                cursor += 1;
            }
            if cursor < bytes.len() && bytes[cursor] == b'"' {
                i = cursor + 1;
                while i < bytes.len() {
                    if bytes[i] == b'"' && bytes[i + 1..].starts_with(&vec![b'#'; hashes]) {
                        i += 1 + hashes;
                        break;
                    }
                    i += 1;
                }
                continue;
            }
        }
        let quote = if bytes[i] == b'"' {
            Some(i)
        } else if bytes[i..].starts_with(b"b\"") {
            Some(i + 1)
        } else {
            None
        };
        if let Some(quote) = quote {
            i = quote + 1;
            while i < bytes.len() {
                if bytes[i] == b'\\' {
                    i = (i + 2).min(bytes.len());
                } else if bytes[i] == b'"' {
                    i += 1;
                    break;
                } else {
                    i += 1;
                }
            }
            continue;
        }
        if bytes[i] == b'\'' {
            let mut end = i + 1;
            if end < bytes.len() && bytes[end] == b'\\' {
                end += 2;
            } else {
                end += 1;
            }
            if end < bytes.len() && bytes[end] == b'\'' {
                i = end + 1;
                continue;
            }
        }
        if bytes[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }
        let start = i;
        if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
            i += 1;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
        } else {
            i += if bytes[i..].starts_with(b"::") { 2 } else { 1 };
        }
        lexemes.push(Lexeme {
            text: source[start..i].to_owned(),
            offset: start,
        });
    }
    lexemes
}

fn lexeme_positions(lexemes: &[Lexeme], sequence: &[&str]) -> Vec<usize> {
    lexemes
        .windows(sequence.len())
        .filter(|window| {
            window
                .iter()
                .zip(sequence)
                .all(|(lexeme, expected)| lexeme.text == *expected)
        })
        .map(|window| window[0].offset)
        .collect()
}

fn contains_lexeme_subsequence(lexemes: &[Lexeme], sequence: &[&str]) -> bool {
    let mut next = 0;
    for lexeme in lexemes {
        if lexeme.text == sequence[next] {
            next += 1;
            if next == sequence.len() {
                return true;
            }
        }
    }
    false
}

fn forbidden_scatter_aliases(lexemes: &[Lexeme]) -> Vec<usize> {
    let protected = [
        "update_window_len",
        "scatter_float_kernel",
        "scatter_complex_kernel",
        "indexing",
    ];
    let mut violations = Vec::new();
    let mut index = 0;
    while index < lexemes.len() {
        if lexemes[index].text != "use" {
            index += 1;
            continue;
        }
        let end = lexemes[index..]
            .iter()
            .position(|lexeme| lexeme.text == ";")
            .map_or(lexemes.len(), |offset| index + offset);
        let statement = &lexemes[index..end];
        if statement.iter().any(|lexeme| lexeme.text == "as")
            && statement
                .iter()
                .any(|lexeme| protected.contains(&lexeme.text.as_str()))
        {
            violations.push(lexemes[index].offset);
        }
        index = end.saturating_add(1);
    }
    violations
}

fn scatter_update_window_contract(sources: &[(PathBuf, String)]) -> Result<(), String> {
    let invariants = [
        "    // INVARIANT: `scatter_update_len` returns the checked batch-window product, including zero;\n    // `scatter_float_typed` returns before launch when that checked length is zero.\n    let window_iters = update_window_len(updates, update_window_dims.clone());",
        "    // INVARIANT: `scatter_update_len` returns the checked batch-window product, including zero;\n    // `scatter_complex_typed` returns before launch when that checked length is zero.\n    let window_iters = update_window_len(updates, update_window_dims.clone());",
    ];
    let mut product_calls = Vec::new();
    let mut definitions = Vec::new();
    let mut launches = Vec::new();
    let mut aliases = Vec::new();
    for (path, source) in sources {
        let lexemes = rust_code_lexemes(source);
        for offset in forbidden_scatter_aliases(&lexemes) {
            aliases.push((path, offset));
        }
        let local_definitions: Vec<_> = lexemes
            .windows(2)
            .filter(|window| window[0].text == "fn" && window[1].text == "update_window_len")
            .map(|window| window[1].offset)
            .collect();
        definitions.extend(local_definitions.iter().copied());
        for offset in lexeme_positions(&lexemes, &["update_window_len", "("]) {
            if local_definitions.contains(&offset) {
                continue;
            }
            let line_start = source[..offset]
                .rfind('\n')
                .map_or(0, |newline| newline + 1);
            let before = source[..line_start].trim_end_matches('\n');
            let context_start = before
                .rmatch_indices('\n')
                .nth(1)
                .map_or(0, |(newline, _)| newline + 1);
            let call_end = source[offset..]
                .find(';')
                .map_or(source.len(), |end| offset + end + 1);
            let context = &source[context_start..call_end];
            product_calls.push((
                path,
                offset,
                invariants.iter().position(|item| context.contains(item)),
            ));
        }
        for sequence in [
            &["scatter_float_kernel", "::", "launch_unchecked"][..],
            &["scatter_complex_kernel", "::", "launch_unchecked"][..],
        ] {
            for offset in lexeme_positions(&lexemes, sequence) {
                launches.push((path, offset));
            }
        }
    }

    if !aliases.is_empty() {
        return Err(format!(
            "scatter update-window proof symbols must not be imported or re-exported with aliases: {aliases:?}"
        ));
    }

    if definitions.len() != 1 {
        return Err(format!(
            "expected one update_window_len definition, found {definitions:?}"
        ));
    }

    if product_calls.len() != 2 {
        return Err(format!(
            "expected exactly two update_window_len call sites, found {product_calls:?}"
        ));
    }
    let mut seen_invariants = Vec::new();
    for (path, line, invariant_index) in product_calls {
        let Some(invariant_index) = invariant_index else {
            return Err(format!(
                "{path:?}:{line} lacks the adjacent checked host invariant"
            ));
        };
        seen_invariants.push(invariant_index);
    }
    seen_invariants.sort_unstable();
    if seen_invariants != [0, 1] {
        return Err(format!(
            "expected one concrete invariant per kernel, found {seen_invariants:?}"
        ));
    }

    if launches.len() != 2 {
        return Err(format!(
            "expected exactly two scatter update kernel launches, found {launches:?}"
        ));
    }
    let mod_source = sources
        .iter()
        .find_map(|(path, source)| path.ends_with("cubecl/mod.rs").then_some(source.as_str()))
        .ok_or_else(|| "cubecl/mod.rs was not inventoried".to_owned())?;
    let meta_lexemes = rust_code_lexemes(source_section(
        mod_source,
        "fn scatter_launch_meta(",
        "impl TensorIndexing for CudaBackend",
    ));
    for required in [
        &[
            "ensure_axes_unique",
            "(",
            ",",
            ",",
            "&",
            "config",
            ".",
            "update_window_dims",
            ",",
            "updates_shape",
            ".",
            "len",
            "(",
            ")",
        ][..],
        &[
            "window_shape_updates",
            "=",
            "config",
            ".",
            "update_window_dims",
            ".",
            "iter",
            "(",
            ")",
            ".",
            "map",
            "(",
            "|",
            "&",
            "axis",
            "|",
            "updates_shape",
            "[",
            "axis",
            "]",
            ")",
            ".",
            "collect",
            "(",
            ")",
        ][..],
    ] {
        if lexeme_positions(&meta_lexemes, required).is_empty() {
            return Err(format!(
                "scatter_launch_meta lacks update-window derivation proof {required:?}"
            ));
        }
    }
    for (name, start, end, launch) in [
        (
            "scatter_float_typed",
            "    fn scatter_float_typed<",
            "    fn scatter_complex_typed<",
            "indexing::scatter_float_kernel::launch_unchecked",
        ),
        (
            "scatter_complex_typed",
            "    fn scatter_complex_typed<",
            "impl BackendRuntimeCache for CudaBackend",
            "indexing::scatter_complex_kernel::launch_unchecked",
        ),
    ] {
        let section = source_section(mod_source, start, end);
        let section_lexemes = rust_code_lexemes(section);
        let kernel = launch
            .strip_prefix("indexing::")
            .and_then(|launch| launch.strip_suffix("::launch_unchecked"))
            .expect("launch descriptor should be qualified");
        let required = [
            "scatter_launch_meta",
            "(",
            "let",
            "update_len",
            "=",
            "scatter_update_len",
            "(",
            "&",
            "meta",
            ")",
            "?",
            ";",
            "if",
            "update_len",
            "=",
            "=",
            "0",
            "{",
            "return",
            "Ok",
            "(",
            "output",
            ")",
            ";",
            "}",
            "indexing",
            "::",
            kernel,
            "::",
            "launch_unchecked",
        ];
        if !contains_lexeme_subsequence(&section_lexemes, &required) {
            return Err(format!("{name} lacks ordered checked launch proof"));
        }
    }
    Ok(())
}

fn cubecl_source(file: &str) -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("cubecl")
            .join(file),
    )
    .unwrap_or_else(|err| panic!("CubeCL source {file} should be readable: {err}"))
}

fn gpu_source(path: &[&str]) -> String {
    let mut full_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    for component in path {
        full_path = full_path.join(component);
    }
    fs::read_to_string(&full_path)
        .unwrap_or_else(|err| panic!("GPU source {full_path:?} should be readable: {err}"))
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

fn source_tail<'a>(source: &'a str, start: &str) -> &'a str {
    let start_idx = source
        .find(start)
        .unwrap_or_else(|| panic!("source should contain section start {start:?}"));
    &source[start_idx..]
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
fn cubecl_scatter_update_window_product_has_checked_host_invariant() {
    let sources = production_rust_sources();
    let mod_source = sources
        .iter()
        .find_map(|(path, source)| path.ends_with("cubecl/mod.rs").then_some(source.as_str()))
        .expect("cubecl/mod.rs should be inventoried");
    let update_len = source_section(
        mod_source,
        "fn scatter_update_len(",
        "/// CubeCL-based GPU backend.",
    );
    assert_ordered_needles(
        "scatter_update_len",
        update_len,
        &[
            "checked_dim_product(\"scatter\", \"batch shape\", &meta.batch_shape)?",
            "checked_dim_product(\"scatter\", \"window update shape\", &meta.window_shape_updates)?",
            "batch_len.checked_mul(window_len)",
        ],
    );

    scatter_update_window_contract(&sources).unwrap_or_else(|err| panic!("{err}"));
}

#[test]
fn rust_lexeme_inventory_ignores_literals_and_accepts_multiline_calls() {
    let source = r####"
        // update_window_len(fake)
        /* scatter_float_kernel::launch_unchecked(fake) */
        let normal = "update_window_len(fake)";
        let raw = r#"scatter_complex_kernel::launch_unchecked(fake)"#;
        let byte = b"update_window_len(fake)";
        let character = '(';
        update_window_len
            (
                updates,
                dims,
            );
        scatter_float_kernel
            ::
            launch_unchecked
            ();
    "####;
    let lexemes = rust_code_lexemes(source);
    assert_eq!(
        lexeme_positions(&lexemes, &["update_window_len", "("]).len(),
        1
    );
    assert_eq!(
        lexeme_positions(
            &lexemes,
            &["scatter_float_kernel", "::", "launch_unchecked", "("]
        )
        .len(),
        1
    );
    assert!(lexeme_positions(
        &lexemes,
        &["scatter_complex_kernel", "::", "launch_unchecked", "("]
    )
    .is_empty());
}

#[test]
fn scatter_update_window_inventory_rejects_unproved_new_paths() {
    let sources = production_rust_sources();

    let invariant = "// INVARIANT: `scatter_update_len` returns the checked batch-window product, including zero;\n    // `scatter_float_typed` returns before launch when that checked length is zero.\n    let window_iters = update_window_len(updates, update_window_dims.clone());";
    let bare_call = "let window_iters = update_window_len(updates, update_window_dims.clone());";
    let mut one_unmarked = sources.clone();
    let indexing_source = one_unmarked
        .iter_mut()
        .find_map(|(path, source)| path.ends_with("kernels/indexing.rs").then_some(source))
        .expect("indexing.rs should be inventoried");
    *indexing_source = indexing_source.replacen(invariant, bare_call, 1);
    assert!(scatter_update_window_contract(&one_unmarked).is_err());

    let mut divergent_derivation = sources.clone();
    let mod_source = divergent_derivation
        .iter_mut()
        .find_map(|(path, source)| path.ends_with("cubecl/mod.rs").then_some(source))
        .expect("cubecl/mod.rs should be inventoried");
    *mod_source = mod_source.replacen("updates_shape[axis]", "operand_shape[axis]", 1);
    assert!(scatter_update_window_contract(&divergent_derivation).is_err());

    let mut divergent_axes = sources.clone();
    let mod_source = divergent_axes
        .iter_mut()
        .find_map(|(path, source)| path.ends_with("cubecl/mod.rs").then_some(source))
        .expect("cubecl/mod.rs should be inventoried");
    *mod_source = mod_source.replacen(
        "&config.update_window_dims,\n        updates_shape.len(),",
        "&config.inserted_window_dims,\n        updates_shape.len(),",
        1,
    );
    assert!(scatter_update_window_contract(&divergent_axes).is_err());

    let mut extra_product = sources.clone();
    extra_product.push((
        PathBuf::from("synthetic/unmarked.rs"),
        "let third = update_window_len(updates, dims);".to_owned(),
    ));
    assert!(scatter_update_window_contract(&extra_product).is_err());

    let mut aliased_product = sources.clone();
    aliased_product.push((
        PathBuf::from("synthetic/aliased_product.rs"),
        "use crate::kernels::indexing::update_window_len as checked_window;\n\
         let third = checked_window(updates, dims);"
            .to_owned(),
    ));
    assert!(scatter_update_window_contract(&aliased_product).is_err());

    let mut aliased_scatter_module = sources.clone();
    aliased_scatter_module.push((
        PathBuf::from("synthetic/aliased_launch.rs"),
        "use crate::kernels::indexing as scatter_kernels;\n\
         scatter_kernels::scatter_float_kernel::launch_unchecked();"
            .to_owned(),
    ));
    assert!(scatter_update_window_contract(&aliased_scatter_module).is_err());

    let mut extra_launch = sources;
    extra_launch.push((
        PathBuf::from("synthetic/launch.rs"),
        "indexing::scatter_float_kernel::launch_unchecked();".to_owned(),
    ));
    assert!(scatter_update_window_contract(&extra_launch).is_err());
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

    let backend_source = cubecl_source("mod.rs");
    let reduction_section = source_section(
        &backend_source,
        "    fn launch_reduce_axis_typed<T>(",
        "    fn reduce_axes_typed<T>(",
    );
    assert_ordered_needles(
        "launch_reduce_axis_typed",
        reduction_section,
        &[
            "let input_binding = typed_tensor_binding(input, op)?;",
            "let output = alloc_output::<T>(self.runtime(), &output_shape)?;",
            "if output.n_elements() == 0",
        ],
    );

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
fn cubecl_binary_elementwise_kernels_do_not_materialize_scalar_broadcasts() {
    let dispatch_source = cubecl_source("dispatch.rs");
    let binary_macro = source_section(
        &dispatch_source,
        "macro_rules! launch_binary_elementwise_kernel",
        "macro_rules! dispatch_binary_float_complex_int",
    );

    assert!(
        !binary_macro.contains("broadcast_typed"),
        "raw binary elementwise launchers must not allocate dense scalar-broadcast temporaries"
    );
    assert!(
        !binary_macro.contains("shape().is_empty()"),
        "scalar broadcast should be represented as BroadcastInDim and fused by backend hooks"
    );

    let mod_source = cubecl_source("mod.rs");
    let fusion_impl = source_section(
        &mod_source,
        "impl TensorFusion for CudaBackend",
        "impl BackendCachedDot for CudaBackend",
    );
    assert_ordered_needles(
        "CudaBackend broadcast multiply hook",
        fusion_impl,
        &[
            "fn execute_broadcast_multiply(",
            "launch_broadcast_multiply_typed",
            "launch_broadcast_multiply_int_typed",
            "launch_broadcast_multiply_complex_typed",
        ],
    );
}

#[test]
fn cubecl_scalar_div_rem_is_narrow_and_pow_remains_equal_shape() {
    let mod_source = cubecl_source("mod.rs");
    let helper = source_section(
        &mod_source,
        "fn launch_scalar_binary",
        "fn launch_checked_integer_scalar_binary",
    );
    assert!(!helper.contains("broadcast_typed"));
    assert_ordered_needles(
        "scalar binary shape gate",
        helper,
        &[
            "lhs.shape().is_empty() ^ rhs.shape().is_empty()",
            "ensure_resident_on_runtime(backend.runtime(), lhs, op)?",
            "ensure_resident_on_runtime(backend.runtime(), rhs, op)?",
            "let lhs_scalar = lhs.shape().is_empty()",
            "if lhs_scalar",
            "rhs.shape()",
            "lhs.shape()",
            "let output = alloc_output::<I>",
            "typed_tensor_array_arg(&output, op)?",
            "typed_tensor_array_arg(lhs, op)?",
            "typed_tensor_array_arg(rhs, op)?",
            "if output.n_elements() == 0",
        ],
    );

    let checked_helper = source_section(
        &mod_source,
        "fn launch_checked_integer_scalar_binary",
        "impl TensorElementwise for CudaBackend",
    );
    assert_ordered_needles(
        "checked scalar binary launch validation",
        checked_helper,
        &[
            "lhs.shape().is_empty() ^ rhs.shape().is_empty()",
            "ensure_resident_on_runtime(backend.runtime(), lhs, op)?",
            "ensure_resident_on_runtime(backend.runtime(), rhs, op)?",
            "let output = alloc_output::<I>",
            "typed_tensor_array_arg(&output, op)?",
            "typed_tensor_array_arg(lhs, op)?",
            "typed_tensor_array_arg(rhs, op)?",
            "if output.n_elements() == 0",
            "let flag = alloc_output::<i32>",
            "typed_tensor_array_arg(&flag, op)?",
            "launch_nullary_into(",
        ],
    );

    for (op, end) in [("fn div(", "fn rem("), ("fn rem(", "fn abs(")] {
        let section = source_section(&mod_source, op, end);
        assert!(
            section.contains("launch_scalar_binary"),
            "{op} must use the narrow scalar launcher"
        );
        assert!(
            !section.contains("broadcast_typed"),
            "{op} must not materialize the scalar"
        );
    }
    let pow = source_section(&mod_source, "fn pow(", "fn transpose(");
    assert!(!pow.contains("launch_scalar_binary"));
    assert_ordered_needles(
        "pow dtype and shape validation",
        pow,
        &[
            "if lhs.dtype() != rhs.dtype()",
            "return Err(dtype_mismatch(op, lhs, rhs))",
            "ensure_same_shape(op, lhs.shape(), rhs.shape())?",
        ],
    );
    assert!(pow.contains("launch_binary("));
}

#[test]
fn cubecl_real_complex_scalar_promotion_stays_device_native_and_narrow() {
    let mod_source = cubecl_source("mod.rs");
    let helper = source_section(
        &mod_source,
        "fn launch_real_complex_scalar_binary",
        "fn promoted_real_complex_scalar_binary",
    );
    assert_ordered_needles(
        "mixed real-complex scalar validation",
        helper,
        &[
            "if !real.shape().is_empty()",
            "ensure_resident_on_runtime(backend.runtime(), real, op)?",
            "ensure_resident_on_runtime(backend.runtime(), complex, op)?",
            "let component_len = complex\n        .n_elements()\n        .checked_mul(2)",
            "let real_arg = typed_tensor_array_arg(real, op)?",
            "// INVARIANT: `num_complex::Complex<T>` is `repr(C)` with interleaved `{ re, im }`",
            "let complex_arg = typed_tensor_array_arg_as::<C, R>(complex, component_len, op)?",
            "let output = alloc_output::<C>",
            "let output_arg = typed_tensor_array_arg_as::<C, R>(&output, component_len, op)?",
            "if output.n_elements() == 0",
            "scalar_real_complex_binary::launch_unchecked",
        ],
    );
    for banned in [
        "download_tensor",
        "upload_tensor",
        "broadcast_typed",
        "convert(",
    ] {
        assert!(
            !helper.contains(banned),
            "mixed scalar promotion must not use host transfer or full-size materialization: {banned}"
        );
    }

    let dispatch = source_section(
        &mod_source,
        "fn promoted_real_complex_scalar_binary",
        "fn launch_checked_integer_scalar_binary",
    );
    for accepted in [
        "Tensor::F32(real), Tensor::C32(complex)",
        "Tensor::C32(complex), Tensor::F32(real)",
        "Tensor::F64(real), Tensor::C64(complex)",
        "Tensor::C64(complex), Tensor::F64(real)",
    ] {
        assert!(
            dispatch.contains(accepted),
            "missing accepted pair {accepted}"
        );
    }
    assert_eq!(dispatch.matches("if real.shape().is_empty()").count(), 4);

    let kernel_source = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/kernels/elementwise.rs"),
    )
    .expect("elementwise kernel source should be readable");
    let kernel = source_section(
        &kernel_source,
        "pub fn scalar_real_complex_binary",
        "pub fn scalar_div_int_checked",
    );
    assert!(kernel.contains("let complex_idx = ABSOLUTE_POS * 2"));
    assert!(kernel
        .contains("// INVARIANT: These unsimplified component expressions must evaluate in the"));
    assert!(kernel.contains("Keep zero cross terms"));
    for generic_complex_term in [
        "zero + im",
        "im + zero",
        "scalar * re - zero * im",
        "re * scalar - im * zero",
        "let norm_sqr = scalar * scalar + zero * zero",
        "(re * scalar + im * zero) / norm_sqr",
        "let norm_sqr = re * re + im * im",
        "(scalar * re + zero * im) / norm_sqr",
        "(zero * re - scalar * im) / norm_sqr",
    ] {
        assert!(
            kernel.contains(generic_complex_term),
            "mixed scalar kernel must preserve generic complex operation order: {generic_complex_term}"
        );
    }
}

#[test]
fn cubecl_interop_download_validates_buffer_before_empty_fast_path() {
    let interop_source = cubecl_source("interop.rs");
    let download_source = source_section(
        &interop_source,
        "pub fn download_typed_tensor<",
        "/// Allocate a CubeCL-owned byte workspace",
    );

    assert_ordered_needles(
        "interop::download_typed_tensor",
        download_source,
        &[
            "dispatch::ensure_resident_on_runtime(rt, tensor, op)?;",
            "let buffer = dispatch::cubecl_buffer(tensor, op)?;",
            "if tensor.n_elements() == 0",
            "rt.synchronize()?;",
            ".read_one(buffer.handle().clone())",
        ],
    );
}

#[test]
fn cubecl_raw_device_pointer_paths_validate_runtime_residency() {
    let memory_source = cubecl_source("memory.rs");
    let memory_ptr = source_section(
        &memory_source,
        "pub fn device_ptr(rt: &CudaRuntime, tensor: &Tensor) -> crate::Result<u64> {",
        "fn upload_typed<",
    );
    assert_ordered_needles(
        "memory::device_ptr",
        memory_ptr,
        &[
            "ensure_tensor_resident_on_runtime(rt, tensor, \"device_ptr\")?;",
            "let handle = cubecl_handle(tensor)?;",
            ".get_resource(handle)",
        ],
    );

    let interop_source = cubecl_source("interop.rs");
    let interop_ptr = source_section(
        &interop_source,
        "pub fn typed_device_ptr<T: 'static>(",
        "/// Upload host data into a dense GPU tensor",
    );
    assert_ordered_needles(
        "interop::typed_device_ptr",
        interop_ptr,
        &[
            "dispatch::ensure_resident_on_runtime(rt, tensor, op)?;",
            "dispatch::cubecl_buffer(tensor, op)?;",
            ".get_resource(buffer.handle().clone())",
        ],
    );

    let gemm_source = cubecl_source("gemm.rs");
    let gemm_ptr = source_section(
        &gemm_source,
        "fn typed_device_ptr<T: 'static>(",
        "fn zero_alloc<T>",
    );
    assert_ordered_needles(
        "gemm::typed_device_ptr",
        gemm_ptr,
        &[
            "ensure_resident_on_runtime(rt, tensor, OP)?;",
            "cubecl_buffer(tensor, OP)?;",
            ".get_resource(buffer.handle().clone())",
        ],
    );
}

#[test]
fn cubecl_host_download_paths_synchronize_before_reading() {
    let memory_source = cubecl_source("memory.rs");
    let typed_download = source_section(
        &memory_source,
        "fn download_typed<T: CubeElement + Clone + 'static>(",
        "fn upload_bool(",
    );
    assert_ordered_needles(
        "memory::download_typed",
        typed_download,
        &[
            "if typed.n_elements() == 0",
            "rt.synchronize()?;",
            ".read_one(handle)",
        ],
    );

    let bool_download = source_section(&memory_source, "fn download_bool(", "fn cubecl_handle(");
    assert_ordered_needles(
        "memory::download_bool",
        bool_download,
        &[
            "if typed.n_elements() == 0",
            "rt.synchronize()?;",
            ".read_one(handle)",
        ],
    );
}

#[test]
fn cubecl_scatter_validates_all_device_inputs_before_binding() {
    let mod_source = cubecl_source("mod.rs");
    let scatter_float = source_section(
        &mod_source,
        "    fn scatter_float_typed<",
        "    fn scatter_complex_typed<",
    );
    assert_ordered_needles(
        "scatter_float_typed",
        scatter_float,
        &[
            "ensure_resident_on_runtime(self.runtime(), scatter_indices, \"scatter\")?;",
            "ensure_resident_on_runtime(self.runtime(), updates, \"scatter\")?;",
            "typed_tensor_binding(scatter_indices, \"scatter\")?;",
            "typed_tensor_binding(updates, \"scatter\")?;",
        ],
    );

    let scatter_complex = source_section(
        &mod_source,
        "    fn scatter_complex_typed<",
        "impl BackendRuntimeCache for CudaBackend",
    );
    assert_ordered_needles(
        "scatter_complex_typed",
        scatter_complex,
        &[
            "ensure_resident_on_runtime(self.runtime(), scatter_indices, \"scatter\")?;",
            "ensure_resident_on_runtime(self.runtime(), updates, \"scatter\")?;",
            "typed_tensor_binding(scatter_indices, \"scatter\")?;",
            "typed_tensor_binding(updates, \"scatter\")?;",
        ],
    );
}

#[test]
fn cubecl_indexing_kernels_use_saturating_window_arithmetic() {
    let indexing_source = gpu_source(&["kernels", "indexing.rs"]);
    let clamp = source_section(
        &indexing_source,
        "pub(crate) fn clamp_window_start",
        "#[cube]\npub(crate) fn index_component",
    );
    assert!(
        clamp.contains("dim_size.saturating_sub(window_size)"),
        "GPU gather/scatter clamp_window_start must not underflow when window_size exceeds dim_size"
    );
    assert!(
        !clamp.contains("dim_size - window_size"),
        "GPU clamp_window_start must not use unchecked usize subtraction"
    );

    let scatter_float = source_section(
        &indexing_source,
        "pub fn scatter_float_kernel",
        "#[cube(launch_unchecked)]\npub fn scatter_complex_kernel",
    );
    assert!(
        scatter_float.contains("clamp_window_start::<I>"),
        "GPU float scatter must clamp out-of-range starts like the CPU backend"
    );
    assert!(
        !scatter_float.contains("start < I::from_int(0)"),
        "GPU float scatter must not skip negative starts instead of clamping them"
    );

    let scatter_complex = source_tail(&indexing_source, "pub fn scatter_complex_kernel");
    assert!(
        scatter_complex.contains("clamp_window_start::<I>"),
        "GPU complex scatter must clamp out-of-range starts like the CPU backend"
    );
    assert!(
        !scatter_complex.contains("start < I::from_int(0)"),
        "GPU complex scatter must not skip negative starts instead of clamping them"
    );

    let structural_source = gpu_source(&["kernels", "structural.rs"]);
    let reverse = source_section(
        &structural_source,
        "pub fn reverse_kernel",
        "#[cube(launch_unchecked)]\npub fn concatenate_copy_kernel",
    );
    assert!(
        reverse.contains("dim.saturating_sub(1)"),
        "GPU reverse_kernel should guard zero-sized dimensions with saturating_sub"
    );
    assert!(
        !reverse.contains("dim - 1"),
        "GPU reverse_kernel must not compute dim - 1 directly"
    );
}

#[test]
fn cubecl_gather_and_pad_validate_shape_bounds_before_launch() {
    let mod_source = cubecl_source("mod.rs");
    let gather_meta = source_section(
        &mod_source,
        "fn gather_launch_meta(",
        "struct ScatterLaunchMeta",
    );
    assert!(
        gather_meta.contains("validate_slice_sizes_within_operand(\"gather\""),
        "GPU gather launch metadata must reject per-axis slice sizes larger than the operand"
    );

    let pad_shape = source_section(&mod_source, "fn pad_output_shape(", "fn index_vector_size(");
    assert!(
        pad_shape.contains("i64::try_from(input_shape[axis])"),
        "GPU pad output shape must not cast usize dimensions to i64 with `as`"
    );
    assert!(
        !pad_shape.contains("input_shape[axis] as i64"),
        "GPU pad output shape must use checked conversion before signed arithmetic"
    );
}

#[test]
fn cubecl_pad_mapping_avoids_signed_edge_subtraction_overflow() {
    let indexing_source = gpu_source(&["kernels", "indexing.rs"]);
    let pad_kernel = source_section(
        &indexing_source,
        "pub fn pad_kernel",
        "pub fn gather_kernel",
    );
    assert!(!pad_kernel.contains("out_idx[axis] as i64 - low"));
    assert!(pad_kernel.contains("low.unsigned_abs()"));
    assert_ordered_needles(
        "pad_kernel candidate bounds check",
        pad_kernel,
        &[
            "let candidate = shifted / spacing;",
            "if candidate >= input.shape(axis) as u64",
            "input_idx[axis] = candidate as usize;",
        ],
    );
}

#[test]
fn cubecl_scatter_reports_unsupported_integer_operand_dtypes() {
    let mod_source = cubecl_source("mod.rs");
    let scatter_source = source_section(&mod_source, "    fn scatter(", "    fn slice(");
    for needle in [
        "(Tensor::I32(_), _, _)",
        "(Tensor::I64(_), _, _)",
        "(Tensor::Bool(_), _, _)",
    ] {
        assert!(
            scatter_source.contains(needle),
            "GPU scatter should explicitly reject unsupported operand dtype arm {needle}"
        );
    }
    assert!(
        scatter_source.contains("Err(unsupported_dtype(\"scatter\", operand.dtype()))"),
        "GPU scatter unsupported operand arms should report unsupported dtype rather than ternary mismatch"
    );
}

#[test]
fn cubecl_runtime_initializes_context_before_client_and_syncs_on_drop() {
    let runtime_source = cubecl_source("runtime.rs");
    let new_source = source_section(
        &runtime_source,
        "    pub fn new(device_ordinal: usize) -> crate::Result<Self> {",
        "    pub(crate) fn client(&self)",
    );
    assert_ordered_needles(
        "CudaRuntime::new",
        new_source,
        &[
            "cudarc::runtime::result::device::set",
            "cudarc::driver::result::init()",
            "let primary_context = CudaPrimaryContext::retain(cuda_device)?;",
            "cudarc::driver::result::ctx::set_current",
            "let device = CudaDevice::new(device_ordinal);",
            "let client =",
            "::client(&device);",
        ],
    );

    let drop_source = source_section(&runtime_source, "impl Drop for CudaRuntime", "}");
    assert_ordered_needles(
        "CudaRuntime::drop",
        drop_source,
        &[
            "if let Err(err) = self.synchronize()",
            "report_cuda_runtime_drop_error(&err);",
        ],
    );
}

#[test]
fn cubecl_gemm_zero_contracting_path_stays_device_native() {
    let gemm_source = cubecl_source("gemm.rs");
    let zero_alloc_source = source_section(&gemm_source, "fn zero_alloc<", "fn build_layout<");

    for banned in [
        "vec![T::zero(); len]",
        "create_from_slice(T::as_bytes(&zeros))",
    ] {
        assert!(
            !zero_alloc_source.contains(banned),
            "CubeCL GEMM zero-contracting fast path must not materialize host zeros: {banned}"
        );
    }
    assert_ordered_needles(
        "gemm::zero_alloc",
        zero_alloc_source,
        &[
            "alloc_output::<T>(rt, shape)",
            "structural::fill_zero_kernel",
        ],
    );
}

#[test]
fn cubecl_raw_device_pointer_paths_use_exposed_provenance() {
    for (name, source) in [
        ("cubecl/interop.rs", cubecl_source("interop.rs")),
        ("cubecl/gemm.rs", cubecl_source("gemm.rs")),
    ] {
        assert!(
            !source.contains("as usize as *mut c_void"),
            "{name} must not recreate raw CUDA pointers through an integer-pointer roundtrip"
        );
        assert!(
            source.contains("cuda_device_ptr_from_addr"),
            "{name} should centralize CUDA device address conversion through the provenance-aware helper"
        );
    }
}

#[test]
fn cubecl_runtime_uses_primary_context_guard_during_initialization() {
    let runtime_source = cubecl_source("runtime.rs");
    assert!(
        runtime_source.contains("struct CudaPrimaryContext"),
        "CudaRuntime initialization should retain the CUDA primary context through an RAII guard"
    );
    assert!(
        runtime_source.contains("primary_context: CudaPrimaryContext"),
        "CudaRuntime should own the retained primary context guard"
    );
    assert!(
        runtime_source.contains("impl Drop for CudaPrimaryContext"),
        "Cuda primary context release should be tied to the guard Drop implementation"
    );
    assert!(
        !runtime_source.contains("let _ = unsafe { cudarc::driver::result::primary_ctx::release"),
        "Cuda primary context release status should not be silently discarded from CudaRuntime::drop"
    );
}

#[test]
fn cubecl_extension_cache_guard_validates_downcast_before_deref() {
    let mod_source = cubecl_source("mod.rs");
    let cache_source = source_section(
        &mod_source,
        "pub fn get_or_try_init<T>",
        "impl Default for CudaExtensionCache",
    );
    let guard_source = source_section(
        &mod_source,
        "pub struct CudaExtensionCacheGuard",
        "impl CudaBackend",
    );

    assert!(
        cache_source.contains("downcast_ref::<T>()"),
        "CudaExtensionCacheGuard construction should validate the cached value type before returning"
    );
    assert!(
        guard_source.contains("value:"),
        "CudaExtensionCacheGuard should store a typed pointer validated during construction"
    );
    assert!(
        !guard_source
            .contains(".expect(\"CudaExtensionCache stored value under the wrong TypeId\")"),
        "CudaExtensionCacheGuard::deref should not panic on cache corruption"
    );
}

#[test]
fn cutensor_drop_paths_report_destroy_status() {
    let source = cubecl_source("ffi/cutensor.rs");
    for banned in [
        "let _ = unsafe { (self.lib.vtable.destroy)(self.raw) };",
        "let _ = unsafe { (self.lib.vtable.destroy_tensor_descriptor)(self.raw) };",
        "let _ = unsafe { (self.lib.vtable.destroy_operation_descriptor)(self.raw) };",
        "let _ = unsafe { (self.lib.vtable.destroy_plan_preference)(self.raw) };",
        "let _ = unsafe { (self.lib.vtable.destroy_plan)(self.raw) };",
    ] {
        assert!(
            !source.contains(banned),
            "cuTENSOR Drop paths must inspect destroy status instead of discarding it: found {banned}"
        );
    }
    assert!(
        source.contains("report_cutensor_destroy_status"),
        "cuTENSOR Drop paths should share a helper that reports non-success destroy statuses"
    );
}

#[test]
fn cutensor_data_symbols_validate_pointer_before_deref() {
    let source = cubecl_source("ffi/cutensor.rs");
    let section = source_section(
        &source,
        "unsafe fn load_data_symbol",
        "struct CutensorLibrary",
    );

    assert!(
        !section.contains("Ok(**symbol)"),
        "cuTENSOR data symbol loading must not blindly double-deref the exported pointer"
    );
    assert!(
        section.contains("let ptr = *symbol;"),
        "cuTENSOR data symbol loading should name the exported pointer before validation"
    );
    assert!(
        section.contains("ptr.is_null()") && section.contains("ptr.is_aligned()"),
        "cuTENSOR data symbol loading must reject null or misaligned descriptor pointers"
    );
    assert!(
        section.contains("std::ptr::read(ptr)"),
        "cuTENSOR data symbol loading should read the validated data symbol pointer explicitly"
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

#[test]
fn cuda_float_index_validation_stays_device_native_and_preflighted() {
    let backend = cubecl_source("mod.rs");
    let validation = source_section(
        &backend,
        "fn validate_float_index_tensor<F>(",
        "fn launch_checked_integer_binary<I>(",
    );
    for needle in [
        "ensure_resident_on_runtime(backend.runtime(), indices, \"index_tensor\")?;",
        "typed_tensor_binding(indices, \"index_tensor\")?;",
        "if indices.n_elements() == 0",
        "u32::try_from(indices.n_elements())",
        "cube_count_for_len(indices.n_elements())?;",
        "validate_float_indices_kernel::launch_unchecked",
        "extract_invalid_float_index_kernel::launch_unchecked",
        "F::read_invalid_flag(backend, &flag)?",
    ] {
        assert!(
            validation.contains(needle),
            "missing float-index contract: {needle}"
        );
    }
    let allocation = validation.find("alloc_output::<F>").unwrap();
    assert!(validation.find("ensure_resident_on_runtime").unwrap() < allocation);
    assert!(validation.find("typed_tensor_binding").unwrap() < allocation);
    assert!(validation.find("cube_count_for_len").unwrap() < allocation);
    assert!(validation.find("u32::try_from").unwrap() < allocation);
    assert!(!validation.contains("download_tensor(backend.runtime(), indices"));

    for (start, end, index_name) in [
        (
            "fn dynamic_slice_typed<T, I>(",
            "fn dynamic_slice_bool<I>(",
            "starts",
        ),
        (
            "fn gather_typed<T, I>(",
            "fn gather_bool<I>(",
            "start_indices",
        ),
        (
            "fn scatter_float_typed<T, I>(",
            "fn scatter_complex_typed<T, F, I>(",
            "scatter_indices",
        ),
    ] {
        let operation = source_section(&backend, start, end);
        assert!(
            operation
                .find(&format!("I::validate(self, {index_name})?;"))
                .unwrap()
                < operation.find("alloc_output").unwrap_or(operation.len()),
            "{start} must validate float index values before allocation"
        );
    }

    let kernels = std::fs::read_to_string("src/kernels/indexing.rs").unwrap();
    assert!(kernels.contains("flag[0].fetch_min(ABSOLUTE_POS as u32)"));
    assert!(kernels.contains("flag_values[1] = indices[invalid_index as usize]"));
}

#[cfg(feature = "cuda")]
#[test]
fn cubecl_runtime_exposes_explicit_synchronize() {
    let _sync: fn(&tenferro_gpu::CudaRuntime) -> tenferro_tensor::Result<()> =
        tenferro_gpu::CudaRuntime::synchronize;
}
