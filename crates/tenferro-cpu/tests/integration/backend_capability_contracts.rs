use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use tenferro_cpu::CpuBackend;
use tenferro_tensor::{
    BackendSession, BackendSessionHost, TensorAnalytic, TensorBackend, TensorBuffer,
    TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing,
    TensorReduction, TensorStructural,
};

fn rust_function_body<'a>(source: &'a str, function: &str) -> Option<&'a str> {
    let signature = format!("fn {function}");
    let function_start = source
        .match_indices(&signature)
        .find(|(start, _)| {
            matches!(
                source.as_bytes().get(start + signature.len()),
                Some(b'(') | Some(b'<')
            )
        })?
        .0;
    let body_start = function_start + source[function_start..].find('{')?;
    let mut depth = 0usize;
    for (offset, character) in source[body_start..].char_indices() {
        match character {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&source[body_start..=body_start + offset]);
                }
            }
            _ => {}
        }
    }
    None
}

fn rust_tokens(source: &str) -> Vec<String> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        if let Some((content_start, hashes)) = raw_string_open(bytes, cursor) {
            cursor = content_start;
            while cursor < bytes.len() {
                if bytes[cursor] == b'"'
                    && bytes
                        .get(cursor + 1..cursor + 1 + hashes)
                        .is_some_and(|suffix| suffix.iter().all(|&byte| byte == b'#'))
                {
                    cursor += 1 + hashes;
                    break;
                }
                cursor += 1;
            }
        } else if bytes[cursor..].starts_with(b"//") {
            cursor += 2;
            while cursor < bytes.len() && bytes[cursor] != b'\n' {
                cursor += 1;
            }
        } else if bytes[cursor..].starts_with(b"/*") {
            cursor += 2;
            let mut depth = 1usize;
            while cursor < bytes.len() && depth > 0 {
                if bytes[cursor..].starts_with(b"/*") {
                    depth += 1;
                    cursor += 2;
                } else if bytes[cursor..].starts_with(b"*/") {
                    depth -= 1;
                    cursor += 2;
                } else {
                    cursor += 1;
                }
            }
        } else if bytes[cursor] == b'"' {
            cursor += 1;
            while cursor < bytes.len() {
                if bytes[cursor] == b'\\' {
                    cursor = (cursor + 2).min(bytes.len());
                } else if bytes[cursor] == b'"' {
                    cursor += 1;
                    break;
                } else {
                    cursor += 1;
                }
            }
        } else if bytes[cursor] == b'\''
            && ((cursor + 2 < bytes.len() && bytes[cursor + 2] == b'\'')
                || (cursor + 3 < bytes.len()
                    && bytes[cursor + 1] == b'\\'
                    && bytes[cursor + 3] == b'\''))
        {
            cursor += if bytes[cursor + 1] == b'\\' { 4 } else { 3 };
        } else if bytes[cursor].is_ascii_alphabetic() || bytes[cursor] == b'_' {
            let start = cursor;
            cursor += 1;
            while cursor < bytes.len()
                && (bytes[cursor].is_ascii_alphanumeric() || bytes[cursor] == b'_')
            {
                cursor += 1;
            }
            tokens.push(source[start..cursor].to_string());
        } else if bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        } else {
            tokens.push(char::from(bytes[cursor]).to_string());
            cursor += 1;
        }
    }
    tokens
}

fn raw_string_open(bytes: &[u8], start: usize) -> Option<(usize, usize)> {
    let mut cursor = if bytes.get(start..start + 2) == Some(b"br") {
        start + 2
    } else if bytes.get(start) == Some(&b'r') {
        start + 1
    } else {
        return None;
    };
    let hashes_start = cursor;
    while bytes.get(cursor) == Some(&b'#') {
        cursor += 1;
    }
    (bytes.get(cursor) == Some(&b'"')).then_some((cursor + 1, cursor - hashes_start))
}

fn function_names(tokens: &[String]) -> BTreeSet<String> {
    tokens
        .windows(2)
        .filter(|window| window[0] == "fn")
        .map(|window| window[1].clone())
        .collect()
}

fn public_function_names(tokens: &[String]) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for (index, token) in tokens.iter().enumerate() {
        if token != "pub" || tokens.get(index + 1).is_some_and(|next| next == "(") {
            continue;
        }
        let mut cursor = index + 1;
        while tokens.get(cursor).is_some_and(|modifier| {
            matches!(modifier.as_str(), "async" | "const" | "unsafe" | "extern")
        }) {
            cursor += 1;
        }
        if tokens.get(cursor).is_some_and(|token| token == "fn") {
            if let Some(name) = tokens.get(cursor + 1) {
                names.insert(name.clone());
            }
        }
    }
    names
}

fn rust_source_files(root: &Path) -> Vec<PathBuf> {
    fn visit(directory: &Path, files: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(directory).expect("Rust source directory must be readable") {
            let path = entry.expect("Rust source entry must be readable").path();
            if path.is_dir() {
                visit(&path, files);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(path);
            }
        }
    }

    let mut files = Vec::new();
    visit(root, &mut files);
    files.sort();
    files
}

fn ownership_contract(rules: &str) -> BTreeMap<&str, &str> {
    const BEGIN: &str = "<!-- TENFERRO_CPU_STRIDED_OWNERSHIP_CONTRACT_BEGIN -->";
    const END: &str = "<!-- TENFERRO_CPU_STRIDED_OWNERSHIP_CONTRACT_END -->";
    let block = rules
        .split_once(BEGIN)
        .expect("CPU strided ownership contract begin marker must exist")
        .1
        .split_once(END)
        .expect("CPU strided ownership contract end marker must exist")
        .0;
    block
        .lines()
        .filter_map(|line| line.trim().split_once('='))
        .map(|(key, value)| (key.trim(), value.trim()))
        .collect()
}

fn contract_set<'a>(contract: &BTreeMap<&str, &'a str>, key: &str) -> BTreeSet<&'a str> {
    contract
        .get(key)
        .unwrap_or_else(|| panic!("CPU strided ownership contract lacks field `{key}`"))
        .split(',')
        .collect()
}

fn token_tree_end(tokens: &[String], open: usize) -> Option<usize> {
    let mut delimiters = Vec::new();
    for (index, token) in tokens.iter().enumerate().skip(open) {
        match token.as_str() {
            "(" => delimiters.push(")"),
            "[" => delimiters.push("]"),
            "{" => delimiters.push("}"),
            ")" | "]" | "}" => {
                if delimiters.pop() != Some(token.as_str()) {
                    return None;
                }
                if delimiters.is_empty() {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

fn macro_literal_identifiers(tokens: &[String]) -> BTreeSet<String> {
    let mut identifiers = BTreeSet::new();
    for index in 0..tokens.len() {
        let open = if tokens
            .get(index)
            .is_some_and(|token| token == "macro_rules")
            && tokens.get(index + 1).is_some_and(|token| token == "!")
        {
            (index + 2..tokens.len())
                .find(|&candidate| matches!(tokens[candidate].as_str(), "(" | "[" | "{"))
        } else if tokens.get(index + 1).is_some_and(|token| token == "!")
            && tokens
                .get(index + 2)
                .is_some_and(|token| matches!(token.as_str(), "(" | "[" | "{"))
        {
            Some(index + 2)
        } else {
            None
        };
        let Some(open) = open else {
            continue;
        };
        let Some(end) = token_tree_end(tokens, open) else {
            continue;
        };
        identifiers.extend(
            tokens[open + 1..end]
                .iter()
                .filter(|token| {
                    token
                        .as_bytes()
                        .first()
                        .is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'_')
                })
                .cloned(),
        );
    }
    identifiers
}

fn contains_include_macro(tokens: &[String]) -> bool {
    tokens
        .windows(2)
        .any(|window| window[0] == "include" && window[1] == "!")
}

fn accepts_backend_capabilities<B>()
where
    B: TensorElementwise
        + TensorAnalytic
        + TensorStructural
        + TensorReduction
        + TensorIndexing
        + TensorDot
        + TensorFusion
        + TensorBuffer
        + TensorDeviceTransfer
        + BackendSessionHost
        + TensorBackend,
{
}

fn accepts_session_capabilities<S>(_: &mut S)
where
    S: TensorElementwise
        + TensorAnalytic
        + TensorStructural
        + TensorReduction
        + TensorIndexing
        + TensorDot
        + TensorFusion
        + TensorBuffer
        + BackendSession
        + ?Sized,
{
}

#[test]
fn cpu_backend_exposes_narrow_capability_bounds() {
    accepts_backend_capabilities::<CpuBackend>();
}

#[test]
fn backend_session_exposes_narrow_capability_bounds() {
    let mut backend = CpuBackend::new();
    backend.with_backend_session(|session| {
        accepts_session_capabilities(session);
    });
}

#[test]
fn backend_surface_no_longer_uses_forwarding_macro() {
    let backend_source = include_str!("../../src/backend.rs");
    assert!(!backend_source.contains("forward_exec_to_backend"));
}

#[test]
fn read_elementwise_and_analytic_paths_do_not_materialize_views() {
    let elementwise_source =
        include_str!("../../../tenferro-internal-cpu-kernels/src/elementwise.rs");
    let analytic_source = include_str!("../../src/analytic.rs");

    assert!(
        !elementwise_source.contains("materialize_tensor_read"),
        "elementwise read paths must dispatch over TensorRead views directly"
    );
    assert!(
        !analytic_source.contains("materialize_tensor_read"),
        "analytic read paths must dispatch over TensorRead views directly"
    );
}

#[test]
fn structural_read_paths_dispatch_directly_to_typed_view_helpers() {
    let backend_source = include_str!("../../src/backend.rs");
    let session_source = include_str!("../../src/exec_session.rs");
    let structural_source = include_str!("../../src/structural.rs");

    for (surface, source) in [
        ("CpuBackend", backend_source),
        ("CpuExecSession", session_source),
    ] {
        let structural_impl = source
            .split_once(&format!("impl TensorStructural for {surface}"))
            .expect("TensorStructural implementation must exist")
            .1;
        for (operation, helper) in [
            ("transpose_read", "transpose_read_with_pool"),
            ("reshape_read", "reshape_read_with_pool"),
            ("broadcast_in_dim_read", "broadcast_in_dim_read_with_pool"),
        ] {
            let implementation = rust_function_body(structural_impl, operation)
                .unwrap_or_else(|| panic!("{surface}::{operation} must be implemented"));
            assert!(
                !implementation.contains("materialize_tensor_read"),
                "{surface}::{operation} must not materialize an intermediate input"
            );
            assert!(
                implementation.contains(&format!("structural::{helper}")),
                "{surface}::{operation} must dispatch to structural::{helper}"
            );
        }
    }

    for (read_helper, typed_view_helper) in [
        ("transpose_read_with_pool", "typed_transpose_view_with_pool"),
        ("reshape_read_with_pool", "typed_reshape_view_with_pool"),
        (
            "broadcast_in_dim_read_with_pool",
            "typed_broadcast_in_dim_view_with_pool",
        ),
    ] {
        let implementation = rust_function_body(structural_source, read_helper)
            .unwrap_or_else(|| panic!("structural::{read_helper} must own dtype dispatch"));
        assert!(
            implementation.contains(typed_view_helper),
            "structural::{read_helper} must dispatch to {typed_view_helper}"
        );
    }
}

#[test]
fn indexing_hot_loops_do_not_recompute_multi_indices_from_flat_offsets() {
    let indexing_source = include_str!("../../src/indexing.rs");

    assert!(
        !indexing_source.contains("flat_to_multi"),
        "indexing kernels should carry column-major indices incrementally after validation"
    );
}

#[test]
fn concatenate_hot_loop_does_not_linearly_scan_input_segments() {
    let indexing_source = include_str!("../../src/indexing.rs");

    assert!(
        !indexing_source.contains(".position(|&end| concat_idx < end)"),
        "concatenate should not linearly scan all input segment ends for each output element"
    );
    assert!(
        indexing_source.contains("partition_point"),
        "concatenate should use precomputed ordered segment boundaries for logarithmic lookup"
    );
}

#[test]
fn gather_scatter_index_component_reuses_index_scratch() {
    let indexing_source = include_str!("../../src/indexing.rs");

    assert!(
        !indexing_source.contains("let mut full_idx = vec![0usize; indices.shape.len()];"),
        "gather/scatter should not allocate index vectors for every index component"
    );
    assert!(
        indexing_source.contains("index_scratch"),
        "gather/scatter should carry reusable index scratch through index_component"
    );
}

#[test]
fn cpu_public_ops_require_backend_owner() {
    let lib_source = include_str!("../../src/lib.rs");
    for reexport in [
        "pub use analytic::pow;",
        "pub use elementwise::",
        "pub use indexing::",
        "pub use reduction::",
        "pub use structural::",
    ] {
        assert!(
            !lib_source.contains(reexport),
            "resource-bypassing reexport remains: {reexport}"
        );
    }

    for (module, source) in [
        ("analytic", include_str!("../../src/analytic.rs")),
        (
            "elementwise",
            include_str!("../../../tenferro-internal-cpu-kernels/src/elementwise.rs"),
        ),
        ("indexing", include_str!("../../src/indexing.rs")),
        ("structural", include_str!("../../src/structural.rs")),
    ] {
        assert!(
            !source.contains("fn with_local_pool"),
            "{module} still constructs a throwaway BufferPool"
        );
    }
}

#[test]
fn strided_kernel_ownership_requires_backend_execution_resources() {
    let rules = include_str!("../../../../REPOSITORY_RULES.md");
    let contract = ownership_contract(rules);
    assert_eq!(
        contract.get("schema"),
        Some(&"tenferro.cpu-strided-ownership.v1")
    );
    assert_eq!(contract.get("affine-kernel-owner"), Some(&"strided-rs"));
    assert_eq!(
        contract.get("einsum-owner"),
        Some(&"tenferro:benchmark-backed-exception")
    );
    assert_eq!(contract.get("execution-entry"), Some(&"CpuBackend"));
    assert_eq!(
        contract_set(&contract, "affine-kernels"),
        BTreeSet::from([
            "axis-reduction",
            "broadcast",
            "copy",
            "map",
            "permutation",
            "zip-map",
        ])
    );
    assert_eq!(
        contract_set(&contract, "execution-resources"),
        BTreeSet::from([
            "CpuContext-Rayon",
            "nested-execution",
            "persistent-BufferPool",
            "serial-parallel-threshold",
            "uninitialized-full-overwrite",
        ])
    );
    assert_eq!(
        contract_set(&contract, "noncompliant"),
        BTreeSet::from([
            "ambient-global-Rayon",
            "context-free-strided-call",
            "throwaway-pool",
        ])
    );
    assert_eq!(
        contract.get("resource-classification"),
        Some(&"memory-reuse-and-thread-policy:execution-not-metadata")
    );
    assert_eq!(
        contract.keys().copied().collect::<BTreeSet<_>>(),
        BTreeSet::from([
            "affine-kernel-owner",
            "affine-kernels",
            "einsum-owner",
            "execution-entry",
            "execution-resources",
            "noncompliant",
            "resource-classification",
            "schema",
        ])
    );
}

#[test]
fn tensor_public_surface_has_no_context_free_materialization_api() {
    let tensor_source_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tenferro-tensor/src");
    let files = rust_source_files(&tensor_source_root);
    assert!(
        files.iter().any(|path| path.ends_with("backend.rs"))
            && files.iter().any(|path| path.ends_with("types.rs")),
        "source scan must cover the complete tenferro-tensor source tree"
    );

    let mut public_functions = BTreeMap::<String, Vec<PathBuf>>::new();
    let mut all_functions = BTreeMap::<String, Vec<PathBuf>>::new();
    let forbidden_identifiers = BTreeSet::from([
        "copy_from_contiguous",
        "materialize_typed_view_col_major",
        "materialize_view_buffer_col_major",
        "to_contiguous",
        "to_tensor",
    ]);
    for path in files {
        let source = fs::read_to_string(&path).expect("Rust source file must be readable");
        let tokens = rust_tokens(&source);
        assert!(
            !contains_include_macro(&tokens),
            "`include!` can hide generated public API and is forbidden in tenferro-tensor source: {}",
            path.display()
        );
        let forbidden_macro_identifiers = macro_literal_identifiers(&tokens)
            .into_iter()
            .filter(|identifier| forbidden_identifiers.contains(identifier.as_str()))
            .collect::<BTreeSet<_>>();
        assert!(
            forbidden_macro_identifiers.is_empty(),
            "macro token stream in {} contains forbidden materialization identifiers: {:?}",
            path.display(),
            forbidden_macro_identifiers
        );
        for name in public_function_names(&tokens) {
            public_functions.entry(name).or_default().push(path.clone());
        }
        for name in function_names(&tokens) {
            all_functions.entry(name).or_default().push(path.clone());
        }
    }

    for forbidden in ["to_contiguous", "copy_from_contiguous", "to_tensor"] {
        assert!(
            !public_functions.contains_key(forbidden),
            "context-free public materialization function `{forbidden}` remains in {:?}",
            public_functions.get(forbidden)
        );
    }
    for forbidden in [
        "materialize_view_buffer_col_major",
        "materialize_typed_view_col_major",
    ] {
        assert!(
            !all_functions.contains_key(forbidden),
            "context-free materialization helper `{forbidden}` remains in {:?}",
            all_functions.get(forbidden)
        );
    }
}

#[test]
fn rust_public_function_scan_is_format_and_literal_independent() {
    let fixture = r####"
        // pub fn to_tensor(&self) {}
        const DECOY: &str = "pub fn to_contiguous(&self)";
        const RAW_DECOY: &str = r###"pub fn copy_from_contiguous() { "still raw" }"###;
        pub(crate) fn to_tensor() {}
        pub
        async
        fn relocated_materializer() {}
        fn private_helper() {}
    "####;
    let tokens = rust_tokens(fixture);
    assert_eq!(
        public_function_names(&tokens),
        BTreeSet::from(["relocated_materializer".to_string()])
    );
    assert_eq!(
        function_names(&tokens),
        BTreeSet::from([
            "private_helper".to_string(),
            "relocated_materializer".to_string(),
            "to_tensor".to_string(),
        ])
    );

    let generated_fixture = r#"
        macro_rules! expose { ($name:ident) => { pub fn $name() {} } }
        expose!(to_tensor);
        include!("generated.rs");
    "#;
    let generated_tokens = rust_tokens(generated_fixture);
    assert!(contains_include_macro(&generated_tokens));
    assert!(
        macro_literal_identifiers(&generated_tokens).contains("to_tensor"),
        "literal forbidden identifiers in macro definitions/invocations must be visible"
    );
}

#[test]
fn install_pool_has_no_placeholder_construction_or_gemm_descriptor_clones() {
    let backend_source = include_str!("../../src/backend.rs");
    let buffer_pool_source =
        include_str!("../../../tenferro-internal-cpu-kernels/src/buffer_pool.rs");
    let gemm_source = include_str!("../../src/gemm/mod.rs");
    let exec_session_source = include_str!("../../src/exec_session.rs");

    assert!(!backend_source.contains("std::mem::take(target)"));
    assert!(backend_source.contains("buffers: &'a mut BufferPool"));
    assert!(buffer_pool_source.contains("OnceLock"));
    assert!(buffer_pool_source.contains("parse_default_max_retained_capacity_bytes"));
    assert!(gemm_source.contains("lhs: &TensorRead<'_>"));
    assert!(gemm_source.contains("rhs: &TensorRead<'_>"));
    assert!(!backend_source.contains("lhs.clone()"));
    assert!(!backend_source.contains("rhs.clone()"));
    assert!(!exec_session_source.contains("lhs.clone()"));
    assert!(!exec_session_source.contains("rhs.clone()"));
}

#[test]
fn cpu_provider_dispatch_has_no_runtime_registry_lookup_or_legacy_staging() {
    let assert_direct_dispatch = |owner: &str, source: &str| {
        for forbidden in [
            "HashMap",
            "TypeId",
            "dyn Any",
            "downcast",
            "provider_name.to_string",
            "with_base_dot_general_provider",
            "match self.dot_general_provider",
        ] {
            assert!(
                !source.contains(forbidden),
                "{owner} contains forbidden hot-path token `{forbidden}`"
            );
        }
    };

    let sources = [
        ("dot_runtime", include_str!("../../src/dot_runtime.rs")),
        ("provider", include_str!("../../src/provider.rs")),
        ("exec_session", include_str!("../../src/exec_session.rs")),
        ("gemm_analysis", include_str!("../../src/gemm/mod.rs")),
    ];
    for (module, source) in sources {
        assert_direct_dispatch(module, source);
    }

    // backend.rs also owns opt-in profiling state, so scanning the entire file
    // would reject a HashMap that is not part of contraction dispatch. Scan all
    // session-entry and contraction bodies instead.
    let backend = include_str!("../../src/backend.rs");
    for function in [
        "with_linalg_pool",
        "dot_general",
        "dot_general_read",
        "dot_general_read_into",
        "dot_general_read_into_accum",
        "dot_general_with_conj",
        "dot_general_cached",
        "dot_general_with_conj_cached",
        "dot_general_read_into_accum_cached",
        "grouped_gemm_cached",
        "run_backend_session_cached",
        "with_backend_session",
        "with_backend_session_cached",
    ] {
        let body = rust_function_body(backend, function)
            .unwrap_or_else(|| panic!("backend dispatch function `{function}` must exist"));
        assert_direct_dispatch(&format!("backend::{function}"), body);
    }
}
