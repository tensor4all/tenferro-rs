use std::{fs, path::Path};

fn kernel_source(path: &[&str]) -> String {
    let mut source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("kernels");
    for component in path {
        source.push(component);
    }
    fs::read_to_string(&source).unwrap_or_else(|err| {
        panic!(
            "kernel source {} should be readable: {err}",
            source.display()
        )
    })
}

fn scatter_kernel_names(source: &str) -> Vec<&str> {
    const PREFIX: &str = "pub fn scatter_";

    source
        .match_indices(PREFIX)
        .filter_map(|(start, _)| {
            let name_start = start + "pub fn ".len();
            let name_end = source[name_start..]
                .find(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
                .map_or(source.len(), |offset| name_start + offset);
            let name = &source[name_start..name_end];
            name.ends_with("_kernel").then_some(name)
        })
        .collect()
}

#[test]
fn scatter_kernel_inventory_discovers_unreviewed_definitions() {
    let source = "pub fn scatter_copy_kernel() {}\npub fn scatter_new_kernel() {}";

    assert_eq!(
        scatter_kernel_names(source),
        ["scatter_copy_kernel", "scatter_new_kernel"]
    );
}

/// Why a kernel parameter may be compile-time. Anything else (extents, strides,
/// offsets, lengths, starts, paddings, window sizes, running axis offsets) is a
/// runtime value: compiling it in creates one NVRTC module per distinct value.
/// See "CubeCL kernels that perform logical tensor indexing" in
/// `docs/design/gpu-backend-design.md`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ComptimeCategory {
    /// A rank or other loop bound that sizes local arrays or unrolled loops.
    Rank,
    /// An operation attribute that names axes or roles, not their extents.
    AxisMapping,
    /// Algorithm configuration: tiling, vector width, operand mode or flags.
    AlgorithmConfig,
    /// The WebGPU/Metal native-permutation layout, compiled in on that backend
    /// only. CUDA launches the runtime-layout twin of the kernel.
    DocumentedException,
}

use ComptimeCategory::{AlgorithmConfig, AxisMapping, DocumentedException, Rank};

/// Every `#[comptime]` parameter under `src/`, classified in review.
///
/// `(file under src/, owning fn or `macro_name!` for macro-generated kernels,
/// parameter, category)`. A new compile-time parameter fails
/// `every_comptime_kernel_parameter_is_classified` until it is added here.
const COMPTIME_PARAMETER_ALLOWLIST: &[(&str, &str, &str, ComptimeCategory)] = &[
    (
        "kernels/diagonal.rs",
        "extract_diagonal_kernel",
        "axis_a",
        AxisMapping,
    ),
    (
        "kernels/diagonal.rs",
        "extract_diagonal_kernel",
        "axis_b",
        AxisMapping,
    ),
    (
        "kernels/diagonal.rs",
        "extract_diagonal_kernel",
        "diag_output_axis",
        AxisMapping,
    ),
    (
        "kernels/diagonal.rs",
        "extract_diagonal_kernel",
        "input_rank",
        Rank,
    ),
    (
        "kernels/diagonal.rs",
        "extract_diagonal_kernel",
        "output_rank",
        Rank,
    ),
    (
        "kernels/diagonal.rs",
        "embed_diagonal_copy_kernel",
        "axis_a",
        AxisMapping,
    ),
    (
        "kernels/diagonal.rs",
        "embed_diagonal_copy_kernel",
        "axis_b",
        AxisMapping,
    ),
    (
        "kernels/diagonal.rs",
        "embed_diagonal_copy_kernel",
        "input_rank",
        Rank,
    ),
    (
        "kernels/diagonal.rs",
        "embed_diagonal_copy_kernel",
        "output_rank",
        Rank,
    ),
    (
        "kernels/elementwise.rs",
        "broadcast_source_index",
        "dims",
        AxisMapping,
    ),
    (
        "kernels/elementwise.rs",
        "broadcast_source_index",
        "output_rank",
        Rank,
    ),
    (
        "kernels/elementwise.rs",
        "broadcast_multiply_kernel!",
        "lhs_dims",
        AxisMapping,
    ),
    (
        "kernels/elementwise.rs",
        "broadcast_multiply_kernel!",
        "rhs_dims",
        AxisMapping,
    ),
    (
        "kernels/elementwise.rs",
        "broadcast_multiply_kernel!",
        "output_rank",
        Rank,
    ),
    (
        "kernels/elementwise.rs",
        "broadcast_multiply_int",
        "lhs_dims",
        AxisMapping,
    ),
    (
        "kernels/elementwise.rs",
        "broadcast_multiply_int",
        "rhs_dims",
        AxisMapping,
    ),
    (
        "kernels/elementwise.rs",
        "broadcast_multiply_int",
        "output_rank",
        Rank,
    ),
    (
        "kernels/elementwise.rs",
        "axpby_strided_source_kernel!",
        "rank",
        Rank,
    ),
    (
        "kernels/elementwise.rs",
        "scalar_binary_float_kernel!",
        "lhs_scalar",
        AlgorithmConfig,
    ),
    (
        "kernels/elementwise.rs",
        "scalar_real_complex_binary",
        "real_lhs",
        AlgorithmConfig,
    ),
    (
        "kernels/elementwise.rs",
        "scalar_real_complex_binary",
        "mode",
        AlgorithmConfig,
    ),
    (
        "kernels/elementwise.rs",
        "scalar_div_int_checked",
        "lhs_scalar",
        AlgorithmConfig,
    ),
    (
        "kernels/elementwise.rs",
        "scalar_rem_int_checked",
        "lhs_scalar",
        AlgorithmConfig,
    ),
    (
        "kernels/elementwise.rs",
        "scalar_pow_int_checked",
        "lhs_scalar",
        AlgorithmConfig,
    ),
    (
        "kernels/elementwise.rs",
        "compare_float_bool",
        "mode",
        AlgorithmConfig,
    ),
    (
        "kernels/elementwise.rs",
        "compare_int_bool",
        "mode",
        AlgorithmConfig,
    ),
    ("kernels/helpers.rs", "flat_to_tensor_index", "rank", Rank),
    ("kernels/helpers.rs", "multi_to_tensor_index", "rank", Rank),
    (
        "kernels/helpers.rs",
        "axis_in_sequence",
        "axes",
        AxisMapping,
    ),
    (
        "kernels/helpers.rs",
        "axis_position_in_sequence",
        "axes",
        AxisMapping,
    ),
    (
        "kernels/indexing.rs",
        "index_component",
        "index_vector_dim",
        AxisMapping,
    ),
    (
        "kernels/indexing.rs",
        "index_component",
        "component",
        AxisMapping,
    ),
    ("kernels/indexing.rs", "index_component", "rank", Rank),
    (
        "kernels/indexing.rs",
        "flat_to_index_batch_index",
        "index_vector_dim",
        AxisMapping,
    ),
    (
        "kernels/indexing.rs",
        "flat_to_index_batch_index",
        "indices_rank",
        Rank,
    ),
    (
        "kernels/indexing.rs",
        "flat_to_update_window_index",
        "update_window_dims",
        AxisMapping,
    ),
    (
        "kernels/indexing.rs",
        "update_window_len",
        "update_window_dims",
        AxisMapping,
    ),
    // The slice *step* is a static operation attribute; the starts are runtime.
    (
        "kernels/indexing.rs",
        "slice_kernel",
        "strides",
        AlgorithmConfig,
    ),
    ("kernels/indexing.rs", "dynamic_slice_kernel", "rank", Rank),
    ("kernels/indexing.rs", "pad_kernel", "rank", Rank),
    (
        "kernels/indexing.rs",
        "gather_kernel",
        "window_dims",
        AxisMapping,
    ),
    (
        "kernels/indexing.rs",
        "gather_kernel",
        "offset_dims",
        AxisMapping,
    ),
    (
        "kernels/indexing.rs",
        "gather_kernel",
        "start_index_map",
        AxisMapping,
    ),
    (
        "kernels/indexing.rs",
        "gather_kernel",
        "index_vector_dim",
        AxisMapping,
    ),
    ("kernels/indexing.rs", "gather_kernel", "operand_rank", Rank),
    ("kernels/indexing.rs", "gather_kernel", "out_rank", Rank),
    (
        "kernels/indexing.rs",
        "gather_kernel",
        "start_indices_rank",
        Rank,
    ),
    (
        "kernels/indexing.rs",
        "scatter_float_kernel",
        "window_dims",
        AxisMapping,
    ),
    (
        "kernels/indexing.rs",
        "scatter_float_kernel",
        "update_window_dims",
        AxisMapping,
    ),
    (
        "kernels/indexing.rs",
        "scatter_float_kernel",
        "scatter_dims_to_operand_dims",
        AxisMapping,
    ),
    (
        "kernels/indexing.rs",
        "scatter_float_kernel",
        "index_vector_dim",
        AxisMapping,
    ),
    (
        "kernels/indexing.rs",
        "scatter_float_kernel",
        "operand_rank",
        Rank,
    ),
    (
        "kernels/indexing.rs",
        "scatter_float_kernel",
        "updates_rank",
        Rank,
    ),
    (
        "kernels/indexing.rs",
        "scatter_float_kernel",
        "scatter_indices_rank",
        Rank,
    ),
    (
        "kernels/indexing.rs",
        "scatter_complex_kernel",
        "window_dims",
        AxisMapping,
    ),
    (
        "kernels/indexing.rs",
        "scatter_complex_kernel",
        "update_window_dims",
        AxisMapping,
    ),
    (
        "kernels/indexing.rs",
        "scatter_complex_kernel",
        "scatter_dims_to_operand_dims",
        AxisMapping,
    ),
    (
        "kernels/indexing.rs",
        "scatter_complex_kernel",
        "index_vector_dim",
        AxisMapping,
    ),
    (
        "kernels/indexing.rs",
        "scatter_complex_kernel",
        "operand_rank",
        Rank,
    ),
    (
        "kernels/indexing.rs",
        "scatter_complex_kernel",
        "updates_rank",
        Rank,
    ),
    (
        "kernels/indexing.rs",
        "scatter_complex_kernel",
        "scatter_indices_rank",
        Rank,
    ),
    (
        "kernels/structural.rs",
        "strided_view_offset_from_tensor",
        "rank",
        Rank,
    ),
    (
        "kernels/structural.rs",
        "fill_zero_view_kernel",
        "rank",
        Rank,
    ),
    (
        "kernels/structural.rs",
        "materialize_strided_kernel",
        "rank",
        Rank,
    ),
    (
        "kernels/structural.rs",
        "materialize_strided_comptime_layout_kernel",
        "dims",
        DocumentedException,
    ),
    (
        "kernels/structural.rs",
        "materialize_strided_comptime_layout_kernel",
        "src_strides",
        DocumentedException,
    ),
    (
        "kernels/structural.rs",
        "materialize_strided_comptime_layout_kernel",
        "len",
        DocumentedException,
    ),
    (
        "kernels/structural.rs",
        "materialize_strided_comptime_layout_kernel",
        "rank",
        Rank,
    ),
    (
        "kernels/structural.rs",
        "tiled_transpose_kernel",
        "tile",
        AlgorithmConfig,
    ),
    (
        "kernels/structural.rs",
        "tiled_transpose_kernel",
        "block_rows",
        AlgorithmConfig,
    ),
    (
        "kernels/structural.rs",
        "tiled_transpose_kernel",
        "padding",
        AlgorithmConfig,
    ),
    (
        "kernels/structural.rs",
        "tiled_transpose_kernel",
        "vector_width",
        AlgorithmConfig,
    ),
    (
        "kernels/structural.rs",
        "tiled_transpose_comptime_layout_kernel",
        "batch_stride",
        DocumentedException,
    ),
    (
        "kernels/structural.rs",
        "tiled_transpose_comptime_layout_kernel",
        "dst_fast_extent",
        DocumentedException,
    ),
    (
        "kernels/structural.rs",
        "tiled_transpose_comptime_layout_kernel",
        "src_fast_extent",
        DocumentedException,
    ),
    (
        "kernels/structural.rs",
        "tiled_transpose_comptime_layout_kernel",
        "tile",
        AlgorithmConfig,
    ),
    (
        "kernels/structural.rs",
        "tiled_transpose_comptime_layout_kernel",
        "block_rows",
        AlgorithmConfig,
    ),
    (
        "kernels/structural.rs",
        "tiled_transpose_comptime_layout_kernel",
        "padding",
        AlgorithmConfig,
    ),
    (
        "kernels/structural.rs",
        "tiled_transpose_comptime_layout_kernel",
        "vector_width",
        AlgorithmConfig,
    ),
    (
        "kernels/structural.rs",
        "contiguous_to_view_kernel",
        "rank",
        Rank,
    ),
    (
        "kernels/structural.rs",
        "strided_to_strided_kernel",
        "rank",
        Rank,
    ),
    (
        "kernels/structural.rs",
        "broadcast_in_dim_kernel",
        "dims",
        AxisMapping,
    ),
    (
        "kernels/structural.rs",
        "broadcast_in_dim_kernel",
        "output_rank",
        Rank,
    ),
    // The dtype component stride (1 real, 2 complex), not a tensor stride.
    (
        "kernels/structural.rs",
        "validate_real_cast",
        "stride",
        AlgorithmConfig,
    ),
    (
        "kernels/structural.rs",
        "validate_real_cast",
        "max_inclusive",
        AlgorithmConfig,
    ),
    (
        "kernels/structural.rs",
        "extract_invalid_real_cast",
        "stride",
        AlgorithmConfig,
    ),
    (
        "kernels/structural.rs",
        "reverse_kernel",
        "axes",
        AxisMapping,
    ),
    ("kernels/structural.rs", "reverse_kernel", "rank", Rank),
    (
        "kernels/structural.rs",
        "concatenate_copy_kernel",
        "axis",
        AxisMapping,
    ),
    (
        "kernels/structural.rs",
        "concatenate_copy_kernel",
        "rank",
        Rank,
    ),
    (
        "webgpu/kernels.rs",
        "pack_lhs_dot_general",
        "free_axes",
        AxisMapping,
    ),
    (
        "webgpu/kernels.rs",
        "pack_lhs_dot_general",
        "contract_axes",
        AxisMapping,
    ),
    (
        "webgpu/kernels.rs",
        "pack_lhs_dot_general",
        "batch_axes",
        AxisMapping,
    ),
    (
        "webgpu/kernels.rs",
        "pack_lhs_dot_general",
        "input_rank",
        Rank,
    ),
    (
        "webgpu/kernels.rs",
        "pack_lhs_dot_general",
        "out_rank",
        Rank,
    ),
    (
        "webgpu/kernels.rs",
        "pack_rhs_dot_general",
        "contract_axes",
        AxisMapping,
    ),
    (
        "webgpu/kernels.rs",
        "pack_rhs_dot_general",
        "free_axes",
        AxisMapping,
    ),
    (
        "webgpu/kernels.rs",
        "pack_rhs_dot_general",
        "batch_axes",
        AxisMapping,
    ),
    (
        "webgpu/kernels.rs",
        "pack_rhs_dot_general",
        "input_rank",
        Rank,
    ),
    (
        "webgpu/kernels.rs",
        "pack_rhs_dot_general",
        "out_rank",
        Rank,
    ),
];

fn is_ident_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

fn leading_ident(text: &str) -> &str {
    let end = text
        .find(|ch: char| !is_ident_char(ch))
        .unwrap_or(text.len());
    &text[..end]
}

/// The `fn` (or, for `fn $name` inside a `macro_rules!`, the `macro_name!`)
/// whose parameter list contains byte `at` of `source`.
fn owning_function(source: &str, at: usize) -> String {
    let before = &source[..at];
    let mut search = before.len();
    while let Some(found) = before[..search].rfind("fn ") {
        search = found;
        let boundary = found == 0 || !before[..found].ends_with(is_ident_char);
        let rest = &before[found + 3..];
        let name = rest.trim_start();
        let name = name
            .strip_prefix('$')
            .map_or(leading_ident(name), |macro_arg| {
                if leading_ident(macro_arg).is_empty() {
                    ""
                } else {
                    "$"
                }
            });
        if !boundary || name.is_empty() {
            continue;
        }
        if name != "$" {
            return name.to_string();
        }
        let macro_start = before[..found]
            .rfind("macro_rules!")
            .unwrap_or_else(|| panic!("`fn $name` outside macro_rules! near byte {found}"));
        let macro_name = leading_ident(before[macro_start + "macro_rules!".len()..].trim_start());
        return format!("{macro_name}!");
    }
    panic!("#[comptime] at byte {at} has no enclosing fn");
}

/// Every `(owner, parameter)` marked `#[comptime]` in one source file.
fn comptime_parameters(source: &str) -> Vec<(String, String)> {
    const MARKER: &str = "#[comptime]";
    source
        .match_indices(MARKER)
        .filter(|&(at, _)| {
            // Prose in comments and docs may mention the attribute.
            let line_start = source[..at].rfind('\n').map_or(0, |newline| newline + 1);
            !source[line_start..at].trim_start().starts_with("//")
        })
        .map(|(at, _)| {
            let after = source[at + MARKER.len()..].trim_start();
            let after = after.strip_prefix("mut ").map_or(after, str::trim_start);
            let parameter = leading_ident(after);
            assert!(
                !parameter.is_empty(),
                "#[comptime] at byte {at} is not followed by a parameter name"
            );
            (owning_function(source, at), parameter.to_string())
        })
        .collect()
}

fn rust_sources_under(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in fs::read_dir(dir).expect("source directory should be readable") {
        let path = entry.expect("directory entry should be readable").path();
        if path.is_dir() {
            rust_sources_under(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn comptime_parameter_inventory_attributes_macro_and_helper_parameters() {
    let source = "\
#[cube]
fn helper(#[comptime] rank: usize) {}
/// docs mentioning fn bogus
macro_rules! twin {
    ($name:ident) => {
        pub fn $name(dims: Sequence<usize>, #[comptime] mut rank: usize) {}
    };
}
pub fn kernel<E>(#[comptime] axes: Sequence<usize>) {}
/// Docs may say `#[comptime]` in prose.
// So may comments: #[comptime] values compile per value.
";
    assert_eq!(
        comptime_parameters(source),
        [
            ("helper".to_string(), "rank".to_string()),
            ("twin!".to_string(), "rank".to_string()),
            ("kernel".to_string(), "axes".to_string()),
        ]
    );
}

/// Recurrence guard for #1885 §3.4 / #2010: every compile-time kernel
/// parameter in the crate must be classified. Tensor extents, strides, offsets,
/// lengths, starts, paddings and window sizes are runtime values; a reviewer
/// classifies a new compile-time parameter or moves it to run time.
#[test]
fn every_comptime_kernel_parameter_is_classified() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_sources_under(&src, &mut files);
    files.sort();

    let mut found = Vec::new();
    for file in &files {
        let relative = file
            .strip_prefix(&src)
            .expect("source under src/")
            .to_string_lossy()
            .replace('\\', "/");
        let source = fs::read_to_string(file).expect("source should be readable");
        for (owner, parameter) in comptime_parameters(&source) {
            found.push((relative.clone(), owner, parameter));
        }
    }

    let classified = |file: &str, owner: &str, parameter: &str| {
        COMPTIME_PARAMETER_ALLOWLIST
            .iter()
            .any(|&(f, o, p, _)| f == file && o == owner && p == parameter)
    };
    let unclassified: Vec<String> = found
        .iter()
        .filter(|(file, owner, parameter)| !classified(file, owner, parameter))
        .map(|(file, owner, parameter)| format!("{file}: {owner}(#[comptime] {parameter})"))
        .collect();
    assert!(
        unclassified.is_empty(),
        "unclassified compile-time kernel parameters (every distinct value compiles a new \
         module; pass extents/strides/offsets/lengths/starts/paddings at run time, or classify \
         the parameter in COMPTIME_PARAMETER_ALLOWLIST in review):\n{}",
        unclassified.join("\n")
    );

    let stale: Vec<String> = COMPTIME_PARAMETER_ALLOWLIST
        .iter()
        .filter(|&&(file, owner, parameter, _)| {
            !found
                .iter()
                .any(|(f, o, p)| f == file && o == owner && p == parameter)
        })
        .map(|(file, owner, parameter, _)| format!("{file}: {owner}({parameter})"))
        .collect();
    assert!(
        stale.is_empty(),
        "COMPTIME_PARAMETER_ALLOWLIST lists parameters that are no longer compile-time:\n{}",
        stale.join("\n")
    );
}

/// D-8 (#2010): the compile-time native-permutation layout is a WebGPU/Metal
/// exception only. CUDA must launch the runtime-layout kernels.
#[test]
fn documented_comptime_layout_exception_is_webgpu_only() {
    let exception_kernels: Vec<&str> = COMPTIME_PARAMETER_ALLOWLIST
        .iter()
        .filter(|entry| entry.3 == DocumentedException)
        .map(|entry| entry.1)
        .collect();
    assert!(!exception_kernels.is_empty());
    for kernel in &exception_kernels {
        assert!(
            kernel.ends_with("_comptime_layout_kernel"),
            "documented exceptions must live in a dedicated `*_comptime_layout_kernel`: {kernel}"
        );
    }

    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_sources_under(&src, &mut files);
    for file in files {
        let relative = file.strip_prefix(&src).expect("source under src/");
        if relative.starts_with("webgpu") || relative.starts_with("kernels") {
            continue;
        }
        let source = fs::read_to_string(&file).expect("source should be readable");
        for kernel in &exception_kernels {
            assert!(
                !source.contains(kernel),
                "{} launches the WebGPU-only compile-time layout kernel `{kernel}`",
                relative.display()
            );
        }
    }
}

#[test]
fn reduction_kernels_do_not_hide_unbounded_axis_work_in_one_worker() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("kernels");
    let launch_source = fs::read_to_string(root.join("reduce").join("launch.rs"))
        .expect("reduction launch source should be readable");
    let kernels_source = fs::read_to_string(root.join("reduce").join("kernels.rs"))
        .expect("reduction kernel source should be readable");
    let checks = [
        (
            "reduce/launch.rs",
            launch_source.as_str(),
            "ReduceStrategy::Auto | ReduceStrategy::Unit",
        ),
        (
            "reduce/kernels.rs",
            kernels_source.as_str(),
            "for reduce_index in 1..reduce_len",
        ),
    ];

    let mut violations = Vec::new();
    for (name, source, needle) in checks {
        if source.contains(needle) {
            violations.push(format!("{name} contains {needle}"));
        }
    }

    assert!(
        violations.is_empty(),
        "CubeCL reductions must not route Auto to a per-output worker with unbounded serial axis work; use a parallel reduction strategy or an explicitly bounded fallback:\n{}",
        violations.join("\n")
    );
}

#[test]
fn integer_kernels_route_user_arithmetic_through_wrapping_helpers() {
    let helpers = kernel_source(&["helpers.rs"]);
    for helper in [
        "fn wrapping_add<I: Int>",
        "fn wrapping_sub<I: Int>",
        "fn wrapping_mul<I: Int>",
        "fn wrapping_neg<I: Int>",
        "fn wrapping_plane_sum<I: Int>",
        "fn wrapping_plane_prod<I: Int>",
    ] {
        assert!(helpers.contains(helper), "missing CubeCL helper {helper}");
    }
    assert!(
        helpers.contains("INVARIANT: CubeCL fixed-width Int arithmetic"),
        "wrapping helpers must document the CubeCL codegen proof"
    );

    let elementwise = kernel_source(&["elementwise.rs"]);
    for call in [
        "wrapping_add::<I>(lhs[ABSOLUTE_POS], rhs[ABSOLUTE_POS])",
        "wrapping_sub::<I>(lhs[ABSOLUTE_POS], rhs[ABSOLUTE_POS])",
        "wrapping_mul::<I>(lhs[ABSOLUTE_POS], rhs[ABSOLUTE_POS])",
        "wrapping_mul::<I>(lhs[lhs_idx], rhs[rhs_idx])",
        "wrapping_neg::<I>(input[ABSOLUTE_POS])",
        "wrapping_neg::<I>(value)",
        "wrapping_sub::<I>(x, wrapping_mul::<I>(quotient, y))",
        "wrapping_sub::<I>(exp, wrapping_mul::<I>(quotient, two))",
        "wrapping_mul::<I>(acc, base)",
        "wrapping_mul::<I>(base, base)",
    ] {
        assert!(
            elementwise.contains(call),
            "integer elementwise kernel must use {call}"
        );
    }

    let reductions = kernel_source(&["reduce", "kernels.rs"]);
    // 0.11 folded the four integer-specific reduction macros into one
    // `reduce_binary_kernel!`; the contract is that the integer reductions still
    // route their arithmetic through the wrapping helpers.
    assert!(
        reductions.contains("macro_rules! reduce_binary_kernel"),
        "integer reduction must be built on the shared reduction macro"
    );
    for call in [
        "wrapping_add::<I>(acc, input[input_offset])",
        "wrapping_mul::<I>(acc, input[input_offset])",
        "wrapping_plane_sum::<T>(acc)",
        "wrapping_plane_prod::<T>(acc)",
    ] {
        assert!(
            reductions.contains(call),
            "integer reduction must use {call}"
        );
    }
}

#[test]
fn float_max_min_kernels_propagate_nan_across_lanes_and_planes() {
    let helpers = kernel_source(&["helpers.rs"]);
    for helper in [
        "fn nan_propagating_max<F: Float>",
        "fn nan_propagating_min<F: Float>",
        "fn plane_contains_nan<F: Float>",
        "fn plane_propagate_nan<F: Float>",
    ] {
        assert!(helpers.contains(helper), "missing CubeCL helper {helper}");
    }
    assert!(
        helpers.contains("plane_sum(nan_or_zero)"),
        "plane NaN propagation must carry an input NaN through the collective"
    );

    let elementwise = kernel_source(&["elementwise.rs"]);
    assert!(elementwise.contains("nan_propagating_max::<F>(lhs[ABSOLUTE_POS], rhs[ABSOLUTE_POS])"));
    assert!(elementwise.contains("nan_propagating_min::<F>(lhs[ABSOLUTE_POS], rhs[ABSOLUTE_POS])"));

    let reductions = kernel_source(&["reduce", "kernels.rs"]);
    assert!(
        reductions
            .match_indices("nan_propagating_max::<F>(acc, input[input_offset])")
            .count()
            >= 2,
        "unit and plane max reductions must both propagate NaN within each lane"
    );
    assert!(
        reductions
            .match_indices("nan_propagating_min::<F>(acc, input[input_offset])")
            .count()
            >= 2,
        "unit and plane min reductions must both propagate NaN within each lane"
    );
    assert!(
        reductions
            .match_indices("let contains_nan = plane_contains_nan::<F>(acc);")
            .count()
            >= 2,
        "plane max and min must aggregate a separate NaN flag across lanes"
    );
    assert!(
        reductions
            .match_indices("let propagated_nan = plane_propagate_nan::<F>(acc);")
            .count()
            >= 2,
        "plane max and min must propagate an actual NaN lane value"
    );
    assert!(
        !reductions.contains("F::new(f32::NAN)"),
        "generic CubeCL kernels must not lower a host NaN literal into invalid CUDA source"
    );
}

#[test]
fn fused_float_max_min_codegen_propagates_nan_before_native_extrema() {
    let source = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("cubecl")
            .join("fusion")
            .join("codegen.rs"),
    )
    .expect("fusion codegen source should be readable");

    assert!(
        source.contains("macro_rules! nan_extrema"),
        "fusion codegen must centralize the NaN contract"
    );
    assert!(
        source.contains("IsNanOp::new"),
        "fusion extrema must test both operands for NaN"
    );
    assert!(
        source.contains("SelectOp::new"),
        "fusion extrema must select a NaN operand before the native extrema result"
    );
    assert!(
        source.contains("ElementwiseFusionOp::Maximum => nan_extrema!(FMaxOp::new)"),
        "fused maximum must use the NaN-propagating helper"
    );
    assert!(
        source.contains("ElementwiseFusionOp::Minimum => nan_extrema!(FMinOp::new)"),
        "fused minimum must use the NaN-propagating helper"
    );
}

#[test]
fn scatter_kernels_are_not_single_thread_fallbacks() {
    let indexing_source = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("kernels")
            .join("indexing.rs"),
    )
    .expect("indexing kernel source should be readable");
    let reviewed_scatter_kernels = [
        "scatter_copy_kernel",
        "scatter_float_kernel",
        "scatter_complex_kernel",
    ];
    let scatter_kernels = scatter_kernel_names(&indexing_source);
    assert_eq!(
        scatter_kernels, reviewed_scatter_kernels,
        "review every added or removed pub fn scatter_*_kernel before updating this inventory"
    );
    let banned = ["ABSOLUTE_POS == 0", "for pos in 0..out.len()"];

    let mut violations = Vec::new();
    for kernel in scatter_kernels {
        let signature = format!("pub fn {kernel}");
        let start = indexing_source
            .find(&signature)
            .unwrap_or_else(|| panic!("indexing.rs should define {kernel}"));
        let remainder = &indexing_source[start..];
        let end = remainder.find("\n#[cube").unwrap_or(remainder.len());
        let kernel_source = &remainder[..end];

        for needle in banned {
            if kernel_source.contains(needle) {
                violations.push(format!("{kernel} contains {needle}"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "scatter CubeCL kernels must cover the output or update domain in parallel:\n{}",
        violations.join("\n")
    );
}
