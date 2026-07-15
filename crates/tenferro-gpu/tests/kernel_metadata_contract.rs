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

#[test]
fn logical_kernels_do_not_take_tensor_shapes_as_comptime_parameters() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("kernels");
    let files = [
        root.join("diagonal.rs"),
        root.join("helpers.rs"),
        root.join("indexing.rs"),
        root.join("structural.rs"),
    ];
    let banned = [
        "#[comptime] batch_shape",
        "#[comptime] input_shape",
        "#[comptime] output_shape",
        "#[comptime] operand_shape",
        "#[comptime] updates_shape",
        "#[comptime] scatter_indices_shape",
        "#[comptime] start_indices_shape",
        "#[comptime] window_shape_updates",
        "#[comptime] shape: Sequence<usize>",
    ];

    let mut violations = Vec::new();
    for file in files {
        let source = fs::read_to_string(&file).expect("kernel source should be readable");
        for needle in banned {
            if source.contains(needle) {
                violations.push(format!("{} contains {needle}", file.display()));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "logical CubeCL kernels must receive tensor shape/stride through TensorBinding, not comptime shape parameters:\n{}",
        violations.join("\n")
    );
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
    for invocation in [
        "reduce_wrapping_int_kernel!(reduce_sum_int, wrapping_add);",
        "reduce_wrapping_int_kernel!(reduce_prod_int, wrapping_mul);",
        "reduce_wrapping_int_plane_kernel!(reduce_sum_int_plane, wrapping_add, wrapping_plane_sum);",
        "reduce_wrapping_int_plane_kernel!(reduce_prod_int_plane, wrapping_mul, wrapping_plane_prod);",
    ] {
        assert!(
            reductions.contains(invocation),
            "integer reduction must use {invocation}"
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
    ] {
        assert!(helpers.contains(helper), "missing CubeCL helper {helper}");
    }

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
        source.contains("fn emit_nan_propagating_extrema("),
        "fusion codegen must centralize the NaN contract"
    );
    assert!(
        source.contains("Comparison::IsNan"),
        "fusion extrema must test both operands for NaN"
    );
    assert!(
        source.contains("Operator::Select"),
        "fusion extrema must select a NaN operand before the native extrema result"
    );
    assert!(
        source.contains(
            "ElementwiseFusionOp::Maximum => {\n            emit_nan_propagating_extrema("
        ),
        "fused maximum must use the NaN-propagating helper"
    );
    assert!(
        source.contains(
            "ElementwiseFusionOp::Minimum => {\n            emit_nan_propagating_extrema("
        ),
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
