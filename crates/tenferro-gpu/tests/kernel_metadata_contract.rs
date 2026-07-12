use std::{fs, path::Path};

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
