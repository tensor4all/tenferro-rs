use std::{fs, path::Path};

#[test]
fn logical_kernels_do_not_take_tensor_shapes_as_comptime_parameters() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let files = [
        root.join("diagonal.rs"),
        root.join("indexing.rs"),
        root.join("structural.rs"),
    ];
    let banned = [
        "#[comptime] input_shape",
        "#[comptime] output_shape",
        "#[comptime] operand_shape",
        "#[comptime] updates_shape",
        "#[comptime] scatter_indices_shape",
        "#[comptime] start_indices_shape",
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
