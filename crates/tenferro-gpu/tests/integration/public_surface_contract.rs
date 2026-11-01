use std::path::PathBuf;

fn repo_file(path: &str) -> String {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.push("../..");
    root.push(path);
    std::fs::read_to_string(root).expect("source file must be readable")
}

fn feature_block(manifest: &str, name: &str) -> String {
    let prefix = format!("{name} = [");
    let mut block = String::new();
    let mut in_block = false;
    for line in manifest.lines() {
        if line.trim_start().starts_with(&prefix) {
            in_block = true;
        }
        if in_block {
            block.push_str(line);
            block.push('\n');
            if line.trim_end().ends_with(']') {
                return block;
            }
        }
    }
    panic!("{name} feature must be declared")
}

#[test]
fn gpu_backend_features_are_explicit_and_additive() {
    let manifest = repo_file("crates/tenferro-gpu/Cargo.toml");

    let default_feature = feature_block(&manifest, "default");
    assert!(
        !default_feature.contains("cuda") && !default_feature.contains("rocm"),
        "GPU backend providers must not be enabled by default"
    );
    assert!(
        !manifest.contains("\ngpu = ["),
        "tenferro-gpu should expose concrete backend features, not a vague gpu alias"
    );

    let cuda_feature = feature_block(&manifest, "cuda");
    assert!(
        cuda_feature.contains("dep:t4a-cuda-kernels") && cuda_feature.contains("dep:cudarc"),
        "cuda feature must own the kernel catalogue and the CUDA vendor bindings"
    );
}

#[test]
fn downstream_gpu_features_are_explicit_and_additive() {
    for manifest_path in [
        "crates/tenferro-ad/Cargo.toml",
        "crates/tenferro-einsum/Cargo.toml",
    ] {
        let manifest = repo_file(manifest_path);
        let default_feature = feature_block(&manifest, "default");
        assert!(
            !default_feature.contains("cuda") && !default_feature.contains("rocm"),
            "{manifest_path} must not enable GPU providers by default"
        );
        assert!(
            !manifest.contains("\ngpu = ["),
            "{manifest_path} should expose concrete backend features, not a vague gpu alias"
        );
        assert!(
            feature_block(&manifest, "cuda").contains("cuda"),
            "{manifest_path} should keep an explicit cuda feature"
        );
    }
}

#[test]
fn cuda_runtime_materialization_limitations_are_documented() {
    let design = repo_file("docs/design/gpu-backend-design.md");
    let guide = repo_file("docs/guides/devices-and-gpu.md");

    for document in [&design, &guide] {
        let rendered_text = document.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(rendered_text.contains("to_contiguous_read"));
        assert!(rendered_text.contains("copy_read_into"));
        // Wide copy views (#2334) use the general kernel, not the fixed-layout rank ceiling.
        assert!(rendered_text.contains("ranks above"));
        assert!(rendered_text.contains("device-resident geometry"));
        assert!(rendered_text.contains("must not alias"));
    }
}
