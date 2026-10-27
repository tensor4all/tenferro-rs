use std::path::PathBuf;

fn repo_file(path: &str) -> String {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.push("../..");
    root.push(path);
    std::fs::read_to_string(root).expect("source file must be readable")
}

#[cfg(all(feature = "cuda", not(feature = "cuda")))]
#[test]
fn cuda_public_api_requires_typed_device_and_caller_selected_engine() {
    use tenferro_gpu::{
        cuda::cuda_runtime_engine_registration, cuda::CudaDeviceError, cuda::CudaDeviceId,
        cuda::CudaEngine, cuda::CudaRuntime,
    };
    use tenferro_runtime::{EngineRegistration, RuntimeConfigError};

    let _: fn(CudaDeviceId) -> Result<CudaRuntime, CudaDeviceError> = CudaRuntime::new;
    let _: fn(CudaDeviceId) -> Result<CudaEngine, CudaDeviceError> = CudaEngine::open;
    let _: fn(&CudaEngine) -> Result<EngineRegistration, RuntimeConfigError> =
        cuda_runtime_engine_registration;
}

#[cfg(all(feature = "cuda", not(feature = "cuda")))]
#[test]
fn cuda_prepared_access_has_one_state_and_one_downcast_boundary() {
    let dispatch = repo_file("crates/tenferro-gpu/src/cubecl/dispatch.rs");
    // The two state checks below are about the prepared-access type, so they read
    // its body: the file legitimately carries an `element_size` *parameter* in the
    // single output-allocation path, which a file-wide search would mistake for a
    // retained field.
    let prepared = dispatch
        .split("pub(crate) struct CubeclPreparedAccess {")
        .nth(1)
        .and_then(|rest| rest.split('}').next())
        .expect("CubeclPreparedAccess must be defined in dispatch.rs");
    assert!(
        !prepared.contains("writable"),
        "prepared CUDA state must not retain a dead writable flag"
    );
    assert!(
        !prepared.contains("offset") && !prepared.contains("element_size"),
        "prepared CUDA state must not retain dead layout scalar fields"
    );
    assert_eq!(
        dispatch
            .matches("downcast::<CubeclPreparedAccess>()")
            .count(),
        1,
        "CUDA dispatch must centralize prepared-state downcasting"
    );
    assert!(
        !dispatch.contains("request.allocation_domain()")
            && !dispatch.contains("request.allocation_id()")
            && !dispatch.contains("request.byte_len()"),
        "CUDA preparation must not repeat root identity validation"
    );
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
        assert!(rendered_text.contains("offset zero"));
        assert!(rendered_text.contains("full allocation"));
        assert!(rendered_text.contains("Bool"));
        assert!(rendered_text.contains("must not alias"));
    }
}
