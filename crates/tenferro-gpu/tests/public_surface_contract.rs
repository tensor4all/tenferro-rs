use std::path::PathBuf;

fn repo_file(path: &str) -> String {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.push("../..");
    root.push(path);
    std::fs::read_to_string(root).expect("source file must be readable")
}

fn repo_file_if_exists(path: &str) -> Option<String> {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.push("../..");
    root.push(path);
    std::fs::read_to_string(root).ok()
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
fn cubecl_implementation_module_is_not_public_api() {
    let lib_rs = repo_file("crates/tenferro-gpu/src/lib.rs");
    assert!(
        !lib_rs.contains("pub mod cubecl;"),
        "CubeCL implementation module must not be exported as public API"
    );
    assert!(
        lib_rs.contains("mod cubecl;"),
        "CubeCL implementation module should remain crate-internal"
    );
    assert!(
        !lib_rs.contains("pub struct CubeclBuffer"),
        "CubeCL backend buffer representation must not be public API"
    );
    assert!(
        lib_rs.contains("pub mod cuda_interop"),
        "sibling-crate CUDA bridge should be the only explicit low-level public module"
    );

    let cubecl_mod = repo_file("crates/tenferro-gpu/src/cubecl/mod.rs");
    assert!(
        !cubecl_mod.contains("pub mod ffi;"),
        "CubeCL FFI implementation module must stay private"
    );
    assert!(
        !cubecl_mod.contains("pub mod interop;"),
        "interop bridge should be re-exported from the crate root, not from the implementation module"
    );
}

#[test]
fn gpu_backend_features_are_explicit_and_additive() {
    let manifest = repo_file("crates/tenferro-gpu/Cargo.toml");

    let default_feature = feature_block(&manifest, "default");
    assert!(
        !default_feature.contains("cuda")
            && !default_feature.contains("webgpu")
            && !default_feature.contains("rocm"),
        "GPU backend providers must not be enabled by default"
    );
    assert!(
        !manifest.contains("\ngpu = ["),
        "tenferro-gpu should expose concrete backend features, not a vague gpu alias"
    );

    let cuda_feature = feature_block(&manifest, "cuda");
    assert!(
        cuda_feature.contains("dep:cubecl-cuda") && cuda_feature.contains("dep:cudarc"),
        "cuda feature must own CUDA-specific runtime dependencies"
    );

    let webgpu_feature = feature_block(&manifest, "webgpu");
    assert!(
        webgpu_feature.contains("dep:cubecl-wgpu"),
        "webgpu feature must own the CubeCL WGPU runtime dependency"
    );
    assert!(
        !webgpu_feature.contains("dep:cubecl-cuda") && !webgpu_feature.contains("dep:cudarc"),
        "webgpu feature must not pull CUDA-only dependencies"
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
            !default_feature.contains("cuda")
                && !default_feature.contains("webgpu")
                && !default_feature.contains("rocm"),
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
        assert!(
            feature_block(&manifest, "webgpu").contains("webgpu"),
            "{manifest_path} should expose an explicit webgpu feature"
        );
    }
}

#[test]
fn public_backend_names_are_provider_specific() {
    let lib_rs = repo_file("crates/tenferro-gpu/src/lib.rs");
    assert!(
        lib_rs.contains("CudaBackend"),
        "CUDA backend should have an explicit public CudaBackend name"
    );
    assert!(
        lib_rs.contains("WebGpuBackend"),
        "WebGPU backend should have an explicit public WebGpuBackend name"
    );
    assert!(
        !lib_rs.contains("CubeclBackend") && !lib_rs.contains("CubeclRuntime"),
        "CubeCL implementation names should not remain as public CUDA backend aliases"
    );
}

#[test]
fn cuda_dot_general_stays_cutensor_backed_and_not_cubek_rewired() {
    let cuda_gemm = repo_file("crates/tenferro-gpu/src/cubecl/gemm.rs");
    assert!(
        cuda_gemm.contains("cutensor.contract("),
        "CUDA dot_general must remain cuTENSOR-backed in this work"
    );
    assert!(
        cuda_gemm.contains("alloc_workspace(backend.runtime(), workspace_size)")
            && cuda_gemm.contains("Plan::new(cutensor, &op_desc"),
        "CUDA workspace and plan flow must remain visible"
    );
    assert!(
        !cuda_gemm.contains("cubek_matmul") && !cuda_gemm.contains("DotGeneralPlan"),
        "WebGPU planner/CubeK changes must not rewrite the CUDA algorithm"
    );
    assert!(
        !cuda_gemm.contains("GpuScratchPool"),
        "common GPU scratch-pool design must not be wired into CUDA GEMM in this work"
    );
}

#[test]
fn webgpu_dot_general_is_cubek_backed() {
    let manifest = repo_file("crates/tenferro-gpu/Cargo.toml");
    let webgpu_feature = feature_block(&manifest, "webgpu");
    assert!(
        webgpu_feature.contains("dep:cubek-matmul") && webgpu_feature.contains("dep:cubek-std"),
        "webgpu dot_general must opt into CubeK matmul dependencies explicitly"
    );

    let webgpu_mod = repo_file("crates/tenferro-gpu/src/webgpu/mod.rs");
    let webgpu_gemm =
        repo_file_if_exists("crates/tenferro-gpu/src/webgpu/gemm.rs").unwrap_or_default();
    assert!(
        !webgpu_mod.contains("unsupported!(\"webgpu_dot_general\")"),
        "WebGPU dot_general should dispatch a real matmul path instead of an unsupported stub"
    );
    assert!(
        webgpu_mod.contains("cubek_matmul::launch::launch_ref")
            || webgpu_gemm.contains("cubek_matmul") && webgpu_gemm.contains("launch_ref("),
        "WebGPU dot_general should route matmul through CubeK launch_ref"
    );
}

#[test]
fn webgpu_c32_dot_general_with_conj_uses_cubek_complex_api() {
    let webgpu_gemm =
        repo_file_if_exists("crates/tenferro-gpu/src/webgpu/gemm.rs").unwrap_or_default();
    let webgpu_kernels =
        repo_file_if_exists("crates/tenferro-gpu/src/webgpu/kernels.rs").unwrap_or_default();
    assert!(
        webgpu_gemm.contains("launch_c32_ref"),
        "WebGPU C32 matmul must route through CubeK complex GEMM API"
    );
    assert!(
        !webgpu_gemm.contains("compose_c32_from_products"),
        "tenferro WebGPU must not own complex GEMM compose lowering"
    );
    assert!(
        !webgpu_kernels.contains("compose_c32_parts_from_products"),
        "complex GEMM split/compose kernels belong in CubeK"
    );
}

#[test]
fn webgpu_provider_keeps_runtime_transfer_and_gemm_boundaries_split() {
    let webgpu_mod = repo_file("crates/tenferro-gpu/src/webgpu/mod.rs");

    for module in ["gemm", "kernels", "memory", "runtime"] {
        assert!(
            webgpu_mod.contains(&format!("mod {module};")),
            "WebGPU provider should keep `{module}` behind a dedicated module boundary"
        );
    }

    assert!(
        !webgpu_mod.contains("cubek_matmul::launch::launch_ref"),
        "CubeK matmul launch details should live in the WebGPU GEMM module"
    );
    assert!(
        !webgpu_mod.contains("pub struct WebGpuRuntime"),
        "WebGPU runtime initialization should live in the runtime module"
    );
    assert!(
        !webgpu_mod.contains("pub fn upload_webgpu_tensor")
            && !webgpu_mod.contains("pub fn download_webgpu_tensor"),
        "WebGPU transfer helpers should live in the memory module"
    );
}

#[test]
fn cubecl_output_allocations_use_checked_shape_products() {
    let dispatch = repo_file("crates/tenferro-gpu/src/cubecl/dispatch.rs");
    assert!(
        dispatch.contains("let len = checked_shape_product(\"cubecl_alloc_output\", shape)?;"),
        "CubeCL typed output allocation must reject shape-product overflow"
    );
    assert!(
        dispatch.contains("let len = checked_shape_product(\"cubecl_alloc_bool_output\", shape)?;"),
        "CubeCL bool output allocation must reject shape-product overflow"
    );
    assert!(
        !dispatch.contains("let len: usize = shape.iter().product();"),
        "CubeCL output allocation must not use unchecked shape.iter().product()"
    );
}

#[test]
fn cubecl_structural_shape_arithmetic_is_checked() {
    let cubecl_mod = repo_file("crates/tenferro-gpu/src/cubecl/mod.rs");
    assert!(
        cubecl_mod.contains("checked_dim_product(\"reshape\", \"input shape\", input.shape())?")
            && cubecl_mod.contains("checked_dim_product(\"reshape\", \"output shape\", shape)?"),
        "CubeCL reshape must reject shape-product overflow before reusing buffers"
    );
    assert!(
        cubecl_mod.contains("axis_extent = axis_extent.checked_add(input.shape()[dim])"),
        "CubeCL concatenate axis extent must use checked_add"
    );
    for banned in [
        "let old_n: usize = input.shape().iter().product();",
        "let new_n: usize = shape.iter().product();",
        "axis_extent += input.shape()[dim];",
    ] {
        assert!(
            !cubecl_mod.contains(banned),
            "CubeCL structural path must not use unchecked shape arithmetic: found {banned}"
        );
    }
}

#[test]
fn cubecl_gemm_contracting_element_product_is_checked() {
    let gemm = repo_file("crates/tenferro-gpu/src/cubecl/gemm.rs");
    assert!(
        gemm.contains("checked_mul(lhs.shape()[lhs_axis])"),
        "CubeCL GEMM must reject contracting dimension product overflow"
    );
    assert!(
        !gemm.contains("contracting_elements *= lhs.shape()[lhs_axis];"),
        "CubeCL GEMM must not use unchecked contracting dimension multiplication"
    );
}

#[test]
fn webgpu_allocation_uses_checked_shape_products() {
    let webgpu_mod = repo_file("crates/tenferro-gpu/src/webgpu/mod.rs");
    assert!(
        webgpu_mod.contains("checked_shape_product(op, shape)?"),
        "WebGPU output allocation must validate shape products with checked arithmetic"
    );
    assert!(
        !webgpu_mod.contains("shape.iter().product()"),
        "WebGPU output allocation must not use unchecked shape product arithmetic"
    );
}
