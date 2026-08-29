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

#[cfg(feature = "cuda")]
#[test]
fn cuda_prepared_access_has_one_provider_handle_and_no_dead_request_metadata() {
    let dispatch = repo_file("crates/tenferro-gpu/src/cubecl/dispatch.rs");
    let prepared_start = dispatch
        .find("pub(crate) struct CubeclPreparedAccess")
        .expect("prepared access definition must exist");
    let prepared_end = dispatch[prepared_start..]
        .find("impl CubeclPreparedAccess")
        .map(|offset| prepared_start + offset)
        .expect("prepared access implementation must exist");
    let prepared = &dispatch[prepared_start..prepared_end];
    assert_eq!(
        prepared
            .matches("handle: cubecl_runtime::server::Handle")
            .count(),
        1,
        "prepared CUDA state must retain one provider handle"
    );
    assert!(
        !prepared.contains("binding: TensorBinding"),
        "prepared CUDA state must not retain a second binding owner"
    );

    let group = repo_file("crates/tenferro-tensor/src/storage/group.rs");
    assert!(
        !group.contains("prepare_device_read_for_layout_raw")
            && !group.contains("prepare_device_write_for_layout_raw")
            && !group.contains("DeviceAccessRequest::new"),
        "device preparation must use the single checked storage hierarchy"
    );

    let tensor_types = repo_file("crates/tenferro-tensor/src/types.rs");
    let request_start = tensor_types
        .find("pub struct DeviceAccessRequest")
        .expect("device request definition must exist");
    let request_end = tensor_types[request_start..]
        .find("/// Typed failure returned")
        .map(|offset| request_start + offset)
        .expect("device request boundary must exist");
    let request = &tensor_types[request_start..request_end];
    assert!(
        !request.contains("dtype:") && !request.contains("writable:"),
        "device requests must carry only metadata consumed by the root/provider seam"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_public_api_requires_typed_device_and_caller_selected_engine() {
    use tenferro_gpu::{
        cuda::cuda_runtime_engine_registration, cuda::CudaBackend, cuda::CudaDeviceError,
        cuda::CudaDeviceId, cuda::CudaRuntime,
    };
    use tenferro_runtime::{EngineId, EngineRegistration, RuntimeConfigError};

    let _: fn(CudaDeviceId) -> Result<CudaRuntime, CudaDeviceError> = CudaRuntime::new;
    let _: fn(CudaDeviceId) -> Result<CudaBackend, CudaDeviceError> = CudaBackend::new;
    let _: fn(&CudaBackend, EngineId) -> Result<EngineRegistration, RuntimeConfigError> =
        cuda_runtime_engine_registration;
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_prepared_access_has_one_state_and_one_downcast_boundary() {
    let dispatch = repo_file("crates/tenferro-gpu/src/cubecl/dispatch.rs");
    assert!(
        !dispatch.contains("writable: bool"),
        "prepared CUDA state must not retain a dead writable flag"
    );
    assert!(
        !dispatch.contains("offset: isize") && !dispatch.contains("element_size: usize"),
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
        lib_rs.contains("pub mod cuda"),
        "CUDA provider namespace should be the explicit low-level public module"
    );
    assert!(
        !lib_rs.contains("pub mod cuda_interop"),
        "the legacy flat CUDA interop module must not remain public"
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
    let webgpu_mod = repo_file("crates/tenferro-gpu/src/webgpu/mod.rs");
    assert!(
        lib_rs.contains("CudaBackend"),
        "CUDA backend should have an explicit public CudaBackend name"
    );
    assert!(
        webgpu_mod.contains("pub struct WebGpuBackend"),
        "WebGPU namespace should have an explicit public WebGpuBackend name"
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
        cuda_gemm.contains("cached_cutensor_contraction::<")
            && cuda_gemm.contains("alloc_workspace(backend.runtime(), cached.workspace_size)")
            && cuda_gemm.contains("Plan::new(cutensor, &op_desc"),
        "CUDA workspace and plan flow must remain visible through the cuTENSOR cache helper"
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
fn webgpu_materialization_does_not_inherit_host_defaults() {
    let webgpu_mod = repo_file("crates/tenferro-gpu/src/webgpu/mod.rs");
    let structural = repo_file("crates/tenferro-gpu/src/webgpu/structural.rs");
    assert!(
        webgpu_mod.contains("fn to_contiguous_read")
            && webgpu_mod.contains("structural::to_contiguous_read(self, input)"),
        "WebGPU materialization must delegate to its device-native structural module"
    );
    assert!(
        structural.contains("ensure_placement_resident_on_runtime")
            && structural.contains("view_array_arg")
            && !structural.contains("download_to_host")
            && !structural.contains("upload_host_tensor"),
        "WebGPU materialization must validate and consume resident device views without hidden transfer"
    );
    assert!(
        webgpu_mod.contains("fn copy_read_into")
            && webgpu_mod.contains("WebGpuBackend::copy_read_into"),
        "unsupported WebGPU copy-into must remain an explicit rejection instead of inheriting host defaults"
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
fn cubecl_copy_into_validates_both_views_on_the_active_runtime() {
    let cubecl_mod = repo_file("crates/tenferro-gpu/src/cubecl/mod.rs");
    let copy_body = cubecl_mod
        .split_once("fn copy_view_to_view_typed")
        .expect("CUDA copy-view helper must exist")
        .1
        .split_once("fn convert_float_to_float")
        .expect("CUDA copy-view helper must precede conversion helpers")
        .0;

    assert!(
        copy_body.contains("ensure_view_resident_on_runtime(self.runtime(), src, op)?;")
            && copy_body.contains("ensure_view_mut_resident_on_runtime(self.runtime(), dst, op)?;"),
        "CUDA copy_into must validate source and destination views against the active runtime"
    );
}

#[test]
fn provider_buffer_owners_are_scalar_independent_and_not_cloneable() {
    let cubecl_lib = repo_file("crates/tenferro-gpu/src/lib.rs");
    let webgpu_mod = repo_file("crates/tenferro-gpu/src/webgpu/mod.rs");
    for (name, source) in [("CubeCL", cubecl_lib), ("WebGPU", webgpu_mod)] {
        assert!(
            !source.contains("struct CubeclBuffer<T>")
                && !source.contains("struct WebGpuBuffer<T>"),
            "{name} provider owner must not be typed by scalar"
        );
        assert!(
            !source.contains("impl Clone for CubeclBuffer")
                && !source.contains("impl Clone for WebGpuBuffer"),
            "{name} provider owner must not expose a shallow clone"
        );
        assert!(
            !source.contains("allocation_domain: Option<AllocationDomainId>"),
            "{name} provider owner must require an allocation domain"
        );
    }
}

#[test]
fn provider_buffer_owners_store_bytes_and_derive_typed_lengths() {
    let cubecl_lib = repo_file("crates/tenferro-gpu/src/lib.rs");
    let webgpu_mod = repo_file("crates/tenferro-gpu/src/webgpu/mod.rs");
    for (name, source) in [("CubeCL", cubecl_lib), ("WebGPU", webgpu_mod)] {
        assert!(
            source.contains("byte_len: usize"),
            "{name} provider owner must identify physical storage in bytes"
        );
        assert!(
            source.contains("fn element_len<T: 'static>(&self)"),
            "{name} typed views must derive element counts from the borrowed scalar"
        );
    }
}

#[test]
fn cubecl_copy_into_checks_borrowed_backend_identity() {
    let cubecl_mod = repo_file("crates/tenferro-gpu/src/cubecl/mod.rs");
    let copy_body = cubecl_mod
        .split_once("fn copy_view_to_view_typed")
        .expect("CUDA copy-view helper must exist")
        .1
        .split_once("fn convert_float_to_float")
        .expect("CUDA copy-view helper must precede conversion helpers")
        .0;

    assert!(
        copy_body.contains("std::ptr::eq(source_buffer, destination_buffer)"),
        "CUDA copy_into must compare source and destination through borrowed backend identity"
    );
}

#[test]
fn cubecl_copy_into_reports_typed_shape_mismatch() {
    let cubecl_mod = repo_file("crates/tenferro-gpu/src/cubecl/mod.rs");
    let copy_body = cubecl_mod
        .split_once("fn copy_view_to_view_typed")
        .expect("CUDA copy-view helper must exist")
        .1
        .split_once("fn convert_float_to_float")
        .expect("CUDA copy-view helper must precede conversion helpers")
        .0;

    assert!(
        copy_body.contains("crate::Error::shape_mismatch(")
            && copy_body.contains("src.shape().to_vec()")
            && copy_body.contains("dst.shape().to_vec()"),
        "CUDA copy_into shape mismatch must use the shared typed error"
    );
}

#[test]
fn cubecl_runtime_materialization_and_copy_stay_device_owned_and_typed() {
    let cubecl_mod = repo_file("crates/tenferro-gpu/src/cubecl/mod.rs");
    let structural = cubecl_mod
        .split_once("impl TensorStructural for CudaBackend")
        .expect("CUDA structural implementation must exist")
        .1
        .split_once("impl TensorReduction for CudaBackend")
        .expect("CUDA structural implementation must precede reductions")
        .0;
    let runtime_methods = structural
        .split_once("fn to_contiguous_read(")
        .expect("CUDA runtime materialization override must exist")
        .1
        .split_once("fn transpose(")
        .expect("CUDA runtime methods must precede transpose")
        .0;

    for method in ["fn to_contiguous_read(", "fn copy_read_into("] {
        assert!(
            structural.contains(method),
            "CUDA TensorStructural must override {method}"
        );
    }
    assert!(
        runtime_methods.contains("to_contiguous_view_typed")
            && runtime_methods.contains("\"CudaBackend::to_contiguous_read\""),
        "CUDA erased materialization must reuse the typed path with its own operation name"
    );
    assert!(
        runtime_methods.contains("copy_view_to_view_typed")
            && runtime_methods.contains("\"CudaBackend::copy_read_into\""),
        "CUDA erased copy must reuse the typed path with its own operation name"
    );
    assert!(
        runtime_methods.contains("unsupported_dtype") && runtime_methods.contains("DType::Bool"),
        "CUDA Bool erased materialization/copy must remain explicit unsupported paths"
    );
    assert!(
        !runtime_methods.contains("download_tensor(")
            && !runtime_methods.contains("upload_tensor("),
        "CUDA runtime materialization/copy must not transfer payloads through host memory"
    );

    let copy_helper = cubecl_mod
        .split_once("fn copy_view_to_view_typed")
        .expect("CUDA copy-view helper must exist")
        .1
        .split_once("fn convert_float_to_float")
        .expect("CUDA copy-view helper must precede conversion helpers")
        .0;
    assert!(copy_helper.contains("ensure_view_resident_on_runtime"));
    assert!(copy_helper.contains("ensure_view_mut_resident_on_runtime"));
    assert!(copy_helper.contains("std::ptr::eq(source_buffer, destination_buffer)"));
    assert!(copy_helper.contains("typed_view_binding(src, op)?"));
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

#[test]
fn cubecl_gemm_contracting_element_product_is_checked() {
    let gemm = repo_file("crates/tenferro-gpu/src/cubecl/gemm.rs");
    assert!(
        gemm.contains("checked_mul(lhs_shape[lhs_axis])"),
        "CubeCL GEMM must reject contracting dimension product overflow"
    );
    assert!(
        !gemm.contains("contracting_elements *= lhs_shape[lhs_axis];"),
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
