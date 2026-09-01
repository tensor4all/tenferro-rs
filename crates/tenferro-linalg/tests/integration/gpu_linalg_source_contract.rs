use std::{fs, path::Path};

fn linalg_source() -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/linalg");
    let mut source = fs::read_to_string(root.with_extension("rs"))
        .unwrap_or_else(|err| panic!("GPU linalg source should be readable: {err}"));
    source.push_str(
        &fs::read_to_string(root.join("householder_qr.rs"))
            .unwrap_or_else(|err| panic!("GPU Householder QR source should be readable: {err}")),
    );
    source
}

fn gpu_ffi_source() -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("gpu")
            .join("ffi")
            .join("cusolver.rs"),
    )
    .unwrap_or_else(|err| panic!("GPU linalg FFI source should be readable: {err}"))
}

fn extension_source() -> String {
    fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/extension.rs"))
        .unwrap_or_else(|err| panic!("linalg extension source should be readable: {err}"))
}

fn tensor_ext_source() -> String {
    fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/tensor_ext.rs"))
        .unwrap_or_else(|err| panic!("tensor extension source should be readable: {err}"))
}

fn gpu_mod_source() -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("gpu")
            .join("mod.rs"),
    )
    .unwrap_or_else(|err| panic!("GPU linalg module source should be readable: {err}"))
}

fn workspace_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("tenferro-linalg should live inside the workspace")
}

fn read_workspace_source(path: &str) -> String {
    fs::read_to_string(workspace_root().join(path))
        .unwrap_or_else(|err| panic!("workspace source {path} should be readable: {err}"))
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

fn assert_before(section: &str, earlier: &str, later: &str) {
    let earlier_idx = section
        .find(earlier)
        .unwrap_or_else(|| panic!("source section should contain {earlier:?}"));
    let later_idx = section
        .find(later)
        .unwrap_or_else(|| panic!("source section should contain {later:?}"));
    assert!(
        earlier_idx < later_idx,
        "{earlier:?} should appear before {later:?}"
    );
}

fn assert_unsafe_blocks_have_safety_comments(path: &str, source: &str) {
    let lines: Vec<_> = source.lines().collect();
    let mut missing = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        if !line.contains("unsafe {") {
            continue;
        }
        let window_start = idx.saturating_sub(3);
        let has_safety = lines[window_start..idx]
            .iter()
            .any(|candidate| candidate.trim_start().starts_with("// SAFETY:"));
        if !has_safety {
            missing.push(format!("{}:{}", path, idx + 1));
        }
    }
    assert!(
        missing.is_empty(),
        "GPU linalg unsafe blocks need local SAFETY comments:\n{}",
        missing.join("\n")
    );
}

#[test]
fn gpu_linalg_unsafe_blocks_document_safety_invariants() {
    assert_unsafe_blocks_have_safety_comments("src/gpu/linalg.rs", &linalg_source());
}

#[test]
fn compact_householder_cuda_uses_incremental_device_native_paths() {
    let source = linalg_source();
    let append = source_section(
        &source,
        "fn compact_qr_append_typed",
        "fn compact_qr_state_dims",
    );
    assert!(append.contains("apply_householder_reflectors_typed"));
    assert!(append.contains("geqrf_trailing_typed"));
    assert!(!append.contains("\n    qr_typed("));
    assert!(!append.contains("download_tensor"));

    let apply = source_section(
        &source,
        "fn apply_householder_reflectors_typed",
        "fn geqrf_trailing_typed",
    );
    assert!(apply.contains("build_explicit_v"));
    assert!(apply.contains("larft_buffer_size"));
    assert!(apply.contains(".larft("));
    assert_eq!(apply.matches(".gemm(").count(), 3);
    assert!(!apply.contains("for reflector"));
    assert!(!apply.contains("copy_bytes"));
    assert!(source.contains("householder_explicit_v::launch_unchecked"));

    let from_factors = source_section(
        &source,
        "fn compact_qr_from_factors_typed",
        "fn compact_qr_append_typed",
    );
    assert!(from_factors.contains("gemm_nn_typed"));
    assert!(from_factors.contains("assemble_from_factors_typed"));
    assert!(!from_factors.contains("\n    qr_typed("));
    assert!(!from_factors.contains("download_tensor"));

    let ffi = gpu_ffi_source();
    for symbol in [
        "cusolverDnCreateParams",
        "cusolverDnDestroyParams",
        "cusolverDnXlarft_bufferSize",
        "cusolverDnXlarft",
        "cublasSgemm_v2",
        "cublasDgemm_v2",
        "cublasCgemm_v2",
        "cublasZgemm_v2",
        "cublasSetPointerMode_v2",
    ] {
        assert!(ffi.contains(symbol), "missing CUDA FFI symbol {symbol}");
    }
}

#[test]
fn qr_options_routes_owned_read_and_typed_surfaces_through_backend_hooks() {
    let backend = read_workspace_source("tenferro-linalg/src/backend.rs");
    assert!(backend.contains("fn qr_with_options_read("));
    let tensor_ext = tensor_ext_source();
    assert!(tensor_ext.contains("backend.qr_with_options_read("));
    assert!(!tensor_ext.contains("apply_qr_gauge"));
    let gpu = gpu_mod_source();
    assert!(gpu.contains("fn qr_with_options("));
    assert!(gpu.contains("fn qr_with_options_read("));
    let extension = extension_source();
    let cuda_admission = source_section(
        &extension,
        "if type_id == std::any::TypeId::of::<tenferro_gpu::cuda::CudaBackend>()",
        "false\n}",
    );
    assert!(!cuda_admission.contains("HouseholderQrFactor"));
}

#[test]
fn tenferro_gpu_no_longer_owns_linalg_specific_ffi_or_kernels() {
    for path in [
        "tenferro-gpu/src/cubecl/ffi/cusolver.rs",
        "tenferro-gpu/src/kernels/linalg.rs",
    ] {
        assert!(
            !workspace_root().join(path).exists(),
            "{path} should be owned by tenferro-linalg, not tenferro-gpu"
        );
    }

    let cubecl_mod = read_workspace_source("tenferro-gpu/src/cubecl/mod.rs");
    for needle in ["CudaLinalgHandles", "linalg_handles", "cusolver", "cublas"] {
        assert!(
            !cubecl_mod.contains(needle),
            "CudaBackend should not expose linalg-specific state: found {needle}"
        );
    }
}

#[test]
fn linalg_ad_rules_use_internal_ops_conjugation_helpers() {
    let source = read_workspace_source("tenferro-linalg/src/ad/rules/mod.rs");
    for needle in [
        "fn is_real_dtype",
        "fn conjugate_primal_if_dtype_complex",
        "fn conjugate_linear_if_dtype_complex",
    ] {
        assert!(
            !source.contains(needle),
            "linalg AD should import canonical helper instead of redefining {needle}"
        );
    }
}

#[test]
fn gpu_lu_outputs_are_not_rebuilt_by_host_roundtrip() {
    let source = linalg_source();
    let lu_source = source_section(&source, "fn lu_typed", "fn svd_typed");
    let banned = [
        "download_device_tensor(backend.runtime(), &work, OP)",
        "build_lu_outputs_host(&host_lu",
        "upload_host_tensor(backend.runtime(), p)",
        "upload_host_tensor(backend.runtime(), l)",
        "upload_host_tensor(backend.runtime(), u)",
        "upload_host_tensor(backend.runtime(), parity)",
    ];

    let mut violations = Vec::new();
    for needle in banned {
        if lu_source.contains(needle) {
            violations.push(format!("gpu/linalg.rs lu_typed contains {needle}"));
        }
    }

    assert!(
        violations.is_empty(),
        "GPU LU must not rebuild P/L/U/parity through a full device-to-host-to-device roundtrip:\n{}",
        violations.join("\n")
    );
}

#[test]
fn gpu_triangular_solve_batched_offsets_use_checked_arithmetic() {
    let source = linalg_source();
    let triangular = source_section(
        &source,
        "fn triangular_solve_typed_with_op",
        "fn solve_typed",
    );

    for needle in [
        "checked_mul_usize(op, \"triangular matrix stride\", n, n)",
        "checked_mul_usize(op, \"triangular rhs stride\", rows, cols)",
        "checked_batch_offset(op, \"triangular matrix batch offset\", batch, a_stride)",
        "checked_batch_offset(op, \"triangular rhs batch offset\", batch, out_stride)",
    ] {
        assert!(
            triangular.contains(needle),
            "triangular_solve_typed_with_op should use checked arithmetic: missing {needle}"
        );
    }
}

#[test]
fn gpu_linalg_batched_pointer_offsets_use_checked_arithmetic_contract() {
    let source = linalg_source();

    assert!(
        !source.contains("batch *"),
        "GPU batched linalg pointer offsets must go through checked_batch_offset"
    );
    for banned in [
        "let matrix_stride = n * n",
        "let matrix_stride = m * n",
        "let a_stride = m * n",
        "let u_stride = m * k",
        "let vt_stride = k * n",
        "let v_stride = n * k",
        "let q_stride = m * k",
    ] {
        assert!(
            !source.contains(banned),
            "GPU linalg stride products must use checked_mul_usize: found {banned}"
        );
    }
}

#[test]
fn gpu_zero_sized_lu_factor_parity_is_filled_on_device() {
    let source = linalg_source();
    let zero_lu = source_section(&source, "fn zero_sized_lu_factor_outputs", "fn raw_stream");
    let kernels = read_workspace_source("tenferro-linalg/src/gpu/kernels.rs");

    for needle in [
        "upload_host_tensor",
        "vec![T::one()",
        "TypedTensor::from_vec_col_major",
    ] {
        assert!(
            !zero_lu.contains(needle),
            "zero-sized GPU LU factor parity should not build host tensors: found {needle}"
        );
    }

    assert!(
        zero_lu.contains("fill_one_device_tensor"),
        "zero-sized GPU LU factor parity should be initialized by a device fill helper"
    );
    assert!(
        kernels.contains("pub fn fill_one_kernel"),
        "GPU linalg kernels should expose a one-fill kernel for scalar/empty-shape fast paths"
    );
}

#[test]
fn gpu_solve_uses_packed_lu_without_public_lu_materialization() {
    let source = linalg_source();
    let solve_source = source_section(&source, "pub(super) fn solve", "fn cholesky_typed");
    let banned = [
        "let outputs = lu(backend, a)?;",
        "let p = &outputs[0];",
        "let l = &outputs[1];",
        "let u = &outputs[2];",
        "matmul_preserve_trailing_batch(backend, p, &rhs)?",
    ];

    let mut violations = Vec::new();
    for needle in banned {
        if solve_source.contains(needle) {
            violations.push(format!("gpu/linalg.rs solve contains {needle}"));
        }
    }

    assert!(
        violations.is_empty(),
        "GPU solve must consume packed LU factors directly instead of materializing public P/L/U outputs:\n{}",
        violations.join("\n")
    );
    assert!(
        solve_source.contains("lu_factor("),
        "GPU solve should factor into packed LU with pivots"
    );
    assert!(
        solve_source.contains("lu_solve_prepared("),
        "GPU solve should use the prepared LU solve path"
    );
}

#[test]
fn cuda_linalg_drop_paths_report_destroy_status() {
    let source = read_workspace_source("tenferro-linalg/src/gpu/ffi/cusolver.rs");
    for banned in [
        "let _ = unsafe { (self.lib.vtable.destroy)(self.raw) };",
        "(self.handle.lib.vtable.destroy_gesvdj_info)(self.raw);",
    ] {
        assert!(
            !source.contains(banned),
            "CUDA linalg Drop paths must inspect destroy status instead of discarding it: found {banned}"
        );
    }
    let handle_drop = source_section(
        &source,
        "impl Drop for CusolverDnHandle",
        "pub struct CublasHandle",
    );
    assert_before(handle_drop, "destroy_params", "vtable.destroy)(self.raw)");
    assert!(source.contains("if let Err(error) = lib.check_status"));
    assert!(source.contains("let destroy_status = unsafe { (lib.vtable.destroy)(raw) }"));

    for helper in [
        "report_cusolver_destroy_status",
        "report_cublas_destroy_status",
    ] {
        assert!(
            source.contains(helper),
            "CUDA linalg Drop paths should report non-success destroy statuses through {helper}"
        );
    }
}

#[test]
fn cubecl_linalg_overrides_svd_read_with_backend_canonicalization() {
    let source = gpu_mod_source();
    let svd_read_source = source_section(&source, "fn svd_read", "fn qr");

    for needle in [
        "self.to_contiguous(&view)?",
        "let input = Tensor::F64(compact);",
        "self.svd(&input)",
    ] {
        assert!(
            svd_read_source.contains(needle),
            "CubeCL svd_read should canonicalize borrowed GPU views on the backend: missing {needle}"
        );
    }
}

#[test]
fn cubecl_linalg_overrides_qr_read_with_backend_canonicalization() {
    let source = gpu_mod_source();
    let qr_read_source = source_section(&source, "fn qr_read", "fn eigh");

    for needle in [
        "self.to_contiguous(&view)?",
        "let input = Tensor::F64(compact);",
        "self.qr(&input)",
    ] {
        assert!(
            qr_read_source.contains(needle),
            "CubeCL qr_read should canonicalize borrowed GPU views on the backend: missing {needle}"
        );
    }
}

#[test]
fn cubecl_linalg_overrides_eigh_read_with_backend_canonicalization() {
    let source = gpu_mod_source();
    let eigh_read_source = source_section(&source, "fn eigh_read", "fn eigh_values");

    for needle in [
        "self.to_contiguous(&view)?",
        "let input = Tensor::F64(compact);",
        "self.eigh(&input)",
    ] {
        assert!(
            eigh_read_source.contains(needle),
            "CubeCL eigh_read should canonicalize borrowed GPU views on the backend: missing {needle}"
        );
    }
}

#[test]
fn gpu_solver_info_checks_are_batched_outside_kernel_loops() {
    let source = linalg_source();

    for (start, end, info_name, call_name) in [
        (
            "fn cholesky_typed",
            "fn triangular_solve_typed",
            "let mut info = raw.alloc_output::<i32>(&[batch_total])?;",
            "raw.download_tensor::<i32>(&info, OP)?",
        ),
        (
            "fn svd_typed",
            "fn svd_values_typed",
            "let mut info = raw.alloc_output::<i32>(&[batch_total])?;",
            "raw.download_tensor::<i32>(&info, OP)?",
        ),
        (
            "fn svd_values_typed",
            "fn qr_typed",
            "let mut info = raw.alloc_output::<i32>(&[batch_total])?;",
            "raw.download_tensor::<i32>(&info, OP)?",
        ),
        (
            "fn eigh_typed",
            "fn build_lu_outputs_device",
            "let mut info = raw.alloc_output::<i32>(&[batch_total])?;",
            "raw.download_tensor::<i32>(&info, OP)?",
        ),
    ] {
        let section = source_section(&source, start, end);
        assert!(
            section.contains(info_name),
            "{start} should allocate one solver-info tensor for the whole batch"
        );
        assert!(
            section.contains(call_name),
            "{start} should check solver info once after the batch loop"
        );
        assert!(
            !section.contains("copy_device_to_host"),
            "{start} should not synchronize inside the per-batch loop"
        );
    }

    let qr = source_section(&source, "fn qr_typed", "fn eigh_typed");
    for needle in [
        "geqrf_info = raw.alloc_output::<i32>",
        "orgqr_info = raw.alloc_output::<i32>",
        "raw.download_tensor::<i32>(&geqrf_info, OP)?",
        "raw.download_tensor::<i32>(&orgqr_info, OP)?",
    ] {
        assert!(
            qr.contains(needle),
            "QR should use batched info handling: missing {needle}"
        );
    }
    assert!(
        !qr.contains("copy_device_to_host"),
        "QR should not synchronize inside the per-batch loop"
    );
}

#[test]
fn gpu_solve_paths_validate_residency_before_dtype_and_zero_fast_paths() {
    let source = linalg_source();
    let solve = source_section(
        &source,
        "pub(super) fn solve",
        "pub(super) fn lu_solve_prepared",
    );
    let lu_solve_prepared = source_section(
        &source,
        "pub(super) fn lu_solve_prepared",
        "fn cholesky_typed",
    );

    for needle in [
        "ensure_cubecl_resident_tensor(OP, a)?",
        "ensure_cubecl_resident_tensor(OP, b)?",
    ] {
        assert_before(solve, needle, "ensure_supported_linalg_pair(OP, a, b)?");
        assert_before(
            solve,
            needle,
            "if has_zero_dim(a.shape()) || has_zero_dim(b.shape())",
        );
    }

    for needle in [
        "ensure_cubecl_resident_tensor(OP, a)?",
        "ensure_cubecl_resident_tensor(OP, packed_lu)?",
        "ensure_cubecl_resident_tensor(OP, pivots)?",
        "ensure_cubecl_resident_tensor(OP, b)?",
    ] {
        assert_before(
            lu_solve_prepared,
            needle,
            "ensure_supported_linalg_pair(OP, a, b)?",
        );
        assert_before(
            lu_solve_prepared,
            needle,
            "ensure_supported_linalg_pair(OP, a, packed_lu)?",
        );
        assert_before(
            lu_solve_prepared,
            needle,
            "if !matches!(pivots, Tensor::I32(_))",
        );
        assert_before(
            lu_solve_prepared,
            needle,
            "if has_zero_dim(a.shape()) || has_zero_dim(b.shape())",
        );
    }
}

#[test]
fn gpu_svd_uses_jax_compatible_default_driver_selection() {
    let source = linalg_source();
    let svd = source_section(&source, "fn svd_typed", "fn svd_values_typed");
    let svd_values = source_section(&source, "fn svd_values_typed", "fn qr_typed");
    let ffi = read_workspace_source("tenferro-linalg/src/gpu/ffi/cusolver.rs");
    let kernels = read_workspace_source("tenferro-linalg/src/gpu/kernels.rs");

    for needle in [
        "const JAX_COMPATIBLE_GESVDJ_MAX_DIM: usize = 1024",
        "enum SvdDriver",
        "fn select_svd_driver",
        "m <= JAX_COMPATIBLE_GESVDJ_MAX_DIM && n <= JAX_COMPATIBLE_GESVDJ_MAX_DIM",
    ] {
        assert!(
            source.contains(needle),
            "GPU SVD should encode JAX-compatible default driver selection: missing {needle}"
        );
    }

    for section in [svd, svd_values] {
        for needle in [
            "match select_svd_driver(m, n)",
            "SvdDriver::Gesvdj",
            "SvdDriver::Gesvd",
            "handles.cusolver().gesvdj(",
            "handles.cusolver().gesvd(",
            "check_solver_info(OP, \"cusolverDn*gesvdj\"",
            "check_solver_info(OP, \"cusolverDn*gesvd\"",
        ] {
            assert!(
                section.contains(needle),
                "GPU SVD driver path should contain {needle}"
            );
        }
    }

    for needle in [
        "let mut u = raw.alloc_output::<T>(&u_shape)?;",
        "let mut v = raw.alloc_output::<T>(&v_shape)?;",
        "CusolverEigMode::NoVector",
        "batch_u",
        "batch_v",
    ] {
        assert!(
            svd_values.contains(needle),
            "values-only gesvdj should pass scratch U/V buffers like JAX: missing {needle}"
        );
    }

    for needle in [
        "T::copy_matrix_adjoint(backend, &v, &vt_shape, OP)",
        "copy_matrix_adjoint_real",
        "copy_matrix_adjoint_complex",
    ] {
        assert!(
            source.contains(needle),
            "GPU SVD should materialize gesvdj V^H on CUDA without a host roundtrip: missing {needle}"
        );
    }
    assert!(
        !source.contains("runtime_typed_tensor::transpose"),
        "GPU SVD should not route gesvdj V-to-VT conversion through borrowed-view typed helpers"
    );

    for needle in [
        "pub fn matrix_adjoint_real",
        "pub fn matrix_adjoint_complex",
        ".conj()",
    ] {
        assert!(
            kernels.contains(needle),
            "GPU SVD kernels should expose real and complex V-to-VT copies: missing {needle}"
        );
    }

    for needle in [
        "let transpose_for_gesvd = m < n;",
        "T::copy_matrix_adjoint(backend, input, &work_shape, OP)?",
        "handles.cusolver().gesvd_buffer_size(",
        "gesvd_m_i32",
        "gesvd_n_i32",
        "T::copy_matrix_adjoint(backend, &gesvd_vt, &u_shape, OP)?",
        "T::copy_matrix_adjoint(backend, &gesvd_u, &vt_shape, OP)?",
    ] {
        assert!(
            svd.contains(needle),
            "wide factor gesvd should solve the device adjoint and map factors back: missing {needle}"
        );
    }
    for needle in [
        "let (gesvd_m, gesvd_n) = if m < n { (n, m) } else { (m, n) }",
        "T::copy_matrix_adjoint(backend, input, &work_shape, OP)?",
        "handles.cusolver().gesvd_buffer_size(",
        "gesvd_m_i32",
        "gesvd_n_i32",
    ] {
        assert!(
            svd_values.contains(needle),
            "wide values-only gesvd should solve the device adjoint: missing {needle}"
        );
    }
    // The wide gesvd orientation must never download factor data (u/s/v) to
    // the host. The only host read is the solver `i32` info diagnostic via
    // the raw-session `download_tensor::<i32>`; factor-dtype downloads are
    // banned.
    for banned in [
        "download_tensor::<T>",
        "download_tensor::<<T as LinalgScalar>::Real>",
        "download_typed_tensor",
    ] {
        assert!(
            !svd.contains(banned) && !svd_values.contains(banned),
            "wide gesvd orientation must not download factor data: found {banned}"
        );
    }

    for needle in [
        "pub struct GesvdjInfo",
        "cusolverDnCreateGesvdjInfo",
        "cusolverDnDestroyGesvdjInfo",
        "cusolverDnSgesvdj_bufferSize",
        "cusolverDnDgesvdj_bufferSize",
        "cusolverDnCgesvdj_bufferSize",
        "cusolverDnZgesvdj_bufferSize",
        "cusolverDnSgesvdj",
        "cusolverDnDgesvdj",
        "cusolverDnCgesvdj",
        "cusolverDnZgesvdj",
        "pub fn gesvdj_buffer_size",
        "pub unsafe fn gesvdj",
    ] {
        assert!(
            ffi.contains(needle),
            "cuSOLVER FFI should expose gesvdj support: missing {needle}"
        );
    }
}

#[test]
fn gpu_eigh_values_uses_cusolver_no_vector_mode() {
    let source = linalg_source();
    let gpu_mod = gpu_mod_source();
    let ffi = read_workspace_source("tenferro-linalg/src/gpu/ffi/cusolver.rs");

    assert!(
        source.contains("pub(super) fn eigh_values"),
        "GPU linalg should expose an internal eigh_values hook"
    );
    assert!(
        source.contains("CusolverEigMode::NoVector"),
        "GPU eigvalsh should use cuSOLVER values-only mode"
    );
    assert!(
        gpu_mod.contains("fn eigh_values"),
        "CubeCL linalg backend should override eigh_values"
    );
    assert!(
        ffi.contains("NoVector = 0"),
        "cuSOLVER eig mode enum should expose values-only mode"
    );
}

#[test]
fn gpu_triangular_solve_uses_batched_cublas_when_batching_is_present() {
    let source = linalg_source();
    let ffi = read_workspace_source("tenferro-linalg/src/gpu/ffi/cusolver.rs");
    let triangular = source_section(&source, "fn triangular_solve_typed_with_op", "fn lu_typed");

    assert!(
        ffi.contains("trsm_batched"),
        "cuBLAS FFI should expose a batched triangular solve entry point"
    );
    assert!(
        triangular.contains("trsm_batched"),
        "GPU triangular_solve should use the batched entry point for batched inputs"
    );
}

#[test]
fn gpu_linalg_zero_dim_fast_paths_validate_residency_before_allocating_outputs() {
    let source = linalg_source();

    // Every op validates input residency before the zero-dim allocation fast
    // path. The guard is either the standalone typed residency check before
    // `if has_zero_dim` (LU path) or a `raw.tensor(input)?` residency probe
    // inside the `with_raw` fast-path closure before `raw.alloc_output`.
    for (start, end, check, later) in [
        (
            "fn cholesky_typed",
            "fn triangular_solve_typed",
            "raw.tensor(input)?;",
            "raw.alloc_output",
        ),
        (
            "fn lu_typed",
            "fn svd_typed",
            "ensure_cubecl_resident_typed(OP, input)?;",
            "if has_zero_dim",
        ),
        (
            "fn svd_typed",
            "fn qr_typed",
            "raw.tensor(input)?;",
            "raw.alloc_output",
        ),
        (
            "fn qr_typed",
            "fn eigh_typed",
            "ensure_cubecl_resident_typed(OP, input)?;",
            "if has_zero_dim",
        ),
        (
            "fn eigh_typed",
            "fn validate_nonsingular_gpu",
            "ensure_cubecl_resident_typed(OP, input)?;",
            "if has_zero_dim",
        ),
    ] {
        let section = source_section(&source, start, end);
        assert_before(section, check, later);
    }

    let triangular = source_section(&source, "fn triangular_solve_typed_with_op", "fn lu_typed");
    // The standalone residency guards run before the fast path flag is
    // computed, and the raw-session fast path re-probes residency via
    // `raw.tensor` before allocating the empty output.
    assert_before(
        triangular,
        "ensure_cubecl_resident_typed(op, a)?;",
        "let zero_dim = has_zero_dim(a.shape()) || has_zero_dim(b.shape());",
    );
    assert_before(
        triangular,
        "ensure_cubecl_resident_typed(op, b)?;",
        "let zero_dim = has_zero_dim(a.shape()) || has_zero_dim(b.shape());",
    );
    assert_before(triangular, "raw.tensor(a)?;", "raw.alloc_output");

    let solve = source_section(&source, "pub(super) fn solve", "fn cholesky_typed");
    assert_before(
        solve,
        "ensure_supported_linalg_pair(OP, a, b)?;",
        "if has_zero_dim",
    );
    assert_before(
        solve,
        "ensure_cubecl_resident_tensor(OP, a)?;",
        "if has_zero_dim",
    );
    assert_before(
        solve,
        "ensure_cubecl_resident_tensor(OP, b)?;",
        "if has_zero_dim",
    );
    assert!(
        solve.contains("zero_like_linalg_device_tensor(backend, b, OP)"),
        "GPU solve zero-dim fast path should allocate the result on the GPU"
    );
    assert!(
        !source.contains("fn zeros_like_tensor"),
        "GPU linalg should not build host zero tensors for device fast paths"
    );
}

#[test]
fn gpu_validate_nonsingular_synchronizes_before_host_download() {
    let source = linalg_source();
    let validate = source_section(
        &source,
        "let min_val = backend.reduce_min(&flat, &[0])?;",
        "let is_singular = match &host_min",
    );

    assert_before(
        validate,
        "backend.runtime().synchronize()?;",
        "download_tensor(backend.runtime(), &min_val)?;",
    );
}

#[test]
fn gpu_validate_nonsingular_uses_complex_magnitude_and_tolerance() {
    let source = linalg_source();
    let validate = source_section(
        &source,
        "fn validate_nonsingular_gpu",
        "fn singularity_tolerance",
    );
    let kernels = read_workspace_source("tenferro-linalg/src/gpu/kernels.rs");

    assert!(
        validate.contains("diagonal_magnitude(backend, &diag)?"),
        "GPU singularity validation should compute diagonal magnitudes through a dedicated helper"
    );
    assert!(
        !validate.contains("backend.cast(&diag, DType::F64)"),
        "GPU singularity validation must not cast complex diagonals to real and discard imaginary parts"
    );
    assert!(
        validate.contains("let max_val = backend.reduce_max(&flat, &[0])?;"),
        "GPU singularity validation should compute max diagonal magnitude for a scaled tolerance"
    );
    assert!(
        source.contains("fn singularity_tolerance(dtype: DType, max_magnitude: f64) -> f64"),
        "GPU singularity validation should use a dtype-aware tolerance helper"
    );
    assert!(
        validate.contains("value <= tolerance"),
        "GPU singularity validation should reject near-zero pivots with an epsilon-scaled tolerance"
    );

    for needle in [
        "pub fn complex32_magnitude",
        "pub fn complex64_magnitude",
        ".abs()",
    ] {
        assert!(
            kernels.contains(needle),
            "GPU linalg kernels should compute complex magnitude on device: missing {needle}"
        );
    }
}

#[test]
fn gpu_lu_shape_extent_k_is_runtime_not_compile_time_specialized() {
    let source = read_workspace_source("tenferro-linalg/src/gpu/kernels.rs");
    let lu_kernels = source_section(
        &source,
        "#[cube(launch_unchecked)]\npub fn lu_extract_outputs",
        "fn zero_sized_lu_factor_outputs",
    );

    assert!(
        !lu_kernels.contains("#[comptime] k: usize"),
        "LU kernels must not specialize on matrix-size extent k"
    );
    assert!(
        !lu_kernels.contains("#[unroll]\n        for step in 0usize..k"),
        "LU kernels must not unroll loops over matrix-size extent k"
    );
    assert!(
        !lu_kernels.contains("#[unroll]\n            for step in 0usize..k"),
        "LU kernels must not unroll nested loops over matrix-size extent k"
    );
    assert!(
        !lu_kernels.contains("#[unroll]\n            for offset in 0usize..k"),
        "LU kernels must not unroll reverse loops over matrix-size extent k"
    );

    for needle in ["while step < k", "while step > 0usize"] {
        assert!(
            lu_kernels.contains(needle),
            "LU kernels should iterate over runtime k with {needle}"
        );
    }
}

#[test]
fn cubecl_linalg_overrides_cholesky_read_with_backend_canonicalization() {
    let source = gpu_mod_source();
    let cholesky_read_source = source_section(&source, "fn cholesky_read", "fn lu_read");

    for needle in [
        "self.to_contiguous(&view)?",
        "let input = Tensor::F64(compact);",
        "self.cholesky(&input)",
    ] {
        assert!(
            cholesky_read_source.contains(needle),
            "CubeCL cholesky_read should canonicalize borrowed GPU views on the backend: missing {needle}"
        );
    }
}

#[test]
fn cubecl_linalg_overrides_lu_read_with_backend_canonicalization() {
    let source = gpu_mod_source();
    let lu_read_source = source_section(&source, "fn lu_read", "fn full_piv_lu_read");

    for needle in [
        "self.to_contiguous(&view)?",
        "let input = Tensor::F64(compact);",
        "self.lu(&input)",
    ] {
        assert!(
            lu_read_source.contains(needle),
            "CubeCL lu_read should canonicalize borrowed GPU views on the backend: missing {needle}"
        );
    }
}

#[test]
fn cubecl_linalg_overrides_full_piv_lu_read_with_backend_canonicalization() {
    let source = gpu_mod_source();
    let full_piv_lu_read_source = source_section(&source, "fn full_piv_lu_read", "fn eig_read");

    for needle in [
        "self.to_contiguous(&view)?",
        "let input = Tensor::F64(compact);",
        "self.full_piv_lu(&input)",
    ] {
        assert!(
            full_piv_lu_read_source.contains(needle),
            "CubeCL full_piv_lu_read should canonicalize borrowed GPU views on the backend: missing {needle}"
        );
    }
}

#[test]
fn cubecl_linalg_overrides_eig_read_with_backend_canonicalization() {
    let source = gpu_mod_source();
    let eig_read_source = source_section(&source, "fn eig_read", "fn eigh_values");

    for needle in [
        "self.to_contiguous(&view)?",
        "let input = Tensor::F64(compact);",
        "self.eig(&input)",
    ] {
        assert!(
            eig_read_source.contains(needle),
            "CubeCL eig_read should canonicalize borrowed GPU views on the backend: missing {needle}"
        );
    }
}
