use std::{fs, path::Path};

fn linalg_source() -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("gpu")
            .join("linalg.rs"),
    )
    .unwrap_or_else(|err| panic!("GPU linalg source should be readable: {err}"))
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
            "CubeclBackend should not expose linalg-specific state: found {needle}"
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
fn gpu_solver_info_checks_are_batched_outside_kernel_loops() {
    let source = linalg_source();

    for (start, end, info_name, call_name) in [
        (
            "fn cholesky_typed",
            "fn triangular_solve_typed",
            "info = alloc_output::<i32>(backend.runtime(), &[batch_total])",
            "check_solver_info_tensor(backend.runtime(), &info, OP, \"cusolverDn*potrf\")",
        ),
        (
            "fn svd_typed",
            "fn svd_values_typed",
            "info = alloc_output::<i32>(backend.runtime(), &[batch_total])",
            "check_solver_info_tensor(backend.runtime(), &info, OP, \"cusolverDn*gesvd\")",
        ),
        (
            "fn svd_values_typed",
            "fn qr_typed",
            "info = alloc_output::<i32>(backend.runtime(), &[batch_total])",
            "check_solver_info_tensor(backend.runtime(), &info, OP, \"cusolverDn*gesvd\")",
        ),
        (
            "fn eigh_typed",
            "fn build_lu_outputs_device",
            "info = alloc_output::<i32>(backend.runtime(), &[batch_total])",
            "check_solver_info_tensor(backend.runtime(), &info, OP, \"cusolverDn*syevd\")",
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
        "geqrf_info = alloc_output::<i32>(backend.runtime(), &[batch_total])",
        "orgqr_info = alloc_output::<i32>(backend.runtime(), &[batch_total])",
        "check_solver_info_tensor(backend.runtime(), &geqrf_info, OP, \"cusolverDn*geqrf\")",
        "check_solver_info_tensor(backend.runtime(), &orgqr_info, OP, \"cusolverDn*orgqr\")",
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
            "check_solver_info_tensor(backend.runtime(), &info, OP, \"cusolverDn*gesvdj\")",
            "check_solver_info_tensor(backend.runtime(), &info, OP, \"cusolverDn*gesvd\")",
        ] {
            assert!(
                section.contains(needle),
                "GPU SVD driver path should contain {needle}"
            );
        }
    }

    for needle in [
        "let u = alloc_output::<T>(backend.runtime(), &u_shape)?;",
        "let v = alloc_output::<T>(backend.runtime(), &v_shape)?;",
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
        "T::copy_svd_v_to_vt(backend.runtime(), &v, &vt_shape, OP)",
        "copy_svd_v_to_vt_real",
        "copy_svd_v_to_vt_complex",
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
        "pub fn svd_v_to_vt_real",
        "pub fn svd_v_to_vt_complex",
        ".conj()",
    ] {
        assert!(
            kernels.contains(needle),
            "GPU SVD kernels should expose real and complex V-to-VT copies: missing {needle}"
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

    for (start, end, residency_check) in [
        (
            "fn cholesky_typed",
            "fn triangular_solve_typed",
            "ensure_cubecl_resident_typed(OP, input)?;",
        ),
        (
            "fn lu_typed",
            "fn svd_typed",
            "ensure_cubecl_resident_typed(OP, input)?;",
        ),
        (
            "fn svd_typed",
            "fn qr_typed",
            "ensure_cubecl_resident_typed(OP, input)?;",
        ),
        (
            "fn qr_typed",
            "fn eigh_typed",
            "ensure_cubecl_resident_typed(OP, input)?;",
        ),
        (
            "fn eigh_typed",
            "fn validate_nonsingular_gpu",
            "ensure_cubecl_resident_typed(OP, input)?;",
        ),
    ] {
        let section = source_section(&source, start, end);
        assert_before(section, residency_check, "if has_zero_dim");
    }

    let triangular = source_section(&source, "fn triangular_solve_typed_with_op", "fn lu_typed");
    assert_before(
        triangular,
        "ensure_cubecl_resident_typed(op, a)?;",
        "if has_zero_dim",
    );
    assert_before(
        triangular,
        "ensure_cubecl_resident_typed(op, b)?;",
        "if has_zero_dim",
    );

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
        solve.contains("zero_like_linalg_device_tensor(backend.runtime(), b, OP)"),
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
