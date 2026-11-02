use super::cusolver::{xsyev_batched_available, CudaDataType, CusolverVtable};

/// The cuSOLVER symbols `cusolver.rs` resolves, read back from the loader module
/// itself so a stub library cannot drift from the vtable (#2413).
fn cusolver_symbol_names() -> Vec<String> {
    let source = include_str!("cusolver.rs");
    let mut names = Vec::new();
    let mut rest = source;
    while let Some(start) = rest.find("b\"cusolver") {
        rest = &rest[start + 2..];
        let end = rest
            .find('"')
            .expect("unterminated cuSOLVER symbol literal");
        names.push(rest[..end].trim_end_matches("\\0").to_owned());
        rest = &rest[end..];
    }
    names
}

#[test]
fn vtable_opens_when_the_batched_eigh_symbols_are_absent() {
    // #2413: cuSOLVER older than 11.7.1 exports no `cusolverDnXsyevBatched*`.
    // That must cost batched eigh only, not `geqrf`/`orgqr`/`syevd`/`gesvd` and
    // every other routine that shares the vtable. A stub library exporting
    // every symbol except those two reproduces the host without a GPU.
    const MISSING: [&str; 2] = [
        "cusolverDnXsyevBatched",
        "cusolverDnXsyevBatched_bufferSize",
    ];
    let mut code = String::new();
    for name in cusolver_symbol_names() {
        if !MISSING.contains(&name.as_str()) {
            code.push_str(&format!(
                "#[no_mangle]\npub extern \"C\" fn {name}() {{}}\n"
            ));
        }
    }

    // Build beside the test binary rather than in `temp_dir()`: that directory
    // is inside the target dir, so a `noexec` `/tmp` cannot break the dlopen.
    let executable = std::env::current_exe().expect("locate the test binary");
    let dir = executable
        .parent()
        .expect("the test binary has a directory")
        .join(format!("cusolver-stub-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create the stub library directory");
    let source = dir.join("cusolver_stub.rs");
    std::fs::write(&source, code).expect("write the stub library source");
    let library_path = dir.join("libcusolver_stub.so");
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
    let status = std::process::Command::new(&rustc)
        .args([
            "--crate-type=cdylib",
            "--edition=2021",
            "--crate-name=cusolver_stub",
            "-o",
        ])
        .arg(&library_path)
        .arg(&source)
        .status()
        .expect("run rustc to build the stub library");
    assert!(status.success(), "the stub library did not build");

    // SAFETY: the path names the shared object this test just wrote.
    let library =
        unsafe { libloading::Library::new(&library_path) }.expect("the stub cuSOLVER must open");
    // SAFETY: every symbol `load` resolves is declared by the stub except the
    // two batched eigh entries, which is the state under test.
    let vtable = unsafe { CusolverVtable::load(&library) }
        .expect("a cuSOLVER without batched eigh must still load");
    assert!(
        vtable.per_matrix_linear_algebra_resolved(),
        "geqrf, orgqr, syevd and gesvd must stay available"
    );
    assert!(
        !vtable.batched_eigh_resolved(),
        "the two batched eigh symbols are absent from the stub"
    );
    assert!(
        !xsyev_batched_available(12_000, vtable.batched_eigh_resolved()),
        "batched eigh stays unavailable however new the library claims to be"
    );

    drop(library);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn batched_eigh_capability_follows_the_jax_version_boundary() {
    // cuSOLVER 11.7.1 (CUDA 12.6 Update 2) is the first release exporting
    // `cusolverDnXsyevBatched`; JAX's `IsSyevBatchedSupported` gates on the
    // same `version >= 11701`.
    assert!(xsyev_batched_available(11_701, true));
    assert!(xsyev_batched_available(12_000, true));
    assert!(!xsyev_batched_available(11_700, true));
    assert!(!xsyev_batched_available(0, true));
    assert!(!xsyev_batched_available(12_000, false));
}

#[test]
fn cusolver_cuda_data_type_has_c_abi_integer_layout() {
    assert_eq!(
        std::mem::size_of::<CudaDataType>(),
        std::mem::size_of::<i32>()
    );
    assert_eq!(
        std::mem::align_of::<CudaDataType>(),
        std::mem::align_of::<i32>()
    );
}

#[test]
#[ignore = "requires the cuSOLVER and cuBLAS runtime libraries"]
fn vendor_libraries_are_loaded_once_per_process() {
    // Every handle shares one loaded library, so dropping backends never
    // unloads and reloads cuSOLVER or cuBLAS (#1924).
    let solver = super::cusolver::CusolverLibrary::load().unwrap();
    assert!(std::sync::Arc::ptr_eq(
        &solver,
        &super::cusolver::CusolverLibrary::load().unwrap()
    ));
    let blas = super::cusolver::CublasLibrary::load().unwrap();
    assert!(std::sync::Arc::ptr_eq(
        &blas,
        &super::cusolver::CublasLibrary::load().unwrap()
    ));
}
