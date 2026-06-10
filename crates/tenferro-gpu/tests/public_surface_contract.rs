use std::path::PathBuf;

fn repo_file(path: &str) -> String {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.push("../..");
    root.push(path);
    std::fs::read_to_string(root).expect("source file must be readable")
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
