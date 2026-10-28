#[cfg(feature = "cuda")]
#[test]
fn cuda_engine_exposes_engine_operations() {
    use tenferro_gpu::cuda::CudaEngine;
    use tenferro_tensor::{Engine, Tensor, TensorElementwise, TensorRead};

    fn assert_backend_session<S: Engine + ?Sized>() {}

    assert_backend_session::<CudaEngine>();
    let _add: fn(
        &mut CudaEngine,
        TensorRead<'_>,
        TensorRead<'_>,
    ) -> tenferro_tensor::Result<Tensor> = TensorElementwise::add_read;
}

/// The engine-identity spellings the engine cutover removed stay removed.
///
/// Two trybuild cases used to prove this by importing `tenferro_gpu::cuda_runtime_engine_id`
/// and `tenferro_gpu::cuda::cuda_runtime_engine_registration_with_id` and comparing rustc's
/// unresolved-import diagnostic, which changed rendering between stable releases and so
/// tested the compiler's wording rather than this crate's surface (issue #2299). Assert the
/// source instead: no library file names either spelling, so no module (re-exported by glob
/// or otherwise) can bring one back.
#[test]
fn removed_runtime_engine_names_are_not_exported() {
    const REMOVED: [&str; 2] = [
        "cuda_runtime_engine_id",
        "cuda_runtime_engine_registration_with_id",
    ];
    let mut pending = vec![std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src")];
    let mut scanned = 0usize;
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).expect("read the source tree") {
            let path = entry.expect("source entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let source = std::fs::read_to_string(&path).expect("read a source file");
                scanned += 1;
                for removed in REMOVED {
                    assert!(
                        !source.contains(removed),
                        "{} must not name the removed export `{removed}`",
                        path.display()
                    );
                }
            }
        }
    }
    assert!(scanned > 10, "the scan must reach the library sources");
}

#[cfg(feature = "cuda")]
#[test]
fn execution_session_capability_cannot_project_or_escape_owner_borrow() {
    if std::env::var_os("NEXTEST").is_some() {
        eprintln!("skipping compile-only trybuild contract under nextest");
        return;
    }

    let tests = trybuild::TestCases::new();

    tests.compile_fail("tests/ui/cuda_engine_is_not_clone.rs");
    tests.compile_fail("tests/ui/cuda_backend_rejects_bare_integer.rs");
    tests.compile_fail("tests/ui/cuda_pending_upload_moves_the_destination.rs");
    tests.compile_fail("tests/ui/cuda_pending_download_moves_the_source.rs");
    tests.compile_fail("tests/ui/cuda_source_publication_moves_the_value.rs");
}
