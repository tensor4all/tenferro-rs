//! Backend `_read` coverage contract.
//!
//! Issue #1926 made the operation read halves **required** trait items, so a
//! backend that omits one no longer compiles and a trait default can no longer
//! silently reject borrowed views. What this contract still watches is the part
//! the compiler does not enforce: CUDA kernels consume owned compact tensors, so
//! `tenferro-gpu` has to materialize a view through `to_contiguous_read` and
//! then run the owned kernel, and both the backend and the erased session shim
//! must keep implementing every required read entry point.

use std::collections::BTreeSet;
use std::fs;

/// The operation families this contract covers.
const OPERATION_TRAITS: [&str; 6] = [
    "TensorElementwise",
    "TensorAnalytic",
    "TensorStructural",
    "TensorReduction",
    "TensorIndexing",
    "TensorDot",
];

/// Collect every `_read` entry point the operation traits require.
///
/// Only the operation families are scanned: `backend.rs` also declares
/// `_read`-suffixed items in unrelated traits (for example the view-wrapping
/// helper `wrap_read`), which a backend is not expected to implement.
fn required_read_entry_points() -> BTreeSet<String> {
    let source = fs::read_to_string("../tenferro-tensor/src/backend.rs")
        .expect("the tensor backend trait source should be readable");
    let mut names = BTreeSet::new();
    let operation_source: String = source
        .split("\npub trait ")
        .filter(|chunk| {
            OPERATION_TRAITS.iter().any(|trait_name| {
                chunk.starts_with(&format!("{trait_name}:"))
                    || chunk.starts_with(&format!("{trait_name} "))
            })
        })
        .collect::<Vec<_>>()
        .join("\npub trait ");
    for chunk in operation_source.split("\n    fn ").skip(1) {
        let Some(name) = chunk
            .split('(')
            .next()
            .filter(|name| name.ends_with("_read"))
        else {
            continue;
        };
        let head = chunk.split([';', '{']).next().unwrap_or("");
        if head.len() < chunk.len() && chunk.as_bytes()[head.len()] == b';' {
            names.insert(name.to_string());
        }
    }
    names
}

#[test]
fn cuda_backend_implements_every_required_read_entry_point() {
    let entry_points = required_read_entry_points();
    assert!(
        entry_points.contains("transpose_read") && entry_points.contains("add_read"),
        "the trait scan should find the required read entry points, found {entry_points:?}"
    );

    let backend = fs::read_to_string("src/cubecl/mod.rs")
        .expect("the CUDA backend source should be readable");
    let session = fs::read_to_string("src/cubecl/exec_session.rs")
        .expect("the CUDA session source should be readable");

    let missing_backend: Vec<&String> = entry_points
        .iter()
        .filter(|name| !backend.contains(&format!("fn {name}(")))
        .collect();
    assert!(
        missing_backend.is_empty(),
        "CudaBackend must implement {missing_backend:?}; a read entry point materializes the view \
         and then runs the owned kernel"
    );

    let missing_session: Vec<&String> = entry_points
        .iter()
        .filter(|name| !session.contains(&format!("fn {name}(")))
        .collect();
    assert!(
        missing_session.is_empty(),
        "CudaExecSession must forward {missing_session:?} to CudaBackend; the traced runtime reaches \
         these entry points through the erased BackendSession surface"
    );
}
