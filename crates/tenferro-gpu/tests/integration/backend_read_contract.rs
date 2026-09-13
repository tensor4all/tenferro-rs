//! Backend `_read` coverage contract.
//!
//! `tenferro-tensor` defaults for `_read` entry points reject borrowed views:
//! a backend only accepts the borrowed-view spelling when it implements the
//! entry point itself. CUDA kernels consume owned compact tensors, so
//! `tenferro-gpu` materializes a view through `to_contiguous_read` and
//! delegates to the owned kernel. That implementation is easy to lose when the
//! trait surface grows, which is what this contract watches.

use std::collections::BTreeSet;
use std::fs;

/// Collect every `_read` entry point whose `tenferro-tensor` default refuses
/// borrowed views.
fn view_rejecting_read_entry_points() -> BTreeSet<String> {
    let source = fs::read_to_string("../tenferro-tensor/src/backend.rs")
        .expect("the tensor backend trait source should be readable");
    let mut names = BTreeSet::new();
    for chunk in source.split("\n    fn ").skip(1) {
        let Some(name) = chunk
            .split('(')
            .next()
            .filter(|name| name.ends_with("_read"))
        else {
            continue;
        };
        let body = chunk.split("\n    }").next().unwrap_or(chunk);
        if body.contains("read_tensor(") {
            names.insert(name.to_string());
        }
    }
    names
}

#[test]
fn cuda_backend_implements_every_view_rejecting_read_entry_point() {
    let entry_points = view_rejecting_read_entry_points();
    assert!(
        entry_points.contains("transpose_read") && entry_points.contains("add_read"),
        "the trait scan should find the known view-rejecting defaults, found {entry_points:?}"
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
        "CudaBackend must implement {missing_backend:?}; the trait default rejects borrowed views, \
         which the traced runtime can hand to every operation operand"
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
