use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use tenferro_cpu::{CpuBackend, CpuPlacement};
use tenferro_runtime::Error;
use tenferro_tensor::ErrorKind;

use crate::eager_backend::EagerBackend;

use super::super::EagerRuntime;

fn item_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source
        .find(signature)
        .unwrap_or_else(|| panic!("missing source item: {signature}"));
    let open = start
        + source[start..]
            .find('{')
            .unwrap_or_else(|| panic!("missing source body: {signature}"));
    let mut depth = 0usize;
    for (offset, character) in source[open..].char_indices() {
        match character {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..=open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated source body: {signature}");
}

#[test]
fn non_cpu_runtime_rejects_placement_binding_as_typed_unsupported() {
    let runtime = Arc::new(EagerRuntime::from_backend(EagerBackend::recording_cpu(
        Arc::new(AtomicUsize::new(0)),
    )));

    let error = runtime.on_cpu(CpuPlacement::Auto).unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(matches!(error, Error::Unsupported { .. }));
}

#[test]
fn placement_binding_reports_poisoned_runtime_backend_lock() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = runtime.backend.lock().unwrap();
        panic!("poison placement-bound eager backend lock");
    }));
    assert!(poisoned.is_err());

    let error = runtime.on_cpu(CpuPlacement::Auto).unwrap_err();

    assert_eq!(error.kind(), ErrorKind::RuntimeState);
    assert!(matches!(error, Error::RuntimeState { .. }));
}

#[test]
fn placement_bridge_source_keeps_runtime_and_executor_entry_boundaries_single() {
    let eager_source = include_str!("../../eager.rs");
    let eager_backend_source = include_str!("../../eager_backend.rs");
    let on_cpu = item_body(eager_source, "    pub fn on_cpu(");
    let with_session = item_body(eager_source, "    pub fn with_eager_session<R: Send>(");
    let snapshot = item_body(
        eager_backend_source,
        "    pub(crate) fn cpu_snapshot(&self)",
    );

    assert_eq!(on_cpu.matches("self.lock_backend()").count(), 1);
    assert_eq!(on_cpu.matches("cpu_snapshot()").count(), 1);
    assert_eq!(on_cpu.matches("for_placement(placement)").count(), 1);
    assert!(
        on_cpu.contains("        };\n        let backend = backend.for_placement(placement)"),
        "placement resolution must happen after the eager backend guard scope"
    );
    assert!(!on_cpu.contains("extension_executor"));
    assert!(!on_cpu.contains(".install("));

    assert_eq!(with_session.matches("with_backend_session(f)").count(), 1);
    for forbidden in [
        "lock_backend",
        ".lock(",
        "Mutex",
        "extension_executor",
        ".install(",
        "self.runtime.",
    ] {
        assert!(
            !with_session.contains(forbidden),
            "with_eager_session contains forbidden second entry `{forbidden}`"
        );
    }

    assert_eq!(snapshot.matches("backend.clone()").count(), 1);
    assert!(!snapshot.contains(".lock("));
}

#[test]
fn placement_bridge_is_handwritten_debug_without_clone_and_documents_phase_two_scope() {
    let source = include_str!("../../eager.rs");
    let declaration = source
        .split_once("pub struct CpuPlacementBoundEager")
        .expect("placement-bound eager type declaration")
        .0;
    let nearby_attributes = declaration
        .rsplit_once("/// Placement-selected CPU view")
        .expect("placement-bound eager rustdoc")
        .1;

    assert!(!nearby_attributes.contains("derive(Clone"));
    assert!(!source.contains("impl Clone for CpuPlacementBoundEager"));
    assert!(source.contains("impl fmt::Debug for CpuPlacementBoundEager"));
    for excluded_family in ["linalg", "FFT", "einsum", "extension-runtime"] {
        assert!(
            nearby_attributes.contains(excluded_family),
            "public type docs must exclude phase-later family `{excluded_family}`"
        );
    }
}
