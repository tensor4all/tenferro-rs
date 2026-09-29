//! Typed extension slots of `CpuProviderBundle`: operation-family crates
//! (tenferro-linalg) install their own provider traits without tenferro-cpu
//! naming them.
use std::sync::Arc;

use tenferro_cpu::{with_cpu_exec_session, CpuBackend, CpuBackendKind, CpuProviderBundle};
use tenferro_tensor::BackendSessionHost;

#[derive(Debug, PartialEq)]
struct Kernels(u32);

#[derive(Debug)]
struct Other;

#[test]
fn extensions_are_keyed_by_type_and_the_last_install_wins() {
    let bundle = CpuProviderBundle::builder(CpuBackendKind::default_compiled())
        .extension(Arc::new(Kernels(1)))
        .extension(Arc::new(Kernels(2)))
        .build()
        .unwrap();
    assert_eq!(bundle.extension::<Kernels>().as_deref(), Some(&Kernels(2)));
    assert!(bundle.extension::<Other>().is_none());
    let plain = CpuProviderBundle::builder(CpuBackendKind::default_compiled())
        .build()
        .unwrap();
    assert!(plain.extension::<Kernels>().is_none());
}

#[test]
fn a_session_sees_the_extensions_of_its_backend() {
    let bundle = CpuProviderBundle::builder(CpuBackendKind::default_compiled())
        .extension(Arc::new(Kernels(7)))
        .build()
        .unwrap();
    let mut backend = CpuBackend::with_threads(1)
        .unwrap()
        .with_provider_bundle(bundle)
        .unwrap();
    let seen = backend
        .with_backend_session(|session| {
            with_cpu_exec_session(session, |cpu| {
                cpu.provider_extension::<Kernels>().map(|k| k.0)
            })
            .expect("a CPU backend session")
        })
        .unwrap();
    assert_eq!(seen, Some(7));
    let mut plain = CpuBackend::with_threads(1).unwrap();
    let none = plain
        .with_backend_session(|session| {
            with_cpu_exec_session(session, |cpu| cpu.provider_extension::<Kernels>().is_none())
                .expect("a CPU backend session")
        })
        .unwrap();
    assert!(none);
}
