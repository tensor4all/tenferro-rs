use std::num::NonZeroUsize;

use super::*;
use crate::{CpuId, CpuPlacementGuarantee, CpuSet};

fn expected(
    thread_count: CpuThreadCountControl,
    placement: CpuPlacementControl,
    worker_local_sequential: bool,
    accepts_sequential: bool,
    accepts_outer: bool,
    accepts_inner: bool,
) -> CpuProviderExecutionCapabilities {
    CpuProviderExecutionCapabilities {
        thread_count,
        placement,
        worker_local_sequential,
        accepts_sequential,
        accepts_outer,
        accepts_inner,
    }
}

#[test]
fn injected_probe_table_classifies_known_and_unknown_blas_once() {
    let controlled_external = expected(
        CpuThreadCountControl::PerCallUpperBound,
        CpuPlacementControl::ExternalWorkers,
        true,
        true,
        true,
        true,
    );
    let binary_external = expected(
        CpuThreadCountControl::BinaryClampToOne,
        CpuPlacementControl::ExternalWorkers,
        true,
        true,
        true,
        true,
    );
    let uncontrolled_external = expected(
        CpuThreadCountControl::GlobalOrUncontrolled,
        CpuPlacementControl::ExternalWorkers,
        false,
        false,
        false,
        true,
    );
    let serial = expected(
        CpuThreadCountControl::Sequential,
        CpuPlacementControl::CallingThread,
        true,
        true,
        true,
        true,
    );
    let engine = expected(
        CpuThreadCountControl::PerCallUpperBound,
        CpuPlacementControl::EngineWorkers,
        true,
        true,
        true,
        true,
    );
    let unknown = expected(
        CpuThreadCountControl::GlobalOrUncontrolled,
        CpuPlacementControl::None,
        false,
        false,
        false,
        true,
    );

    let cases = [
        ("faer/native", CpuProviderProbe::FaerOrNative, engine),
        (
            "MKL thread-local setter wired by its adapter",
            CpuProviderProbe::Mkl {
                thread_local_setter_wired: true,
            },
            controlled_external,
        ),
        (
            "MKL symbol without an adapter guard",
            CpuProviderProbe::Mkl {
                thread_local_setter_wired: false,
            },
            uncontrolled_external,
        ),
        (
            "pthread OpenBLAS global set-and-restore remains uncontrolled",
            CpuProviderProbe::OpenBlas(OpenBlasProbe {
                parallelism: OpenBlasParallelism::Pthread,
                process_global_set_restore_wired: true,
            }),
            uncontrolled_external,
        ),
        (
            "pthread OpenBLAS without a global adapter remains uncontrolled",
            CpuProviderProbe::OpenBlas(OpenBlasProbe {
                parallelism: OpenBlasParallelism::Pthread,
                process_global_set_restore_wired: false,
            }),
            uncontrolled_external,
        ),
        (
            "OpenMP OpenBLAS cannot claim local isolation",
            CpuProviderProbe::OpenBlas(OpenBlasProbe {
                parallelism: OpenBlasParallelism::OpenMp,
                process_global_set_restore_wired: true,
            }),
            uncontrolled_external,
        ),
        (
            "sequential OpenBLAS",
            CpuProviderProbe::OpenBlas(OpenBlasProbe {
                parallelism: OpenBlasParallelism::Sequential,
                process_global_set_restore_wired: false,
            }),
            serial,
        ),
        (
            "unknown OpenBLAS build",
            CpuProviderProbe::OpenBlas(OpenBlasProbe {
                parallelism: OpenBlasParallelism::Unknown,
                process_global_set_restore_wired: true,
            }),
            unknown,
        ),
        (
            "macOS 15 Accelerate adapter",
            CpuProviderProbe::Accelerate(AccelerateProbe {
                binary_thread_local_control_wired: true,
            }),
            binary_external,
        ),
        (
            "older or unwired Accelerate",
            CpuProviderProbe::Accelerate(AccelerateProbe {
                binary_thread_local_control_wired: false,
            }),
            uncontrolled_external,
        ),
        (
            "ArmPL _mp",
            CpuProviderProbe::ArmPlOpenMp,
            uncontrolled_external,
        ),
        ("serial ArmPL", CpuProviderProbe::ArmPlSerial, serial),
        ("serial NVPL", CpuProviderProbe::NvplSerial, serial),
        ("unknown BLAS", CpuProviderProbe::UnknownBlas, unknown),
        (
            "injected BLAS without descriptor",
            CpuProviderProbe::Injected(None),
            unknown,
        ),
    ];

    for (name, probe, expected) in cases {
        assert_eq!(classify_provider(probe), expected, "{name}");
    }
}

#[test]
fn openblas_global_set_restore_never_claims_per_call_count_control() {
    for parallelism in [OpenBlasParallelism::Pthread, OpenBlasParallelism::OpenMp] {
        for process_global_set_restore_wired in [false, true] {
            let capabilities = classify_provider(CpuProviderProbe::OpenBlas(OpenBlasProbe {
                parallelism,
                process_global_set_restore_wired,
            }));
            assert_eq!(
                capabilities.thread_count,
                CpuThreadCountControl::GlobalOrUncontrolled,
            );
            assert!(!capabilities.worker_local_sequential);
        }
    }
}

#[test]
fn binary_clamp_to_one_never_selects_auto_for_a_finite_domain_budget() {
    for budget in [1, 2, 8] {
        assert_eq!(
            enforced_provider_thread_limit(
                CpuThreadCountControl::BinaryClampToOne,
                NonZeroUsize::new(budget).unwrap(),
            ),
            Some(NonZeroUsize::new(1).unwrap()),
        );
    }
}

#[test]
fn injected_explicit_descriptor_is_preserved_exactly() {
    let explicit = expected(
        CpuThreadCountControl::PerCallUpperBound,
        CpuPlacementControl::None,
        false,
        true,
        false,
        true,
    );

    assert_eq!(
        classify_provider(CpuProviderProbe::Injected(Some(explicit))),
        explicit,
    );
}

#[test]
fn builtin_blas_without_a_wired_scope_guard_stays_conservative() {
    let capabilities = builtin_blas_execution_capabilities();

    assert_eq!(
        capabilities.thread_count,
        CpuThreadCountControl::GlobalOrUncontrolled,
    );
    assert!(!capabilities.worker_local_sequential);
    assert!(!capabilities.accepts_outer);
}

fn cpu_set(ids: &[usize]) -> CpuSet {
    CpuSet::new(ids.iter().copied().map(CpuId::new)).unwrap()
}

fn budget(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).unwrap()
}

#[test]
fn strict_budget_one_allows_only_providers_that_can_force_inline_execution() {
    let domain = cpu_set(&[0, 1]);
    let process_allowed = cpu_set(&[0, 1, 2, 3]);
    let probes = [
        CpuProviderProbe::Mkl {
            thread_local_setter_wired: true,
        },
        CpuProviderProbe::Accelerate(AccelerateProbe {
            binary_thread_local_control_wired: true,
        }),
        CpuProviderProbe::ArmPlSerial,
    ];

    for probe in probes {
        validate_provider_for_domain(
            classify_provider(probe),
            budget(1),
            CpuPlacementGuarantee::ExactDeclared,
            &domain,
            &process_allowed,
        )
        .unwrap();
    }

    let openblas_error = validate_provider_for_domain(
        classify_provider(CpuProviderProbe::OpenBlas(OpenBlasProbe {
            parallelism: OpenBlasParallelism::Pthread,
            process_global_set_restore_wired: true,
        })),
        budget(1),
        CpuPlacementGuarantee::ExactDeclared,
        &domain,
        &process_allowed,
    )
    .unwrap_err();
    assert!(matches!(
        openblas_error,
        CpuProviderDomainError::ThreadCountNotEnforceable { .. }
    ));

    let error = validate_provider_for_domain(
        builtin_blas_execution_capabilities(),
        budget(1),
        CpuPlacementGuarantee::ExactDeclared,
        &domain,
        &process_allowed,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        CpuProviderDomainError::ThreadCountNotEnforceable { .. }
    ));
}

#[test]
fn domain_compatibility_table_enforces_count_and_placement_independently() {
    #[derive(Clone, Copy, Debug)]
    enum Expected {
        Compatible,
        CountError,
        PlacementError,
    }

    let strict_subdomain = cpu_set(&[0, 1]);
    let process_allowed = cpu_set(&[0, 1, 2, 3]);
    let controlled_external = classify_provider(CpuProviderProbe::Mkl {
        thread_local_setter_wired: true,
    });
    let binary_external = classify_provider(CpuProviderProbe::Accelerate(AccelerateProbe {
        binary_thread_local_control_wired: true,
    }));
    let uncontrolled_external = classify_provider(CpuProviderProbe::ArmPlOpenMp);
    let serial = classify_provider(CpuProviderProbe::ArmPlSerial);
    let engine = classify_provider(CpuProviderProbe::FaerOrNative);
    let explicit_without_placement = expected(
        CpuThreadCountControl::PerCallUpperBound,
        CpuPlacementControl::None,
        false,
        true,
        false,
        true,
    );

    let cases = [
        (
            "engine workers honor an exact subdomain",
            engine,
            4,
            CpuPlacementGuarantee::ExactDeclared,
            &strict_subdomain,
            Expected::Compatible,
        ),
        (
            "serial providers satisfy every exact-domain budget",
            serial,
            4,
            CpuPlacementGuarantee::ExactDeclared,
            &strict_subdomain,
            Expected::Compatible,
        ),
        (
            "controlled external workers reject a strict multi-thread subdomain",
            controlled_external,
            2,
            CpuPlacementGuarantee::ExactDeclared,
            &strict_subdomain,
            Expected::PlacementError,
        ),
        (
            "controlled external workers accept the process-wide exact domain",
            controlled_external,
            2,
            CpuPlacementGuarantee::ExactDeclared,
            &process_allowed,
            Expected::Compatible,
        ),
        (
            "controlled external workers accept an advisory subdomain",
            controlled_external,
            2,
            CpuPlacementGuarantee::AdvisoryDeclared,
            &strict_subdomain,
            Expected::Compatible,
        ),
        (
            "binary control may clamp advisory execution to one thread",
            binary_external,
            8,
            CpuPlacementGuarantee::AdvisoryDeclared,
            &strict_subdomain,
            Expected::Compatible,
        ),
        (
            "binary control still rejects strict external placement above budget one",
            binary_external,
            8,
            CpuPlacementGuarantee::ExactDeclared,
            &strict_subdomain,
            Expected::PlacementError,
        ),
        (
            "global control cannot enforce an advisory upper bound",
            uncontrolled_external,
            2,
            CpuPlacementGuarantee::AdvisoryDeclared,
            &strict_subdomain,
            Expected::CountError,
        ),
        (
            "unknown providers stay conservative even at budget one",
            CpuProviderExecutionCapabilities::default(),
            1,
            CpuPlacementGuarantee::AdvisoryDeclared,
            &strict_subdomain,
            Expected::CountError,
        ),
        (
            "advisory domains need no placement claim after count validation",
            explicit_without_placement,
            2,
            CpuPlacementGuarantee::AdvisoryDeclared,
            &strict_subdomain,
            Expected::Compatible,
        ),
        (
            "strict domains reject a provider with no placement claim",
            explicit_without_placement,
            2,
            CpuPlacementGuarantee::ExactDeclared,
            &process_allowed,
            Expected::PlacementError,
        ),
    ];

    for (name, capabilities, threads, guarantee, domain, expected) in cases {
        let result = validate_provider_for_domain(
            capabilities,
            budget(threads),
            guarantee,
            domain,
            &process_allowed,
        );
        match expected {
            Expected::Compatible => assert_eq!(result, Ok(()), "{name}"),
            Expected::CountError => assert!(
                matches!(
                    result,
                    Err(CpuProviderDomainError::ThreadCountNotEnforceable { .. })
                ),
                "{name}: {result:?}",
            ),
            Expected::PlacementError => assert!(
                matches!(
                    result,
                    Err(CpuProviderDomainError::PlacementNotEnforceable { .. })
                ),
                "{name}: {result:?}",
            ),
        }
    }
}
