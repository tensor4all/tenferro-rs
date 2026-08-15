//! Session-level behavioral tests for the uninitialized allocated-dot output
//! path (issue #1690): the GEMM provider witness gating in
//! `execute_dot_allocated`, the `Unsupported` fallback, and values parity
//! between the uninit and zeroed paths.

use super::*;
use num_complex::Complex64;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::provider::{
    CpuExecutionContext, CpuGemmProvider, CpuGemmRequest, CpuGroupedGemmRequest,
    CpuProviderOutcome, CpuUninitGemmProvider,
};
use crate::{
    CpuDomainId, CpuPlacementGuarantee, CpuProviderBundle, ExternalCpuDomain, ResolvedCpuPlacement,
};
use tenferro_tensor::{DType, DotGeneralConfig};

/// Build a CPU backend that runs a custom provider bundle on one managed
/// thread, mirroring `from_external_managed_domains_with_provider_bundle`
/// semantics with the real process topology.
fn backend_with_bundle(bundle: CpuProviderBundle) -> CpuBackend {
    let topology = crate::discover_cpu_topology().unwrap();
    let cpus = topology.allowed_cpus().clone();
    CpuBackend::from_external_managed_domains_with_provider_bundle(
        CpuDomainId::new(31337),
        [ExternalCpuDomain::new(
            CpuDomainId::new(31337),
            ResolvedCpuPlacement::AllAllowed { cpus },
            Arc::new(CpuContext::with_threads(1).unwrap()),
            NonZeroUsize::new(1).unwrap(),
            CpuPlacementGuarantee::AdvisoryDeclared,
        )
        .unwrap()],
        bundle,
    )
    .unwrap()
}

/// GEMM provider that delegates to the built-in faer provider without exposing
/// the uninitialized-output witness (default opt-out).
#[derive(Debug)]
struct OptOutGemmProvider {
    gemm_calls: Arc<AtomicUsize>,
}

impl CpuGemmProvider for OptOutGemmProvider {
    fn execution_capabilities(&self) -> crate::CpuProviderExecutionCapabilities {
        crate::provider_capability::engine_worker_capabilities()
    }

    fn gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.gemm_calls.fetch_add(1, Ordering::Relaxed);
        crate::provider::FaerGemmProvider.gemm(context, request)
    }

    fn strided_batched_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.gemm_calls.fetch_add(1, Ordering::Relaxed);
        crate::provider::FaerGemmProvider.strided_batched_gemm(context, request)
    }

    fn grouped_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        crate::provider::FaerGemmProvider.grouped_gemm(context, request)
    }
}

/// GEMM provider that opts into the uninitialized-output contract and executes
/// it by delegating to the built-in faer provider.
#[derive(Debug)]
struct ExecutingUninitGemmProvider {
    gemm_calls: Arc<AtomicUsize>,
    uninit_calls: Arc<AtomicUsize>,
}

impl CpuGemmProvider for ExecutingUninitGemmProvider {
    fn execution_capabilities(&self) -> crate::CpuProviderExecutionCapabilities {
        crate::provider_capability::engine_worker_capabilities()
    }

    fn gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.gemm_calls.fetch_add(1, Ordering::Relaxed);
        crate::provider::FaerGemmProvider.gemm(context, request)
    }

    fn strided_batched_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.gemm_calls.fetch_add(1, Ordering::Relaxed);
        crate::provider::FaerGemmProvider.strided_batched_gemm(context, request)
    }

    fn grouped_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        crate::provider::FaerGemmProvider.grouped_gemm(context, request)
    }

    fn uninit_provider(&self) -> Option<&dyn CpuUninitGemmProvider> {
        Some(self)
    }
}

// SAFETY: the delegated faer implementation writes every destination element
// before `Executed` (Accum::Replace for beta == 0; zeros for empty
// contractions), so this wrapper inherits the full-overwrite contract.
unsafe impl CpuUninitGemmProvider for ExecutingUninitGemmProvider {
    unsafe fn gemm_into_uninit(
        &self,
        context: &CpuExecutionContext<'_>,
        request: crate::provider::CpuGemmUninitRequest<'_, '_>,
        output_bytes: &mut [std::mem::MaybeUninit<u8>],
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.uninit_calls.fetch_add(1, Ordering::Relaxed);
        // SAFETY: delegation preserves the caller's beta == 0 full-overwrite
        // precondition and the destination-write guarantee.
        unsafe {
            crate::provider::FaerGemmProvider.gemm_into_uninit(context, request, output_bytes)
        }
    }
}

/// GEMM provider that opts into the uninitialized-output contract but always
/// reports `Unsupported`, forcing the caller's discard-and-reallocate fallback
/// onto the zeroed path.
#[derive(Debug)]
struct UnsupportedUninitGemmProvider {
    gemm_calls: Arc<AtomicUsize>,
    uninit_calls: Arc<AtomicUsize>,
}

impl CpuGemmProvider for UnsupportedUninitGemmProvider {
    fn execution_capabilities(&self) -> crate::CpuProviderExecutionCapabilities {
        crate::provider_capability::engine_worker_capabilities()
    }

    fn gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.gemm_calls.fetch_add(1, Ordering::Relaxed);
        crate::provider::FaerGemmProvider.gemm(context, request)
    }

    fn strided_batched_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.gemm_calls.fetch_add(1, Ordering::Relaxed);
        crate::provider::FaerGemmProvider.strided_batched_gemm(context, request)
    }

    fn grouped_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        crate::provider::FaerGemmProvider.grouped_gemm(context, request)
    }

    fn uninit_provider(&self) -> Option<&dyn CpuUninitGemmProvider> {
        Some(self)
    }
}

// SAFETY: this test provider asserts the unsafe trait only to exercise the
// caller's `Unsupported` fallback; it never writes the destination, so it must
// always return `Unsupported` (never `Executed`).
unsafe impl CpuUninitGemmProvider for UnsupportedUninitGemmProvider {
    unsafe fn gemm_into_uninit(
        &self,
        _context: &CpuExecutionContext<'_>,
        _request: crate::provider::CpuGemmUninitRequest<'_, '_>,
        _output_bytes: &mut [std::mem::MaybeUninit<u8>],
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.uninit_calls.fetch_add(1, Ordering::Relaxed);
        Ok(CpuProviderOutcome::Unsupported(
            crate::provider::CpuProviderUnsupported::DType(DType::F64),
        ))
    }
}

fn matmul_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    }
}

fn dot_bundle(gemm: Arc<dyn CpuGemmProvider>) -> CpuProviderBundle {
    CpuProviderBundle::custom_builder()
        .gemm_provider(gemm)
        .layout_transform_provider(Arc::new(crate::provider::StridedLayoutTransformProvider))
        .build()
        .unwrap()
}

#[test]
fn opted_out_gemm_provider_keeps_zeroed_dot_output_values() {
    let gemm = Arc::new(OptOutGemmProvider {
        gemm_calls: Arc::new(AtomicUsize::new(0)),
    });
    let gemm_calls = Arc::clone(&gemm.gemm_calls);
    let mut backend = backend_with_bundle(dot_bundle(gemm));

    let lhs = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap(),
    );
    let rhs = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![3, 2], vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0]).unwrap(),
    );
    let output = backend.dot_general(&lhs, &rhs, &matmul_config()).unwrap();

    assert_eq!(output.as_slice::<f64>().unwrap(), &[17.0, 41.0, 33.0, 81.0]);
    // Without the witness the uninit path is never attempted; the zeroed path
    // executed the GEMM exactly once.
    assert_eq!(gemm_calls.load(Ordering::Relaxed), 1);
}

#[test]
fn opted_in_gemm_provider_unsupported_falls_back_to_zeroed_dot() {
    let gemm = Arc::new(UnsupportedUninitGemmProvider {
        gemm_calls: Arc::new(AtomicUsize::new(0)),
        uninit_calls: Arc::new(AtomicUsize::new(0)),
    });
    let gemm_calls = Arc::clone(&gemm.gemm_calls);
    let uninit_calls = Arc::clone(&gemm.uninit_calls);
    let mut backend = backend_with_bundle(dot_bundle(gemm));

    let lhs = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap(),
    );
    let rhs = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![3, 2], vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0]).unwrap(),
    );
    let output = backend.dot_general(&lhs, &rhs, &matmul_config()).unwrap();

    assert_eq!(output.as_slice::<f64>().unwrap(), &[17.0, 41.0, 33.0, 81.0]);
    // The uninit attempt fired once and was discarded; the zeroed fallback
    // then executed the GEMM once.
    assert_eq!(uninit_calls.load(Ordering::Relaxed), 1);
    assert_eq!(gemm_calls.load(Ordering::Relaxed), 1);
}

/// Compare every allocated-dot case between the uninit path (executing
/// witness) and the zeroed path (opt-out provider).
#[test]
fn uninit_dot_path_values_match_zeroed_path_for_allocated_dots() {
    let executing = Arc::new(ExecutingUninitGemmProvider {
        gemm_calls: Arc::new(AtomicUsize::new(0)),
        uninit_calls: Arc::new(AtomicUsize::new(0)),
    });
    let uninit_calls = Arc::clone(&executing.uninit_calls);
    let mut uninit_backend = backend_with_bundle(dot_bundle(executing));
    let mut zeroed_backend = backend_with_bundle(dot_bundle(Arc::new(OptOutGemmProvider {
        gemm_calls: Arc::new(AtomicUsize::new(0)),
    })));

    let cases: Vec<(Tensor, Tensor, DotGeneralConfig)> = vec![
        // Plain matmul.
        (
            Tensor::F64(
                TypedTensor::from_vec_col_major(vec![2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
                    .unwrap(),
            ),
            Tensor::F64(
                TypedTensor::from_vec_col_major(vec![3, 2], vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0])
                    .unwrap(),
            ),
            matmul_config(),
        ),
        // Batched matmul: batch dim 0 on both operands.
        (
            Tensor::F64(
                TypedTensor::from_vec_col_major(
                    vec![2, 2, 3],
                    vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0],
                )
                .unwrap(),
            ),
            Tensor::F64(
                TypedTensor::from_vec_col_major(
                    vec![2, 3, 2],
                    vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                )
                .unwrap(),
            ),
            DotGeneralConfig {
                lhs_contracting_dims: vec![2],
                rhs_contracting_dims: vec![1],
                lhs_batch_dims: vec![0],
                rhs_batch_dims: vec![0],
            },
        ),
        // Empty contraction (k == 0): output must be zeros.
        (
            Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 0], Vec::<f64>::new()).unwrap()),
            Tensor::F64(TypedTensor::from_vec_col_major(vec![0, 2], Vec::<f64>::new()).unwrap()),
            matmul_config(),
        ),
        // Empty output (zero rows).
        (
            Tensor::F64(TypedTensor::from_vec_col_major(vec![0, 3], Vec::<f64>::new()).unwrap()),
            Tensor::F64(
                TypedTensor::from_vec_col_major(vec![3, 2], vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0])
                    .unwrap(),
            ),
            matmul_config(),
        ),
    ];

    for (lhs, rhs, config) in cases {
        let uninit_output = uninit_backend.dot_general(&lhs, &rhs, &config).unwrap();
        let zeroed_output = zeroed_backend.dot_general(&lhs, &rhs, &config).unwrap();
        assert_eq!(uninit_output.shape(), zeroed_output.shape());
        assert_eq!(
            uninit_output.as_slice::<f64>().unwrap(),
            zeroed_output.as_slice::<f64>().unwrap(),
        );
    }

    // Conjugated complex dot (direct faer conjugation) also matches.
    let lhs = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex64::new(1.0, 2.0),
                Complex64::new(-3.0, 0.5),
                Complex64::new(2.0, -1.0),
                Complex64::new(0.25, 4.0),
            ],
        )
        .unwrap(),
    );
    let rhs = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex64::new(-2.0, 1.0),
                Complex64::new(1.5, -0.25),
                Complex64::new(0.5, 3.0),
                Complex64::new(-1.0, -2.0),
            ],
        )
        .unwrap(),
    );
    let config = matmul_config();
    let uninit_output = uninit_backend
        .dot_general_with_conj(&lhs, &rhs, &config, true, true)
        .unwrap();
    let zeroed_output = zeroed_backend
        .dot_general_with_conj(&lhs, &rhs, &config, true, true)
        .unwrap();
    assert_eq!(
        uninit_output.as_slice::<Complex64>().unwrap(),
        zeroed_output.as_slice::<Complex64>().unwrap(),
    );

    // The executing witness handled every direct case above (the conjugated
    // complex dot included); only the zeroed fallback never fires.
    assert_eq!(uninit_calls.load(Ordering::Relaxed), 5);
}
