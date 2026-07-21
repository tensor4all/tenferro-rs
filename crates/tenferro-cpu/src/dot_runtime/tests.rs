use super::{
    validate_axis_groups, validate_dot_general, validate_layout_metadata, CpuProviderBundle,
};
use crate::buffer_pool::{BufferPool, PoolScalar};
use crate::gemm::GemmAnalysisCache;
use crate::provider::{
    CpuGemmProvider, CpuGemmRequest, CpuGeneralContractionProvider, CpuGroupedGemmRequest,
    CpuKernelParallelism, CpuLayoutTransformProvider, CpuLayoutTransformRequest, CpuOperand,
    CpuProviderContext, CpuProviderOutcome, CpuProviderUnsupported, StridedLayoutTransformProvider,
};
use crate::{CpuBackendKind, CpuContext};
use num_complex::Complex64;
use std::sync::{Arc, Mutex};
use tenferro_tensor::backend::{GroupedGemmConfig, GroupedGemmJob};
use tenferro_tensor::{
    ContractionScalar, DType, DotGeneralAccumulation, DotGeneralConfig, Tensor, TensorRead,
    TensorViewMut, TensorWrite, TypedTensorViewMut,
};

fn config(
    lhs_contracting_dims: &[usize],
    rhs_contracting_dims: &[usize],
    lhs_batch_dims: &[usize],
    rhs_batch_dims: &[usize],
) -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: lhs_contracting_dims.to_vec(),
        rhs_contracting_dims: rhs_contracting_dims.to_vec(),
        lhs_batch_dims: lhs_batch_dims.to_vec(),
        rhs_batch_dims: rhs_batch_dims.to_vec(),
    }
}

#[test]
fn axis_groups_preserve_order_and_find_free_axes() {
    let config = config(&[1], &[0], &[2], &[2]);
    let groups = validate_axis_groups(4, 4, &config).unwrap();

    assert_eq!(groups.contracting_pairs().collect::<Vec<_>>(), vec![(1, 0)]);
    assert_eq!(groups.batch_pairs().collect::<Vec<_>>(), vec![(2, 2)]);
    assert_eq!(groups.lhs_free_axes().collect::<Vec<_>>(), vec![0, 3]);
    assert_eq!(groups.rhs_free_axes().collect::<Vec<_>>(), vec![1, 3]);
}

#[test]
fn axis_groups_match_existing_rank_validation_through_rank_seventy() {
    for rank in [0, 1, 2, 8, 63, 64, 65, 70] {
        let valid = if rank == 0 {
            config(&[], &[], &[], &[])
        } else if rank == 1 {
            config(&[0], &[0], &[], &[])
        } else {
            config(&[rank - 1], &[0], &[rank - 2], &[rank - 1])
        };
        let invalid = [
            config(&[rank], &[0], &[], &[]),
            config(&[0, 0], &[0, 1], &[], &[]),
            config(&[0], &[0], &[0], &[0]),
            config(&[0], &[], &[], &[]),
            config(&[], &[], &[0], &[]),
        ];

        assert_eq!(
            validate_axis_groups(rank, rank, &valid).is_ok(),
            valid.validate_dims_with_ranks(rank, rank).is_ok(),
            "valid parity failed at rank {rank}",
        );
        for candidate in invalid {
            assert_eq!(
                validate_axis_groups(rank, rank, &candidate).is_ok(),
                candidate.validate_dims_with_ranks(rank, rank).is_ok(),
                "invalid parity failed at rank {rank}: {candidate:?}",
            );
        }
    }
}

#[test]
fn axis_group_role_conflict_preserves_ordered_error_parity() {
    let config = config(&[5, 2], &[0, 1], &[2, 5], &[2, 3]);
    let current = config.validate_dims_with_ranks(6, 6).unwrap_err();
    let candidate = validate_axis_groups(6, 6, &config).unwrap_err();

    assert_eq!(candidate.to_string(), current.to_string());
}

#[test]
fn axis_group_competing_errors_match_existing_precedence_through_rank_seventy() {
    for rank in [2, 8, 63, 64, 65, 70] {
        let cases = [
            config(&[0, 0], &[rank], &[], &[]),
            config(&[0, 0], &[0, 0], &[1, 1], &[1, 1]),
            config(&[0], &[], &[0], &[]),
            config(&[rank], &[rank], &[0, 0], &[0, 0]),
        ];
        for candidate in cases {
            let current = candidate.validate_dims_with_ranks(rank, rank).unwrap_err();
            let replacement = validate_axis_groups(rank, rank, &candidate).unwrap_err();
            assert_eq!(
                replacement.to_string(),
                current.to_string(),
                "error precedence diverged at rank {rank} for {candidate:?}",
            );
        }
    }
}

#[test]
fn dot_general_validation_checks_extents_output_and_accumulation() {
    let lhs = Tensor::from_vec_col_major(vec![2, 3, 4], vec![1.0_f64; 24]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![3, 5, 4], vec![1.0_f64; 60]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![2, 5, 4], vec![0.0_f64; 40]).unwrap();
    let lhs = TensorRead::from_tensor(&lhs);
    let rhs = TensorRead::from_tensor(&rhs);
    let mut output = TensorWrite::from_tensor(&mut output);
    let config = config(&[1], &[0], &[2], &[2]);
    let accumulation = DotGeneralAccumulation::overwrite(DType::F64).unwrap();

    let validated = validate_dot_general(&lhs, &rhs, &output, &config, accumulation).unwrap();
    assert_eq!(validated.output_element_count(), 40);
    assert_eq!(
        validated.axes().lhs_free_axes().collect::<Vec<_>>(),
        vec![0]
    );

    let wrong_accumulation = DotGeneralAccumulation {
        lhs_conj: false,
        rhs_conj: false,
        alpha: ContractionScalar::F32(1.0),
        beta: ContractionScalar::F32(0.0),
    };
    assert!(validate_dot_general(&lhs, &rhs, &output, &config, wrong_accumulation).is_err());

    let mut wrong_shape = Tensor::from_vec_col_major(vec![2, 5], vec![0.0_f64; 10]).unwrap();
    let wrong_shape = TensorWrite::from_tensor(&mut wrong_shape);
    assert!(validate_dot_general(&lhs, &rhs, &wrong_shape, &config, accumulation).is_err());

    let bad_rhs = Tensor::from_vec_col_major(vec![7, 5, 4], vec![1.0_f64; 140]).unwrap();
    let bad_rhs = TensorRead::from_tensor(&bad_rhs);
    assert!(validate_dot_general(&lhs, &bad_rhs, &output, &config, accumulation).is_err());

    let _ = &mut output;
}

#[test]
fn layout_validation_checks_strides_and_reachable_ranges() {
    assert!(validate_layout_metadata("output", &[2, 3], &[1], 0, 6).is_err());
    assert!(validate_layout_metadata("output", &[2], &[-1], 0, 2).is_err());
    assert!(validate_layout_metadata("output", &[2], &[isize::MAX], 1, 2).is_err());
    assert!(validate_layout_metadata("output", &[2, 3], &[1, 2], 0, 5).is_err());
    assert!(validate_layout_metadata("output", &[2, 3], &[-1, 2], 1, 6).is_ok());
}

#[test]
fn dot_general_validation_accepts_checked_negative_stride_output() {
    let lhs = Tensor::from_vec_col_major(vec![2, 3, 4], vec![1.0_f64; 24]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![3, 5, 4], vec![1.0_f64; 60]).unwrap();
    let mut output_storage = vec![0.0_f64; 40];
    let output =
        TypedTensorViewMut::from_slice(vec![2, 5, 4], vec![-1, 2, 10], 1, &mut output_storage)
            .unwrap();
    let output = TensorWrite::from_view(TensorViewMut::F64(output));
    let lhs = TensorRead::from_tensor(&lhs);
    let rhs = TensorRead::from_tensor(&rhs);
    let config = config(&[1], &[0], &[2], &[2]);
    let accumulation = DotGeneralAccumulation::overwrite(DType::F64).unwrap();

    let validated = validate_dot_general(&lhs, &rhs, &output, &config, accumulation).unwrap();
    assert_eq!(validated.output_element_count(), 40);
}

#[derive(Clone, Copy, Debug)]
enum GeneralBehavior {
    Outcome(CpuProviderOutcome),
    Error,
}

#[derive(Debug)]
struct GeneralSpy {
    behavior: GeneralBehavior,
    calls: Arc<Mutex<usize>>,
}

impl CpuGeneralContractionProvider for GeneralSpy {
    fn dot_general(
        &self,
        _context: &CpuProviderContext<'_>,
        _request: crate::provider::CpuDotGeneralRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        *self.calls.lock().unwrap() += 1;
        match self.behavior {
            GeneralBehavior::Outcome(outcome) => Ok(outcome),
            GeneralBehavior::Error => Err(tenferro_tensor::Error::runtime_state(
                "dot_general",
                "general spy failure",
            )),
        }
    }
}

#[derive(Debug)]
struct GemmSpy {
    behavior: GemmBehavior,
    gemm_calls: Arc<Mutex<usize>>,
    strided_calls: Arc<Mutex<usize>>,
    grouped_calls: Arc<Mutex<usize>>,
    parallelism: Arc<Mutex<Vec<CpuKernelParallelism>>>,
    grouped_job_counts: Arc<Mutex<Vec<usize>>>,
    in_selected_pool: Arc<Mutex<Vec<bool>>>,
}

#[derive(Clone, Copy, Debug)]
enum GemmBehavior {
    Outcome(CpuProviderOutcome),
    Error,
}

impl GemmSpy {
    fn new(outcome: CpuProviderOutcome) -> Self {
        Self {
            behavior: GemmBehavior::Outcome(outcome),
            gemm_calls: Arc::new(Mutex::new(0)),
            strided_calls: Arc::new(Mutex::new(0)),
            grouped_calls: Arc::new(Mutex::new(0)),
            parallelism: Arc::new(Mutex::new(Vec::new())),
            grouped_job_counts: Arc::new(Mutex::new(Vec::new())),
            in_selected_pool: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn error() -> Self {
        let mut spy = Self::new(CpuProviderOutcome::Executed);
        spy.behavior = GemmBehavior::Error;
        spy
    }

    fn result(&self) -> tenferro_tensor::Result<CpuProviderOutcome> {
        match self.behavior {
            GemmBehavior::Outcome(outcome) => Ok(outcome),
            GemmBehavior::Error => Err(tenferro_tensor::Error::runtime_state(
                "dot_general",
                "GEMM provider spy failure",
            )),
        }
    }
}

impl CpuGemmProvider for GemmSpy {
    fn gemm(
        &self,
        context: &CpuProviderContext<'_>,
        _request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        *self.gemm_calls.lock().unwrap() += 1;
        self.parallelism
            .lock()
            .unwrap()
            .push(context.kernel_parallelism());
        self.result()
    }

    fn strided_batched_gemm(
        &self,
        context: &CpuProviderContext<'_>,
        _request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        *self.strided_calls.lock().unwrap() += 1;
        self.parallelism
            .lock()
            .unwrap()
            .push(context.kernel_parallelism());
        self.result()
    }

    fn grouped_gemm(
        &self,
        context: &CpuProviderContext<'_>,
        request: CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        *self.grouped_calls.lock().unwrap() += 1;
        self.parallelism
            .lock()
            .unwrap()
            .push(context.kernel_parallelism());
        self.grouped_job_counts
            .lock()
            .unwrap()
            .push(request.jobs().len());
        self.in_selected_pool
            .lock()
            .unwrap()
            .push(context.cpu_context().owns_current_worker_for_test());
        self.result()
    }
}

#[derive(Clone, Copy, Debug)]
enum CanonicalFallbackBehavior {
    Real,
    ConjugatedComplex,
}

#[derive(Debug)]
struct CanonicalFallbackSpy {
    first_outcome: CpuProviderUnsupported,
    second_terminal: Option<CanonicalFallbackTerminal>,
    behavior: CanonicalFallbackBehavior,
    calls: Arc<Mutex<usize>>,
}

#[derive(Clone, Copy, Debug)]
enum CanonicalFallbackTerminal {
    Unsupported(CpuProviderUnsupported),
    Error,
}

impl CanonicalFallbackSpy {
    fn new(first_outcome: CpuProviderUnsupported, behavior: CanonicalFallbackBehavior) -> Self {
        Self {
            first_outcome,
            second_terminal: None,
            behavior,
            calls: Arc::new(Mutex::new(0)),
        }
    }

    fn rejecting_retry(
        first_outcome: CpuProviderUnsupported,
        second_outcome: CpuProviderUnsupported,
    ) -> Self {
        Self {
            first_outcome,
            second_terminal: Some(CanonicalFallbackTerminal::Unsupported(second_outcome)),
            behavior: CanonicalFallbackBehavior::Real,
            calls: Arc::new(Mutex::new(0)),
        }
    }

    fn erroring_retry(first_outcome: CpuProviderUnsupported) -> Self {
        Self {
            first_outcome,
            second_terminal: Some(CanonicalFallbackTerminal::Error),
            behavior: CanonicalFallbackBehavior::Real,
            calls: Arc::new(Mutex::new(0)),
        }
    }

    fn execute(
        &self,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        let call = {
            let mut calls = self.calls.lock().unwrap();
            *calls += 1;
            *calls
        };
        if call == 1 {
            return Ok(CpuProviderOutcome::Unsupported(self.first_outcome));
        }
        assert_eq!(call, 2, "canonical fallback must retry exactly once");
        if let Some(terminal) = self.second_terminal {
            return match terminal {
                CanonicalFallbackTerminal::Unsupported(reason) => {
                    Ok(CpuProviderOutcome::Unsupported(reason))
                }
                CanonicalFallbackTerminal::Error => Err(tenferro_tensor::Error::runtime_state(
                    "dot_general",
                    "canonical retry spy failure",
                )),
            };
        }

        let parts = request.into_parts();
        assert_eq!(parts.lhs_layout.row_stride(), 1);
        assert_eq!(parts.rhs_layout.row_stride(), 1);
        assert_eq!(parts.output_layout.row_stride(), 1);
        match self.behavior {
            CanonicalFallbackBehavior::Real => {
                assert!(!parts.accumulation.lhs_conj);
                assert!(!parts.accumulation.rhs_conj);
                match &mut *parts.output {
                    TensorWrite::Tensor(Tensor::F64(output)) => {
                        assert_eq!(output.host_data()?, &[41.0; 4]);
                        output
                            .host_data_mut()?
                            .copy_from_slice(&[19.0, 43.0, 22.0, 50.0]);
                    }
                    other => panic!("unexpected fallback output: {other:?}"),
                }
            }
            CanonicalFallbackBehavior::ConjugatedComplex => {
                assert!(!parts.accumulation.lhs_conj);
                assert!(!parts.accumulation.rhs_conj);
                assert_eq!(
                    parts.accumulation.alpha,
                    ContractionScalar::C64(Complex64::new(2.0, 0.0)),
                );
                assert_eq!(
                    parts.accumulation.beta,
                    ContractionScalar::C64(Complex64::new(3.0, 0.0)),
                );
                match parts.lhs {
                    TensorRead::Tensor(Tensor::C64(lhs)) => {
                        assert_eq!(lhs.host_data()?, &[Complex64::new(1.0, -2.0)]);
                    }
                    other => panic!("conjugated lhs was not materialized: {other:?}"),
                }
                match parts.rhs {
                    TensorRead::Tensor(Tensor::C64(rhs)) => {
                        assert_eq!(rhs.host_data()?, &[Complex64::new(3.0, 4.0)]);
                    }
                    other => panic!("rhs was not materialized: {other:?}"),
                }
                match &mut *parts.output {
                    TensorWrite::Tensor(Tensor::C64(output)) => {
                        assert_eq!(output.host_data()?, &[Complex64::new(5.0, 1.0)]);
                        output.host_data_mut()?[0] = Complex64::new(37.0, -1.0);
                    }
                    other => panic!("unexpected fallback output: {other:?}"),
                }
            }
        }
        Ok(CpuProviderOutcome::Executed)
    }
}

impl CpuGemmProvider for CanonicalFallbackSpy {
    fn gemm(
        &self,
        _context: &CpuProviderContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.execute(request)
    }

    fn strided_batched_gemm(
        &self,
        _context: &CpuProviderContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.execute(request)
    }

    fn grouped_gemm(
        &self,
        _context: &CpuProviderContext<'_>,
        _request: CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        panic!("canonical fallback tests do not issue grouped GEMM")
    }
}

#[derive(Debug)]
struct LayoutSpy {
    calls: Arc<Mutex<usize>>,
}

impl CpuLayoutTransformProvider for LayoutSpy {
    fn materialize(
        &self,
        context: &CpuProviderContext<'_>,
        request: CpuLayoutTransformRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        *self.calls.lock().unwrap() += 1;
        StridedLayoutTransformProvider.materialize(context, request)
    }
}

fn route_operands() -> (Tensor, Tensor, Tensor, DotGeneralConfig) {
    (
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap(),
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap(),
        Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64; 4]).unwrap(),
        config(&[1], &[0], &[], &[]),
    )
}

fn route_bundle(
    gemm: Arc<dyn CpuGemmProvider>,
    general: Option<(Arc<dyn CpuGeneralContractionProvider>, bool)>,
) -> CpuProviderBundle {
    let builder = CpuProviderBundle::custom_builder()
        .gemm_provider(gemm)
        .engine_outer_grouped_gemm()
        .layout_transform_provider(Arc::new(StridedLayoutTransformProvider));
    match general {
        Some((provider, true)) => builder.require_general_contraction_provider(provider),
        Some((provider, false)) => builder.prefer_general_contraction_provider(provider),
        None => builder,
    }
    .build()
    .unwrap()
}

#[test]
fn route_general_executed_short_circuits_gemm() {
    let general_calls = Arc::new(Mutex::new(0));
    let general = Arc::new(GeneralSpy {
        behavior: GeneralBehavior::Outcome(CpuProviderOutcome::Executed),
        calls: Arc::clone(&general_calls),
    });
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Executed));
    let bundle = route_bundle(gemm.clone(), Some((general, false)));
    let (lhs, rhs, mut output, config) = route_operands();
    bundle
        .execute_dot_general_into(
            &CpuContext::with_threads(1).unwrap(),
            &mut BufferPool::new(),
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap();
    assert_eq!(*general_calls.lock().unwrap(), 1);
    assert_eq!(*gemm.gemm_calls.lock().unwrap(), 0);
}

#[test]
fn route_general_unsupported_falls_back_only_when_preferred() {
    let general = Arc::new(GeneralSpy {
        behavior: GeneralBehavior::Outcome(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::Layout(crate::provider::CpuOperand::Lhs),
        )),
        calls: Arc::new(Mutex::new(0)),
    });
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Executed));
    let bundle = route_bundle(gemm.clone(), Some((general, false)));
    let (lhs, rhs, mut output, config) = route_operands();
    bundle
        .execute_dot_general_into(
            &CpuContext::with_threads(1).unwrap(),
            &mut BufferPool::new(),
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap();
    assert_eq!(*gemm.gemm_calls.lock().unwrap(), 1);
}

#[test]
fn route_general_error_is_terminal() {
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Executed));
    let general = Arc::new(GeneralSpy {
        behavior: GeneralBehavior::Error,
        calls: Arc::new(Mutex::new(0)),
    });
    let bundle = route_bundle(gemm.clone(), Some((general, false)));
    let (lhs, rhs, mut output, config) = route_operands();
    let error = bundle
        .execute_dot_general_into(
            &CpuContext::with_threads(1).unwrap(),
            &mut BufferPool::new(),
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap_err();
    assert!(error.to_string().contains("general spy failure"));
    assert_eq!(*gemm.gemm_calls.lock().unwrap(), 0);
}

#[test]
fn route_required_general_unsupported_is_typed_and_terminal() {
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Executed));
    let general = Arc::new(GeneralSpy {
        behavior: GeneralBehavior::Outcome(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::RuntimeUnavailable,
        )),
        calls: Arc::new(Mutex::new(0)),
    });
    let bundle = route_bundle(gemm.clone(), Some((general, true)));
    let (lhs, rhs, mut output, config) = route_operands();
    let error = bundle
        .execute_dot_general_into(
            &CpuContext::with_threads(1).unwrap(),
            &mut BufferPool::new(),
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap_err();
    assert_eq!(error.kind(), tenferro_tensor::ErrorKind::Unsupported);
    assert_eq!(*gemm.gemm_calls.lock().unwrap(), 0);
}

#[test]
fn route_gemm_unsupported_is_terminal() {
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Unsupported(
        CpuProviderUnsupported::DType(DType::F64),
    )));
    let bundle = route_bundle(gemm.clone(), None);
    let (lhs, rhs, mut output, config) = route_operands();
    let error = bundle
        .execute_dot_general_into(
            &CpuContext::with_threads(1).unwrap(),
            &mut BufferPool::new(),
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap_err();
    assert_eq!(error.kind(), tenferro_tensor::ErrorKind::Unsupported);
    assert_eq!(*gemm.gemm_calls.lock().unwrap(), 1);
    assert_eq!(output.as_slice::<f64>().unwrap(), &[0.0; 4]);
}

#[test]
fn route_gemm_output_layout_unsupported_is_terminal_without_mutation() {
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Unsupported(
        CpuProviderUnsupported::Layout(CpuOperand::Output),
    )));
    let bundle = route_bundle(gemm.clone(), None);
    let (lhs, rhs, mut output, config) = route_operands();

    let error = bundle
        .execute_dot_general_into(
            &CpuContext::with_threads(1).unwrap(),
            &mut BufferPool::new(),
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap_err();

    assert_eq!(error.kind(), tenferro_tensor::ErrorKind::Unsupported);
    assert_eq!(*gemm.gemm_calls.lock().unwrap(), 1);
    assert_eq!(output.as_slice::<f64>().unwrap(), &[0.0; 4]);
}

#[test]
fn route_gemm_provider_error_is_terminal_without_mutation() {
    let gemm = Arc::new(GemmSpy::error());
    let bundle = route_bundle(gemm.clone(), None);
    let (lhs, rhs, mut output, config) = route_operands();

    let error = bundle
        .execute_dot_general_into(
            &CpuContext::with_threads(1).unwrap(),
            &mut BufferPool::new(),
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap_err();

    assert_eq!(error.kind(), tenferro_tensor::ErrorKind::RuntimeState);
    assert!(error.to_string().contains("GEMM provider spy failure"));
    assert_eq!(*gemm.gemm_calls.lock().unwrap(), 1);
    assert_eq!(output.as_slice::<f64>().unwrap(), &[0.0; 4]);
}

#[test]
fn route_gemm_canonical_retry_unsupported_is_terminal_without_mutation() {
    for second_outcome in [
        CpuProviderUnsupported::Layout(CpuOperand::Lhs),
        CpuProviderUnsupported::Conjugation,
        CpuProviderUnsupported::DType(DType::F64),
    ] {
        let gemm = Arc::new(CanonicalFallbackSpy::rejecting_retry(
            CpuProviderUnsupported::Layout(CpuOperand::Rhs),
            second_outcome,
        ));
        let bundle = route_bundle(gemm.clone(), None);
        let lhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();
        let rhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();
        let mut output = Tensor::from_vec_col_major(vec![2, 2], vec![41.0_f64; 4]).unwrap();
        let mut buffers = BufferPool::new();
        <f64 as PoolScalar>::pool_release(&mut buffers, vec![0.0; 4]);
        <f64 as PoolScalar>::pool_release(&mut buffers, vec![0.0; 4]);
        let seed_stats = buffers.stats();

        let error = bundle
            .execute_dot_general_into(
                &CpuContext::with_threads(1).unwrap(),
                &mut buffers,
                &mut GemmAnalysisCache::default(),
                None,
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &config(&[1], &[0], &[], &[]),
                DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
                TensorWrite::from_tensor(&mut output),
            )
            .unwrap_err();

        assert_eq!(error.kind(), tenferro_tensor::ErrorKind::Unsupported);
        assert_eq!(*gemm.calls.lock().unwrap(), 2);
        assert_eq!(output.as_slice::<f64>().unwrap(), &[41.0; 4]);
        assert_eq!(buffers.stats(), seed_stats);
    }
}

#[test]
fn route_gemm_canonical_retry_error_reclaims_both_materializations() {
    let gemm = Arc::new(CanonicalFallbackSpy::erroring_retry(
        CpuProviderUnsupported::Layout(CpuOperand::Rhs),
    ));
    let bundle = route_bundle(gemm.clone(), None);
    let lhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![2, 2], vec![41.0_f64; 4]).unwrap();
    let mut buffers = BufferPool::new();
    <f64 as PoolScalar>::pool_release(&mut buffers, vec![0.0; 4]);
    <f64 as PoolScalar>::pool_release(&mut buffers, vec![0.0; 4]);
    let seed_stats = buffers.stats();

    let error = bundle
        .execute_dot_general_into(
            &CpuContext::with_threads(1).unwrap(),
            &mut buffers,
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config(&[1], &[0], &[], &[]),
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap_err();

    assert_eq!(error.kind(), tenferro_tensor::ErrorKind::RuntimeState);
    assert!(error.to_string().contains("canonical retry spy failure"));
    assert_eq!(*gemm.calls.lock().unwrap(), 2);
    assert_eq!(output.as_slice::<f64>().unwrap(), &[41.0; 4]);
    assert_eq!(buffers.stats(), seed_stats);
}

fn assert_layout_unsupported_materializes_and_retries(operand: CpuOperand) {
    let gemm = Arc::new(CanonicalFallbackSpy::new(
        CpuProviderUnsupported::Layout(operand),
        CanonicalFallbackBehavior::Real,
    ));
    let bundle = route_bundle(gemm.clone(), None);
    let lhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 7.0, 6.0, 8.0]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![2, 2], vec![41.0_f64; 4]).unwrap();
    let mut buffers = BufferPool::new();
    <f64 as PoolScalar>::pool_release(&mut buffers, vec![0.0; 4]);
    <f64 as PoolScalar>::pool_release(&mut buffers, vec![0.0; 4]);
    let seed_stats = buffers.stats();

    bundle
        .execute_dot_general_into(
            &CpuContext::with_threads(1).unwrap(),
            &mut buffers,
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config(&[1], &[0], &[], &[]),
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap();

    assert_eq!(*gemm.calls.lock().unwrap(), 2);
    assert_eq!(output.as_slice::<f64>().unwrap(), &[19.0, 43.0, 22.0, 50.0]);
    assert_eq!(buffers.stats(), seed_stats);
}

#[test]
fn route_gemm_lhs_layout_unsupported_materializes_and_retries() {
    assert_layout_unsupported_materializes_and_retries(CpuOperand::Lhs);
}

#[test]
fn route_gemm_rhs_layout_unsupported_materializes_and_retries() {
    assert_layout_unsupported_materializes_and_retries(CpuOperand::Rhs);
}

#[test]
fn route_gemm_conjugation_unsupported_materializes_conjugate_and_retries() {
    let gemm = Arc::new(CanonicalFallbackSpy::new(
        CpuProviderUnsupported::Conjugation,
        CanonicalFallbackBehavior::ConjugatedComplex,
    ));
    let bundle = route_bundle(gemm.clone(), None);
    let lhs = Tensor::from_vec_col_major(vec![1, 1], vec![Complex64::new(1.0, 2.0)]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1, 1], vec![Complex64::new(3.0, 4.0)]).unwrap();
    let mut output =
        Tensor::from_vec_col_major(vec![1, 1], vec![Complex64::new(5.0, 1.0)]).unwrap();
    let mut accumulation = DotGeneralAccumulation::scaled(
        ContractionScalar::C64(Complex64::new(2.0, 0.0)),
        ContractionScalar::C64(Complex64::new(3.0, 0.0)),
    )
    .unwrap();
    accumulation.lhs_conj = true;

    bundle
        .execute_dot_general_into(
            &CpuContext::with_threads(1).unwrap(),
            &mut BufferPool::new(),
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config(&[1], &[0], &[], &[]),
            accumulation,
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap();

    assert_eq!(*gemm.calls.lock().unwrap(), 2);
    assert_eq!(
        output.as_slice::<Complex64>().unwrap(),
        &[Complex64::new(37.0, -1.0)],
    );
}

#[test]
fn route_canonical_fallback_uses_layout_slot_before_gemm() {
    let layout_calls = Arc::new(Mutex::new(0));
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Executed));
    let bundle = CpuProviderBundle::custom_builder()
        .gemm_provider(gemm.clone())
        .layout_transform_provider(Arc::new(LayoutSpy {
            calls: Arc::clone(&layout_calls),
        }))
        .build()
        .unwrap();
    let lhs = Tensor::from_vec_col_major(vec![2, 2, 2, 2], vec![1.0_f64; 16]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2, 2, 2, 2], vec![1.0_f64; 16]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![2, 2, 2, 2], vec![0.0_f64; 16]).unwrap();

    bundle
        .execute_dot_general_into(
            &CpuContext::with_threads(1).unwrap(),
            &mut BufferPool::new(),
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config(&[1, 3], &[2, 1], &[], &[]),
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap();

    assert_eq!(*layout_calls.lock().unwrap(), 2);
    assert_eq!(*gemm.gemm_calls.lock().unwrap(), 1);
}

#[test]
fn route_strided_batch_allows_inner_parallelism() {
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Executed));
    let bundle = route_bundle(gemm.clone(), None);
    let lhs = Tensor::from_vec_col_major(vec![2, 2, 2], vec![1.0_f64; 8]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2, 2, 2], vec![1.0_f64; 8]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![2, 2, 2], vec![0.0_f64; 8]).unwrap();
    bundle
        .execute_dot_general_into(
            &CpuContext::with_threads(2).unwrap(),
            &mut BufferPool::new(),
            &mut GemmAnalysisCache::default(),
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config(&[1], &[0], &[2], &[2]),
            DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap();
    assert_eq!(*gemm.strided_calls.lock().unwrap(), 1);
    assert_eq!(
        gemm.parallelism.lock().unwrap().as_slice(),
        &[CpuKernelParallelism::Inner]
    );
}

#[test]
fn route_engine_outer_grouped_runs_single_jobs_sequentially_in_selected_pool() {
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Executed));
    let bundle = route_bundle(gemm.clone(), None);
    let lhs = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2], vec![4.0_f64, 5.0]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2]).unwrap();
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 1, 1, 1),
        GroupedGemmJob::new(1, 1, 1, 1, 1, 1),
    ];
    bundle
        .execute_grouped_gemm(
            &CpuContext::with_threads(2).unwrap(),
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &GroupedGemmConfig::new(
                &jobs,
                DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            ),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap();
    assert_eq!(*gemm.grouped_calls.lock().unwrap(), 2);
    assert_eq!(gemm.grouped_job_counts.lock().unwrap().as_slice(), &[1, 1]);
    assert_eq!(
        gemm.parallelism.lock().unwrap().as_slice(),
        &[
            CpuKernelParallelism::Sequential,
            CpuKernelParallelism::Sequential,
        ]
    );
    assert!(gemm
        .in_selected_pool
        .lock()
        .unwrap()
        .iter()
        .all(|inside| *inside));
}

#[test]
fn route_provider_owned_grouped_delegates_whole_group_with_inner_parallelism() {
    let gemm = Arc::new(GemmSpy::new(CpuProviderOutcome::Executed));
    let bundle = CpuProviderBundle::builder(CpuBackendKind::Blas)
        .gemm_provider(gemm.clone())
        .build()
        .unwrap();
    let lhs = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2], vec![4.0_f64, 5.0]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2]).unwrap();
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 1, 1, 1),
        GroupedGemmJob::new(1, 1, 1, 1, 1, 1),
    ];
    bundle
        .execute_grouped_gemm(
            &CpuContext::with_threads(2).unwrap(),
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &GroupedGemmConfig::new(
                &jobs,
                DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
            ),
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap();

    assert_eq!(*gemm.grouped_calls.lock().unwrap(), 1);
    assert_eq!(gemm.grouped_job_counts.lock().unwrap().as_slice(), &[2]);
    assert_eq!(
        gemm.parallelism.lock().unwrap().as_slice(),
        &[CpuKernelParallelism::Inner]
    );
    assert_eq!(gemm.in_selected_pool.lock().unwrap().as_slice(), &[false]);
}
