//! The four configurations #1790 requires.
//!
//! - **Standard**: canonical support alone, assembled from the repository's own linalg
//!   module.
//! - **Df64-only**: the contribution's numerical support alone, with no linalg support
//!   installed.
//! - **Mixed**: one backend that carries both supports in one runtime.
//! - **Cooperating**: two CPU backends under distinct engine identities in one runtime,
//!   the standard family served by one and the contribution's family by the other.
//!
//! Every case runs the same algorithm from the algorithm crate and reaches the same
//! number in its own dtype.

use std::sync::Arc;

use tenferro_ad::semantic_extension::SemanticExtensionRuleSet;
use tenferro_ad::AdContext;
use tenferro_cpu::{runtime_engine_registration, runtime_engine_registration_with_id, CpuBackend};
use tenferro_df64_proof::ad::Df64VjpRule;
use tenferro_df64_proof::extension::{
    module, module_for_engine, Df64Qr, Df64ToF64, DF64_SCALAR_IDENTITY,
};
use tenferro_df64_proof::Df64;
use tenferro_linalg::TracedTensorLinalgExt;
use tenferro_runtime::extension::apply;
use tenferro_runtime::{EngineId, Error, ErrorPhase, GraphCompiler, Runtime, TracedTensor};
use tenferro_scalar_consumer_algorithm::{factor_norm_gradient, ScalarSupport};
use tenferro_tensor::{DType, Tensor};
use tenferro_tensor_core::{ErasedHostTensor, HostTensor};

/// Canonical standard support.
struct Standard;

impl ScalarSupport for Standard {
    fn qr(&self, input: &TracedTensor) -> Result<(TracedTensor, TracedTensor), Error> {
        input.qr()
    }

    fn to_f64(&self, input: &TracedTensor) -> Result<TracedTensor, Error> {
        Ok(input.clone())
    }
}

/// The external scalar contribution.
struct Extended;

impl ScalarSupport for Extended {
    fn qr(&self, input: &TracedTensor) -> Result<(TracedTensor, TracedTensor), Error> {
        let mut outputs = apply(Arc::new(Df64Qr), &[input])?.into_iter();
        let q = outputs.next().ok_or_else(|| {
            Error::runtime_state("Extended::qr", ErrorPhase::GraphBuild, "no factor")
        })?;
        let r = outputs.next().ok_or_else(|| {
            Error::runtime_state(
                "Extended::qr",
                ErrorPhase::GraphBuild,
                "no triangular factor",
            )
        })?;
        Ok((q, r))
    }

    fn to_f64(&self, input: &TracedTensor) -> Result<TracedTensor, Error> {
        Ok(apply(Arc::new(Df64ToF64), &[input])?.remove(0))
    }
}

fn standard_context() -> AdContext {
    AdContext::builder()
        .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules().expect("linalg rules"))
        .expect("extension rules")
        .build()
        .expect("ad context")
}

fn extended_context() -> AdContext {
    let rules = SemanticExtensionRuleSet::new()
        .with_primal_vjp(Arc::new(Df64VjpRule))
        .expect("one rule per family");
    AdContext::builder()
        .with_semantic_extension_rules(rules)
        .expect("extension rules")
        .build()
        .expect("ad context")
}

fn standard_input() -> TracedTensor {
    TracedTensor::from_tensor_concrete_shape(
        Tensor::from_vec_col_major(vec![2, 1], vec![3.0_f64, 4.0]).expect("shape matches data"),
    )
    .expect("traced input")
}

fn extended_input() -> TracedTensor {
    TracedTensor::from_tensor_concrete_shape_declaring_scalar(
        Tensor::external(ErasedHostTensor::new(
            HostTensor::from_vec_col_major(
                vec![2, 1],
                vec![Df64::from_f64(3.0), Df64::from_f64(4.0)],
            )
            .expect("shape matches data"),
        )),
        DF64_SCALAR_IDENTITY,
    )
    .expect("traced input")
}

fn run<S: ScalarSupport>(
    support: &S,
    ad: &AdContext,
    input: TracedTensor,
    runtime: &Runtime,
) -> Tensor {
    let seed = TracedTensor::from_tensor_concrete_shape(
        Tensor::from_vec_col_major(vec![], vec![1.0_f64]).expect("shape matches data"),
    )
    .expect("traced seed");
    let gradient = factor_norm_gradient(ad, &input, &seed, support).expect("the program runs");
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&gradient).expect("compiled program");
    let mut results = runtime
        .run_compiled(&program, &[])
        .expect("executed the program")
        .into_iter();
    results.next().expect("one result")
}

fn assert_standard_gradient(gradient: &Tensor) {
    assert_eq!(gradient.dtype(), DType::F64);
    let values = gradient.as_slice::<f64>().expect("f64 slice");
    assert!((values[0] - 6.0).abs() < 1e-12, "{values:?}");
    assert!((values[1] - 8.0).abs() < 1e-12, "{values:?}");
}

fn assert_extended_gradient(gradient: &Tensor) {
    assert_eq!(
        gradient.dtype(),
        DType::External(std::any::TypeId::of::<Df64>())
    );
    let Tensor::External(payload, _) = gradient else {
        panic!("expected an externally defined gradient");
    };
    let values = payload
        .downcast_ref::<Df64>()
        .expect("external element type")
        .as_slice();
    assert!(
        (values[0] - Df64::from_f64(6.0)).abs_hi() < 1e-30,
        "{values:?}"
    );
    assert!(
        (values[1] - Df64::from_f64(8.0)).abs_hi() < 1e-30,
        "{values:?}"
    );
}

/// Standard support assembled from the repository's linalg module.
fn standard_runtime() -> Runtime {
    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder
        .register_engine(runtime_engine_registration(&backend).expect("engine"))
        .expect("register the CPU engine");
    builder
        .install_extension_module(
            tenferro_linalg::extension_module::<CpuBackend>(
                tenferro_cpu::runtime_engine_id().expect("engine id"),
            )
            .expect("linalg module"),
        )
        .expect("install the linalg module");
    builder.build().expect("runtime")
}

#[test]
fn the_standard_configuration_runs_the_algorithm() {
    let runtime = standard_runtime();
    let gradient = run(&Standard, &standard_context(), standard_input(), &runtime);
    assert_standard_gradient(&gradient);
}

#[test]
fn the_df64_only_configuration_needs_no_linalg_support() {
    // Only the contribution's module is installed, so nothing standard is available.
    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder
        .register_engine(runtime_engine_registration(&backend).expect("engine"))
        .expect("register the CPU engine");
    builder
        .install_extension_module(module().expect("module"))
        .expect("install the Df64 module");
    let runtime = builder.build().expect("runtime");

    let gradient = run(&Extended, &extended_context(), extended_input(), &runtime);
    assert_extended_gradient(&gradient);
}

#[test]
fn the_mixed_configuration_carries_both_supports_in_one_runtime() {
    let backend = CpuBackend::new();
    let engine_id = tenferro_cpu::runtime_engine_id().expect("engine id");
    let mut builder = Runtime::builder();
    builder
        .register_engine(runtime_engine_registration(&backend).expect("engine"))
        .expect("register the CPU engine");
    builder
        .install_extension_module(
            tenferro_linalg::extension_module::<CpuBackend>(engine_id.clone())
                .expect("linalg module"),
        )
        .expect("install the linalg module");
    builder
        .install_extension_module(module().expect("module"))
        .expect("install the Df64 module");
    let runtime = builder.build().expect("runtime");

    assert_standard_gradient(&run(
        &Standard,
        &standard_context(),
        standard_input(),
        &runtime,
    ));
    assert_extended_gradient(&run(
        &Extended,
        &extended_context(),
        extended_input(),
        &runtime,
    ));
}

/// Two CPU resource domains in one runtime, each serving one family.
///
/// This is the configuration #1790 calls "distinct cooperating backends": the standard
/// family is bound to one engine and the contribution's family to another, so the two
/// connected programs meet in one runtime without sharing an engine. The identities
/// differ by resource domain, which is how the runtime tells two CPU owners apart.
#[test]
fn the_cooperating_configuration_serves_each_family_from_its_own_engine() {
    use std::num::NonZeroUsize;
    use std::sync::Arc;

    use tenferro_cpu::{
        discover_cpu_topology, CpuContext, CpuPlacementGuarantee, ExternalCpuDomain,
        ResolvedCpuPlacement,
    };
    use tenferro_tensor::CpuDomainId;

    let topology = discover_cpu_topology().expect("CPU topology");
    if topology.nodes().len() < 2 {
        // One resource domain is all this host declares, so the two-owner shape cannot be
        // built here; the single-domain configurations above still run.
        return;
    }
    let domain = |id: u64, node: &tenferro_cpu::CpuNode| {
        let domain_id = CpuDomainId::new(id);
        (
            domain_id,
            ExternalCpuDomain::new(
                domain_id,
                ResolvedCpuPlacement::NumaNode {
                    id: node.id(),
                    cpus: node.cpus().clone(),
                },
                Arc::new(CpuContext::with_threads(1).expect("CPU context")),
                NonZeroUsize::new(1).expect("nonzero"),
                CpuPlacementGuarantee::AdvisoryDeclared,
            )
            .expect("external CPU domain"),
        )
    };
    let (standard_domain_id, standard_domain) = domain(301, &topology.nodes()[0]);
    let (contribution_domain_id, contribution_domain) = domain(302, &topology.nodes()[1]);
    let standard_backend =
        CpuBackend::from_external_managed_domains(standard_domain_id, [standard_domain])
            .expect("standard backend");
    let contribution_backend =
        CpuBackend::from_external_managed_domains(contribution_domain_id, [contribution_domain])
            .expect("contribution backend");

    let standard_engine = EngineId::new("tenferro-cpu.standard.v1").expect("engine id");
    let contribution_engine = EngineId::new("tenferro-cpu.contribution.v1").expect("engine id");
    let mut builder = Runtime::builder();
    builder
        .register_engine(
            runtime_engine_registration_with_id(&standard_backend, standard_engine.clone())
                .expect("standard engine"),
        )
        .expect("register the standard engine");
    builder
        .register_engine(
            runtime_engine_registration_with_id(&contribution_backend, contribution_engine.clone())
                .expect("contribution engine"),
        )
        .expect("register the contribution engine");
    builder
        .install_extension_module(
            tenferro_linalg::extension_module::<CpuBackend>(standard_engine)
                .expect("linalg module"),
        )
        .expect("install the linalg module");
    builder
        .install_extension_module(
            module_for_engine(contribution_engine).expect("contribution module"),
        )
        .expect("install the contribution module");
    let runtime = builder.build().expect("runtime with two CPU engines");

    assert_standard_gradient(&run(
        &Standard,
        &standard_context(),
        standard_input(),
        &runtime,
    ));
    assert_extended_gradient(&run(
        &Extended,
        &extended_context(),
        extended_input(),
        &runtime,
    ));
}
