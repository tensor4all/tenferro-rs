use std::hint::black_box;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use tenferro_ad::EagerRuntime;
use tenferro_cpu::{
    process_cpu_affinity, CpuBackend, CpuContext, CpuPlacement, CpuPlacementGuarantee,
    ExternalCpuDomain, ResolvedCpuPlacement,
};
use tenferro_tensor::{CpuDomainId, DotGeneralConfig, Tensor};

fn tensor(shape: Vec<usize>, salt: usize) -> Tensor {
    let len = shape.iter().product();
    Tensor::from_vec_col_major(
        shape,
        (0..len)
            .map(|index| ((index * 19 + salt * 11) % 103) as f64 / 41.0 - 1.0)
            .collect(),
    )
    .unwrap()
}

fn phase2e_characterization(c: &mut Criterion) {
    let native_lhs = tensor(vec![65_536], 1);
    let native_rhs = tensor(vec![65_536], 2);
    let dot_lhs = tensor(vec![128, 128], 3);
    let dot_rhs = tensor(vec![128, 128], 4);
    let dot_config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    for ownership in ["managed-exact", "external-exact", "external-advisory"] {
        for budget in [1, 2, 4] {
            let runtime = EagerRuntime::with_cpu_backend(phase2e_backend(ownership, budget));
            let mut placed = runtime.on_cpu(CpuPlacement::AllAllowed).unwrap();
            c.bench_function(&format!("phase2e/{ownership}/budget-{budget}/E-N"), |b| {
                b.iter(|| {
                    black_box(
                        placed
                            .with_eager_session(|session| {
                                session
                                    .add(&native_lhs, &native_rhs)
                                    .map_err(tenferro_ad::Error::from)
                            })
                            .unwrap(),
                    )
                });
            });
            c.bench_function(&format!("phase2e/{ownership}/budget-{budget}/E-D"), |b| {
                b.iter(|| {
                    black_box(
                        placed
                            .with_eager_session(|session| {
                                session
                                    .dot_general(&dot_lhs, &dot_rhs, &dot_config)
                                    .map_err(tenferro_ad::Error::from)
                            })
                            .unwrap(),
                    )
                });
            });
        }
    }
}

fn phase2e_backend(ownership: &str, budget: usize) -> CpuBackend {
    if ownership == "managed-exact" {
        return CpuBackend::with_threads(budget).unwrap();
    }
    let allowed = process_cpu_affinity().expect("Phase 2E benchmark needs process affinity");
    let id = CpuDomainId::new(
        0x2eba + budget as u64 + u64::from(u8::from(ownership == "external-exact")),
    );
    let domain = ExternalCpuDomain::new(
        id,
        ResolvedCpuPlacement::AllAllowed { cpus: allowed },
        Arc::new(CpuContext::with_threads(budget).unwrap()),
        NonZeroUsize::new(budget).unwrap(),
        if ownership == "external-exact" {
            CpuPlacementGuarantee::ExactDeclared
        } else {
            CpuPlacementGuarantee::AdvisoryDeclared
        },
    )
    .unwrap();
    CpuBackend::from_external_managed_domains(id, [domain]).unwrap()
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5))
        .sample_size(100)
        .confidence_level(0.95);
    targets = phase2e_characterization
}
criterion_main!(benches);
