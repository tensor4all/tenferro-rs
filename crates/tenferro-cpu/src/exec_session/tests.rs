use super::*;
use crate::provider::tests::execution_context_fixture;

#[test]
fn native_operation_enters_the_selected_rayon_executor() {
    let fixture = execution_context_fixture(2);
    let mut buffers = BufferPool::new();
    let mut gemm_analysis_cache = gemm::GemmAnalysisCache::default();
    let providers = CpuProviderBundle::standard(crate::CpuBackendKind::default_compiled());
    let mut session = CpuExecSession {
        entry: fixture.entry(),
        buffers: &mut buffers,
        gemm_analysis_cache: &mut gemm_analysis_cache,
        providers: &providers,
    };

    assert!(rayon::current_thread_index().is_none());
    let (worker, pool_size, participants) = session
        .run_native(|_| {
            Ok((
                rayon::current_thread_index(),
                rayon::current_num_threads(),
                crate::provider::tests::run_unscoped_native_map(true),
            ))
        })
        .unwrap();
    assert!(matches!(worker, Some(0 | 1)));
    assert_eq!(pool_size, 2);
    assert_eq!(participants.max_active(), 2);
    assert_eq!(participants.thread_count(), 2);
    assert!(rayon::current_thread_index().is_none());
}
