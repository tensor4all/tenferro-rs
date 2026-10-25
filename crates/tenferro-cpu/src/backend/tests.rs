use super::*;
use crate::buffer_pool::{BufferPool, PoolScalar};
use crate::test_support::with_cpu_session;
use strided_exec::Exec;
use tenferro_tensor::{DotGeneralConfig, Tensor, TensorRead, TensorWrite, TypedTensor};

mod batch_policy;

#[test]
fn cpu_tensor_kernel_parallel_features_are_wired() {
    let workspace_manifest = include_str!("../../../../Cargo.toml");
    let cpu_manifest = include_str!("../../Cargo.toml");

    let strided_kernel_line = workspace_manifest
        .lines()
        .find(|line| line.trim_start().starts_with("strided-kernel ="))
        .expect("workspace manifest should declare strided-kernel");
    assert!(
        strided_kernel_line.contains("features")
            && strided_kernel_line.contains("\"parallel\""),
        "workspace strided-kernel dependency must enable the parallel feature: {strided_kernel_line}"
    );

    assert!(
        !workspace_manifest.contains("strided-einsum2")
            && !cpu_manifest.contains("strided-einsum2"),
        "tenferro-rs must not retain the removed strided-einsum2 dependency"
    );
}

#[test]
fn native_kernel_modules_take_placement_only_from_the_token() {
    let provider = include_str!("../provider.rs");
    let native_modules = [
        ("analytic", include_str!("../analytic.rs")),
        (
            "elementwise",
            include_str!("../../../tenferro-internal-cpu-kernels/src/elementwise.rs"),
        ),
        (
            "fused",
            include_str!("../../../tenferro-cpu-fused/src/lib.rs"),
        ),
        ("indexing", include_str!("../indexing.rs")),
        ("reduction", include_str!("../reduction.rs")),
        ("structural", include_str!("../structural.rs")),
    ];

    // No thread-local or ambient placement exists in the CPU stack: kernels take
    // the placement token and nothing else decides where they run.
    for (name, source) in native_modules.iter().chain([("provider", provider)].iter()) {
        assert!(
            !source.contains("ExecutionPolicy")
                && !source.contains("with_execution_policy(")
                && !source.contains("with_execution_pool(")
                && !source.contains("ExecContext")
                && !source.contains("thread_local!"),
            "{name} must take its placement from the strided_exec::Exec token"
        );
        assert!(
            !source.contains("rayon::") && !source.contains("into_par_iter("),
            "{name} must not fan out through Rayon"
        );
    }
}

#[test]
fn cpu_hot_kernels_delegate_to_erased_strided_replay() {
    let elementwise = include_str!("../../../tenferro-internal-cpu-kernels/src/elementwise.rs");
    let fused = include_str!("../../../tenferro-cpu-fused/src/lib.rs");
    let indexing = include_str!("../indexing.rs");
    let reduction = include_str!("../reduction.rs");
    let structural = include_str!("../structural.rs");

    assert!(
        fused.contains("ErasedFusedPlan::compile"),
        "CPU fused adapter should delegate replay to strided-fused's erased plan"
    );
    assert!(
        !elementwise.contains("ErasedFusedPlan::compile"),
        "ordinary kernels must not instantiate fused replay"
    );
    assert!(
        indexing.contains("ErasedDynamicSlicePlan::compile")
            && indexing.contains("ErasedDynamicUpdateSlicePlan::compile")
            && indexing.contains("ErasedScatterPlan::compile"),
        "CPU indexed slice/update/scatter execution should delegate to erased strided plans"
    );
    assert!(
        indexing.contains("ErasedSlicePlan::compile")
            && indexing.contains("ErasedPadPlan::compile")
            && indexing.contains("ErasedConcatenatePlan::compile")
            && indexing.contains("ErasedReversePlan::compile"),
        "CPU static indexing execution should delegate to erased strided plans"
    );
    assert!(
        reduction.contains("ErasedReducePlan::compile_axes"),
        "CPU sum/product axis reductions should delegate to erased strided reduce plans"
    );
    for (name, source) in [
        ("elementwise", elementwise),
        ("indexing", indexing),
        ("reduction", reduction),
    ] {
        let compact_source = source.split_whitespace().collect::<String>();
        assert!(
            !compact_source.contains(".execute(Exec::serial()")
                && !compact_source
                    .contains(".execute_uninit(&mutout_ref,&input_ptr,Exec::serial()"),
            "{name} erased strided replay should take the engine's token, not force serial"
        );
    }
    let pooled_triangular = structural
        .split_once("fn typed_triangular_mask_with_fill_pool")
        .and_then(|(_, rest)| rest.split_once("fn checked_triangular_extent"))
        .map(|(body, _)| body)
        .expect("pooled triangular kernel should remain discoverable");
    assert!(
        !pooled_triangular.contains("for row in 0..rows")
            && pooled_triangular.contains("triangular_mask_from_view"),
        "CPU pooled triangular masks should delegate traversal to strided"
    );
    // The kernels hand the source layout to strided instead of a compact copy.
    assert!(structural.contains("strided_kernel::triangular_mask_view_into_uninit"));
    assert!(structural.contains("strided_kernel::embed_diagonal_view_into_uninit"));
    assert!(include_str!("../blas1.rs").contains("strided_kernel::axpby_accum"));
}

// The shared native-participant fixture lives in the broad unit suite, which
// `provider-inject` excludes (its symbols are registered by the integration
// fixture instead).
#[cfg(not(feature = "provider-inject"))]
#[test]
fn explicit_native_operation_uses_the_selected_rayon_budget() {
    let mut backend = crate::test_support::cpu_engine(2).unwrap();
    let caller = std::thread::current().id();
    let participants = with_cpu_session(&mut backend, |cpu| {
        cpu.with_linalg_pool(|exec, _| {
            assert_eq!(std::thread::current().id(), caller);
            Ok(crate::tests::native_participants::run_unscoped_native_map(
                true,
                exec.token(),
            ))
        })
    })
    .unwrap();

    assert_eq!(participants.max_active(), 2);
    assert_eq!(participants.thread_count(), 2);
}

#[test]
fn with_linalg_pool_restores_backend_pool_and_context() {
    let mut backend = crate::test_support::cpu_engine(1).unwrap();

    let len_inside_pool = with_cpu_session(&mut backend, |cpu| {
        cpu.with_linalg_pool(|context, pool| {
            assert_eq!(context.budget(), 1);
            <f64 as PoolScalar>::pool_release(pool, vec![1.0, 2.0, 3.0, 4.0]);
            Ok(pool.len())
        })
    })
    .unwrap();

    assert_eq!(len_inside_pool, 1);
    assert_eq!(backend.buffer_pool_len(), 1);
}

#[test]
fn linalg_pool_acquire_then_panic_breaks_engine() {
    let mut backend = crate::test_support::cpu_engine(1).unwrap();
    with_cpu_session(&mut backend, |cpu| {
        cpu.with_linalg_pool(|_, pool| {
            <f64 as PoolScalar>::pool_release(pool, Vec::with_capacity(1024));
            Ok(())
        })
    })
    .unwrap();
    assert_eq!(backend.buffer_pool_len(), 1);
    assert_eq!(
        backend.buffer_pool_stats().capacity_bytes,
        1024 * std::mem::size_of::<f64>()
    );

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = with_cpu_session(&mut backend, |cpu| {
            cpu.with_linalg_pool::<()>(|_, pool| {
                let _in_flight = pool.acquire_with_capacity::<f64>(1024);
                assert_eq!(pool.retained_capacity_bytes(), 0);
                panic!("forced panic after pool acquisition");
            })
        });
    }));

    assert!(result.is_err());
    // The engine is marked broken because its workspace may be half updated;
    // it is never used again, so the buffer the callback took is not repaired.
    assert!(backend.is_broken());
    assert!(with_cpu_session(&mut backend, |cpu| cpu.with_linalg_pool(|_, _| Ok(()))).is_err());
}

#[test]
fn uninit_output_partial_write_then_panic_returns_without_replenishment() {
    let mut backend = crate::test_support::cpu_engine(1).unwrap();
    with_cpu_session(&mut backend, |cpu| {
        cpu.with_linalg_pool(|_, pool| {
            <bool as PoolScalar>::pool_release(pool, Vec::with_capacity(1024));
            Ok(())
        })
    })
    .unwrap();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = with_cpu_session(&mut backend, |cpu| {
            cpu.with_linalg_pool::<()>(|_, pool| {
                let mut output = crate::PooledUninitOutput::<bool>::new(pool, vec![1024]).unwrap();
                output.as_uninit_bytes_mut()[0].write(1);
                panic!("forced panic after a partial uninitialized output write");
            })
        });
    }));

    assert!(result.is_err());
    // The aborted destination hands its storage back, so no replacement has to be manufactured.
    assert_eq!(backend.buffer_pool_len(), 1);
    assert_eq!(
        backend.buffer_pool_stats().capacity_bytes,
        1024 * std::mem::size_of::<bool>()
    );
}

#[test]
fn cached_dot_dispatch_reports_dtype_mismatches() {
    let mut backend = crate::CpuEngine::serial();
    let lhs =
        Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(vec![1], vec![1.0]).unwrap());
    let rhs =
        Tensor::from_typed::<f32>(TypedTensor::from_vec_col_major(vec![1], vec![1.0]).unwrap());
    let config = DotGeneralConfig {
        lhs_contracting_dims: [0].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    };

    let dot_error = crate::test_support::on_engine(&mut backend, |__s| {
        __s.dot_general_cached(Some(0), &lhs, &rhs, &config)
    })
    .unwrap();
    assert!(matches!(
        dot_error,
        Err(crate::Error::Validation {
            op: "dot_general",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        })
    ));

    let dot_conj_error = crate::test_support::on_engine(&mut backend, |__s| {
        __s.dot_general_with_conj_cached(Some(1), &lhs, &rhs, &config, true, false)
    })
    .unwrap();
    assert!(matches!(
        dot_conj_error,
        Err(crate::Error::Validation {
            op: "dot_general",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        })
    ));
}

#[test]
fn elementwise_into_fallback_returns_its_staged_result_to_the_pool() {
    let mut buffers = BufferPool::new();
    let ctx = Exec::serial();
    // The fallback (taken when the one-shot kernel does not apply) stages the
    // sum before copying it into `out`.
    let lhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2, 2], vec![10.0_f64, 20.0, 30.0, 40.0]).unwrap();
    let mut out = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64; 4]).unwrap();
    assert_eq!(buffers.len(), 0);

    elementwise_read_into_fallback_with_pool(
        &mut buffers,
        ElementwiseReadOp::Add,
        &[TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs)],
        TensorWrite::from_tensor(&mut out),
        ctx,
    )
    .unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap(), &[11.0, 22.0, 33.0, 44.0]);
    assert_eq!(buffers.len(), 1, "the staged result must be reclaimed");
}
