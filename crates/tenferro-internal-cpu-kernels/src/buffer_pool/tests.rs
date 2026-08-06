use std::mem::size_of;

use super::{
    parse_default_max_retained_capacity_bytes, BufferPool, PoolScalar,
    DEFAULT_MAX_RETAINED_CAPACITY_BYTES,
};
use crate::PooledUninitOutput;

#[test]
fn default_retention_limit_parser_covers_missing_invalid_zero_and_valid_values() {
    assert_eq!(
        parse_default_max_retained_capacity_bytes(None),
        DEFAULT_MAX_RETAINED_CAPACITY_BYTES
    );
    assert_eq!(
        parse_default_max_retained_capacity_bytes(Some("invalid".into())),
        DEFAULT_MAX_RETAINED_CAPACITY_BYTES
    );
    assert_eq!(
        parse_default_max_retained_capacity_bytes(Some("0".into())),
        0
    );
    assert_eq!(
        parse_default_max_retained_capacity_bytes(Some("4096".into())),
        4096
    );
}

#[cfg(unix)]
#[test]
fn default_retention_limit_parser_rejects_non_unicode_values() {
    use std::os::unix::ffi::OsStringExt;

    assert_eq!(
        parse_default_max_retained_capacity_bytes(Some(std::ffi::OsString::from_vec(vec![0xff]))),
        DEFAULT_MAX_RETAINED_CAPACITY_BYTES
    );
}

#[test]
fn acquire_release_reuse() {
    let mut pool = BufferPool::new();

    let buf = pool.acquire_with_capacity::<f64>(64);
    let ptr = buf.as_ptr();
    let cap = buf.capacity();
    <f64 as PoolScalar>::pool_release(&mut pool, buf);

    let reused = pool.acquire_with_capacity::<f64>(64);
    assert_eq!(reused.as_ptr(), ptr);
    assert_eq!(reused.capacity(), cap);
    assert!(pool.is_empty());
}

#[test]
fn best_fit() {
    let mut pool = BufferPool::new();
    <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(100));
    <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(200));
    <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(300));

    let reused = pool.acquire_with_capacity::<f64>(150);
    assert_eq!(reused.capacity(), 200);
    assert_eq!(pool.len(), 2);
}

#[test]
fn type_separation() {
    let mut pool = BufferPool::new();
    <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(16));
    assert_eq!(pool.len(), 1);

    let f32_buf = pool.acquire_with_capacity::<f32>(16);
    assert_eq!(f32_buf.capacity(), 16);
    assert_eq!(pool.len(), 1);

    let f64_buf = pool.acquire_with_capacity::<f64>(16);
    assert_eq!(f64_buf.capacity(), 16);
    assert!(pool.is_empty());
}

#[test]
fn fresh_alloc_fallback() {
    let mut pool = BufferPool::new();
    let buf = pool.acquire_zeroed::<f64>(32);
    assert_eq!(buf.len(), 32);
    assert!(buf.capacity() >= 32);
    assert!(pool.is_empty());
}

#[test]
fn zeroed_acquire_initializes_fresh_and_reused_buffers() {
    let mut pool = BufferPool::new();

    let fresh = <f64 as PoolScalar>::pool_acquire_zeroed(&mut pool, 4);
    assert_eq!(fresh, vec![0.0; 4]);

    <f64 as PoolScalar>::pool_release(&mut pool, vec![7.0, 8.0, 9.0, 10.0]);
    let reused = <f64 as PoolScalar>::pool_acquire_zeroed(&mut pool, 4);
    assert_eq!(reused, vec![0.0; 4]);
}

#[test]
fn empty_checkout_does_not_expose_initialized_reused_values() {
    let mut pool = BufferPool::new();

    <f64 as PoolScalar>::pool_release(&mut pool, vec![7.0, 8.0, 9.0, 10.0]);
    let reused = pool.acquire_with_capacity::<f64>(4);

    assert_eq!(reused.len(), 0);
    assert!(reused.capacity() >= 4);
}

#[test]
fn zero_len_not_pooled() {
    let mut pool = BufferPool::new();
    <f64 as PoolScalar>::pool_release(&mut pool, Vec::new());
    assert!(pool.is_empty());
}

#[test]
fn acquire_with_capacity_reuses_buffer_as_empty_vec() {
    let mut pool = BufferPool::new();

    let buf = vec![1.0_f64; 8];
    let ptr = buf.as_ptr();
    let cap = buf.capacity();
    <f64 as PoolScalar>::pool_release(&mut pool, buf);

    let reused = pool.acquire_with_capacity::<f64>(8);
    assert_eq!(reused.as_ptr(), ptr);
    assert_eq!(reused.len(), 0);
    assert_eq!(reused.capacity(), cap);
    assert!(pool.is_empty());
}

#[test]
fn acquire_updates_retained_capacity_stats() {
    let mut pool = BufferPool::new();
    <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(8));
    assert_eq!(pool.retained_capacity_bytes(), 8 * size_of::<f64>());

    let _reused = pool.acquire_with_capacity::<f64>(4);

    assert_eq!(pool.retained_capacity_bytes(), 0);
    assert!(pool.is_empty());
}

#[test]
fn replenish_in_flight_retained_restores_lost_capacity() {
    let mut pool = BufferPool::new();
    <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(8));

    let _in_flight = pool.acquire_with_capacity::<f64>(8);
    assert_eq!(pool.retained_capacity_bytes(), 0);
    assert!(pool.is_empty());

    pool.replenish_in_flight_retained();

    assert_eq!(pool.len(), 1);
    assert_eq!(pool.retained_capacity_bytes(), 8 * size_of::<f64>());
    assert!(!pool.is_empty());
}

#[test]
fn replenish_in_flight_retained_skips_successfully_released_buffers() {
    let mut pool = BufferPool::new();
    <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(8));

    let buf = pool.acquire_with_capacity::<f64>(8);
    <f64 as PoolScalar>::pool_release(&mut pool, buf);

    let stats_before = pool.stats();
    pool.replenish_in_flight_retained();

    assert_eq!(pool.stats(), stats_before);
    assert_eq!(pool.len(), 1);
    assert_eq!(pool.retained_capacity_bytes(), 8 * size_of::<f64>());
}

#[test]
fn stats_counts_typed_capacity_bytes() {
    let mut pool = BufferPool::new();
    <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(3));
    <f32 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(5));
    <num_complex::Complex64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(7));

    let stats = pool.stats();
    assert_eq!(stats.buffers, 3);
    assert_eq!(
        stats.capacity_bytes,
        3 * size_of::<f64>() + 5 * size_of::<f32>() + 7 * size_of::<num_complex::Complex64>()
    );
    assert_eq!(pool.retained_capacity_bytes(), stats.capacity_bytes);
}

#[test]
fn clear_drops_retained_buffers() {
    let mut pool = BufferPool::new();
    <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(11));
    <f32 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(13));

    assert!(!pool.is_empty());
    assert!(pool.retained_capacity_bytes() > 0);

    pool.clear();

    assert!(pool.is_empty());
    assert_eq!(pool.stats(), Default::default());
}

#[test]
fn retention_limit_evicts_smallest_obsolete_buffers() {
    let mut pool = BufferPool::with_max_retained_capacity_bytes(200);
    <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(10));
    <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(20));

    assert_eq!(pool.stats().buffers, 1);
    assert_eq!(pool.retained_capacity_bytes(), 20 * size_of::<f64>());

    let reused = pool.acquire_with_capacity::<f64>(10);
    assert_eq!(reused.capacity(), 20);
    assert!(pool.is_empty());
}

#[test]
fn zero_retention_limit_drops_released_buffers() {
    let mut pool = BufferPool::with_max_retained_capacity_bytes(0);
    <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(10));

    assert!(pool.is_empty());
    assert_eq!(pool.retained_capacity_bytes(), 0);
}

#[test]
fn retention_limit_documents_zero_byte_eviction_progress() {
    let source = include_str!("../buffer_pool.rs");
    assert!(
        source.contains("evicted_bytes == 0"),
        "retention-limit eviction must explicitly handle zero-byte retained entries"
    );
}

#[test]
fn pooled_uninit_guard_discards_partial_reused_storage_without_replacement() {
    let mut pool = BufferPool::new();
    <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(8));
    let before = pool.stats();

    let mut output = PooledUninitOutput::<f64>::new(&mut pool, vec![3]).unwrap();
    assert!(output.token_is_reused_with_capacity(8));
    output.as_uninit_slice_mut()[0].write(1.0);
    drop(output);

    assert_eq!(pool.stats().buffers, before.buffers - 1);
    assert_eq!(
        pool.stats().capacity_bytes,
        before.capacity_bytes - 8 * size_of::<f64>()
    );
    assert!(pool.in_flight_is_empty());
}

#[test]
fn pooled_uninit_guard_fresh_handoff_and_panic_discard_are_exact() {
    let mut pool = BufferPool::new();
    {
        let mut output = PooledUninitOutput::<f64>::new(&mut pool, vec![2]).unwrap();
        assert!(output.token_is_fresh());
        output.as_uninit_slice_mut().iter_mut().for_each(|v| {
            v.write(3.0);
        });
        let tensor = unsafe { output.assume_init() }.unwrap();
        assert_eq!(tensor.as_slice().unwrap(), &[3.0, 3.0]);
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut output = PooledUninitOutput::<f64>::new(&mut pool, vec![4]).unwrap();
        output.as_uninit_slice_mut()[0].write(1.0);
        panic!("partial kernel panic");
    }));
    assert!(result.is_err());
    assert_eq!(pool.stats(), Default::default());
}

#[test]
fn pooled_uninit_guard_keeps_unrelated_dtype_markers_untouched() {
    let mut pool = BufferPool::new();
    <f64 as PoolScalar>::pool_release(&mut pool, vec![0.0; 8]);
    let unrelated = pool.acquire_with_capacity::<f64>(8);
    let marker_before = pool.f64_in_flight.clone();
    assert_eq!(marker_before.get(&8), Some(&1));
    <f64 as PoolScalar>::pool_release(&mut pool, vec![0.0; 16]);
    {
        let _output = PooledUninitOutput::<f64>::new(&mut pool, vec![3]).unwrap();
    }
    assert_eq!(pool.f64_in_flight, marker_before);
    drop(unrelated);
    pool.clear_in_flight_retained();
}

#[test]
fn pooled_uninit_guard_bool_invalid_byte_error_drops_without_typed_read() {
    let mut pool = BufferPool::new();
    <bool as PoolScalar>::pool_release(&mut pool, vec![false; 8]);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut output = PooledUninitOutput::<bool>::new(&mut pool, vec![3]).unwrap();
        output.as_uninit_bytes_mut()[0].write(2);
        panic!("invalid bool partial panic");
    }));
    assert!(result.is_err());
    assert!(pool.bool_in_flight.is_empty());
    assert_eq!(pool.retained_capacity_bytes(), 0);
    assert!(!pool.bool_pool.contains_key(&8));
}

#[test]
fn pooled_uninit_guard_reused_success_handoff_reclaims_exact_capacity() {
    let mut pool = BufferPool::new();
    <f64 as PoolScalar>::pool_release(&mut pool, vec![0.0; 8]);
    let mut output = PooledUninitOutput::<f64>::new(&mut pool, vec![3]).unwrap();
    assert!(output.token_is_reused_with_capacity(8));
    output
        .as_uninit_slice_mut()
        .iter_mut()
        .enumerate()
        .for_each(|(i, value)| {
            value.write(i as f64 + 1.0);
        });
    let tensor = unsafe { output.assume_init() }.unwrap();
    assert_eq!(tensor.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    assert_eq!(pool.retained_capacity_bytes(), 0);
    assert_eq!(pool.f64_in_flight.get(&8), Some(&1));
    pool.replenish_in_flight_retained();
    assert!(pool.f64_in_flight.is_empty());
    assert_eq!(pool.retained_capacity_bytes(), 8 * size_of::<f64>());
    assert_eq!(pool.f64_pool.get(&8).map(Vec::len), Some(1));
}

#[test]
fn pooled_uninit_guard_reused_error_discards_exact_capacity() {
    let mut pool = BufferPool::new();
    <f64 as PoolScalar>::pool_release(&mut pool, vec![0.0; 8]);
    let output = PooledUninitOutput::<f64>::new(&mut pool, vec![3]).unwrap();
    let error = unsafe { output.assume_init_as::<tenferro_tensor::Rank<2>>() }.unwrap_err();
    assert!(error.to_string().contains("pooled_uninit_output"));
    assert!(pool.f64_in_flight.is_empty());
    assert_eq!(pool.retained_capacity_bytes(), 0);
}

#[test]
fn pooled_uninit_guard_reused_partial_panic_discards_exact_capacity() {
    let mut pool = BufferPool::new();
    <f64 as PoolScalar>::pool_release(&mut pool, vec![0.0; 8]);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut output = PooledUninitOutput::<f64>::new(&mut pool, vec![3]).unwrap();
        assert!(output.token_is_reused_with_capacity(8));
        output.as_uninit_slice_mut()[0].write(1.0);
        panic!("partial reused C>len panic");
    }));
    assert!(result.is_err());
    assert!(pool.f64_in_flight.is_empty());
    assert_eq!(pool.retained_capacity_bytes(), 0);
    assert!(!pool.f64_pool.contains_key(&8));
}

#[test]
fn internal_full_overwrite_sources_use_the_guard_boundary() {
    let production_sources = [
        ("lib.rs", include_str!("../lib.rs")),
        (
            "pooled_uninit_output.rs",
            include_str!("../pooled_uninit_output.rs"),
        ),
        ("elementwise.rs", include_str!("../elementwise.rs")),
    ];
    for (name, source) in production_sources {
        assert!(
            !source.contains("typed_array_uninit_from_pool"),
            "{name} uses legacy uninit helper"
        );
        assert!(
            !source
                .replace(char::is_whitespace, "")
                .contains("pool_acquire(buffers"),
            "{name} directly acquires pooled output"
        );
        assert!(
            !source.contains("ExecContext::ambient"),
            "{name} selects an ambient context"
        );
        let lines: Vec<_> = source.lines().collect();
        for (index, line) in lines.iter().enumerate() {
            // The call may be split after `unsafe {`; inspect every completion
            // occurrence rather than requiring a single-line expression.
            let part_of_unsafe_call = line.contains("unsafe {")
                || index >= 1 && lines[index - 1].contains("unsafe {")
                || index >= 2 && lines[index - 2].contains("unsafe {");
            if line.contains("assume_init") && part_of_unsafe_call {
                let has_safety = index >= 1 && lines[index - 1].contains("// SAFETY:")
                    || index >= 2 && lines[index - 2].contains("// SAFETY:");
                assert!(
                    has_safety,
                    "{name}:{index} lacks an adjacent assume_init safety proof"
                );
            }
        }
    }
    let pooled_output = include_str!("../pooled_uninit_output.rs");
    assert!(pooled_output.matches("pub unsafe fn assume_init").count() >= 1);

    let cpu_root = include_str!("../../../tenferro-cpu/src/lib.rs");
    assert!(cpu_root.contains("pub(crate) use tenferro_internal_cpu_kernels::PooledUninitOutput;"));
    let cpu_indexing = include_str!("../../../tenferro-cpu/src/indexing.rs");
    assert!(cpu_indexing.contains("use super::PooledUninitOutput;"));

    let pool = include_str!("../buffer_pool.rs");
    assert!(pool.contains("try_reserve_exact(len)"));
    assert!(pool.contains("Error::backend_source(\"pooled_uninit_output\", err)"));
    assert!(!pool.contains("format!(\"unable to reserve"));
    assert!(!pool.contains("std::io::Error::new"));
    let pool_scalar = pool
        .split_once("pub trait PoolScalar")
        .and_then(|(_, suffix)| suffix.split_once("mod private"))
        .expect("PoolScalar and sealed module must remain distinct")
        .0;
    assert!(!pool_scalar.contains("pool_acquire_uninit_tracked"));
    assert!(!pool_scalar.contains("pool_discard_uninit"));
    assert!(!pool.contains("pub enum UninitCheckoutToken"));
    assert!(pool.contains("impl private::Sealed for"));
    let elementwise = include_str!("../elementwise.rs");
    for helper in [
        "pub fn typed_mul_with_pool",
        "pub fn typed_mul_view_with_pool",
    ] {
        let body = elementwise
            .split_once(helper)
            .and_then(|(_, suffix)| suffix.split_once("pub fn typed_"))
            .map(|(body, _)| body)
            .unwrap_or(elementwise);
        assert!(
            body.contains("mul_into_uninit"),
            "{helper} must retain the pinned same-shape kernel"
        );
    }

    assert!(
        !elementwise.contains(
            "// SAFETY: the successful zip/map replay writes every logical destination element and retains no destination view.\\n        // SAFETY:"
        ),
        "generic zip/map overwrite proof must not be duplicated"
    );
    assert!(
        !elementwise.contains(
            "// SAFETY: the successful scalar map replay writes every logical destination element and retains no destination view.\\n        // SAFETY:"
        ),
        "generic scalar-map overwrite proof must not be duplicated"
    );

    let generic_binary = elementwise
        .split_once("fn typed_binary_view_with_pool")
        .and_then(|(_, suffix)| suffix.split_once("fn typed_unary_view_with_pool"))
        .map(|(body, _)| body)
        .expect("generic binary helper must remain present");
    assert!(generic_binary.contains(
        "// SAFETY: the successful runtime-selected zip/map replay writes every logical destination element and retains no destination view."
    ));
    assert!(generic_binary.contains(
        "// SAFETY: the successful runtime-selected scalar-map replay writes every logical destination element and retains no destination view."
    ));

    let add = elementwise
        .split_once("pub fn typed_add_view_with_pool")
        .and_then(|(_, suffix)| suffix.split_once("pub fn typed_sub_with_pool"))
        .map(|(body, _)| body)
        .expect("add view helper must remain present");
    assert!(add.contains(
        "// SAFETY: the successful add zip/map replay writes every logical destination element and retains no destination view."
    ));
    assert!(add.contains(
        "// SAFETY: the successful add scalar-map replay writes every logical destination element and retains no destination view."
    ));
    assert!(
        !add.contains("successful multiplication kernel"),
        "add overwrite proofs must not claim multiplication"
    );

    let multiplication = elementwise
        .split_once("pub fn typed_mul_view_with_pool")
        .and_then(|(_, suffix)| suffix.split_once("fn typed_div_with_pool"))
        .map(|(body, _)| body)
        .expect("multiplication helpers must remain present");
    assert!(
        multiplication.contains("successful multiplication kernel"),
        "same-shape multiplication must retain its operation-specific proof"
    );
}

#[test]
fn pooled_uninit_guard_layout_validation_reports_real_shape_errors() {
    let mut pool = BufferPool::new();
    let error = PooledUninitOutput::<i32>::new(&mut pool, vec![usize::MAX]).unwrap_err();
    assert!(error.to_string().contains("pooled_uninit_output"));
    assert!(pool.is_empty());
}

#[test]
fn pooled_uninit_guard_zero_length_view_bytes_and_drop_are_consistent() {
    let mut pool = BufferPool::new();
    {
        let mut output = PooledUninitOutput::<i32>::new(&mut pool, vec![0]).unwrap();
        assert!(output.token_is_fresh());
        assert!(output.as_uninit_slice_mut().is_empty());
        assert!(output.as_uninit_bytes_mut().is_empty());
        let view = output.as_uninit_view_mut().unwrap();
        assert_eq!(view.dims(), &[0]);
    }
    assert!(pool.is_empty());

    let output = PooledUninitOutput::<i32>::new(&mut pool, vec![0]).unwrap();
    let tensor = unsafe { output.assume_init() }.unwrap();
    assert_eq!(tensor.shape(), &[0]);
    assert!(pool.is_empty());
}

#[test]
fn pooled_uninit_guard_static_rank_handoff_preserves_shape_and_values() {
    let mut pool = BufferPool::new();
    let mut output = PooledUninitOutput::<i32>::new(&mut pool, vec![2, 2]).unwrap();
    output
        .as_uninit_slice_mut()
        .iter_mut()
        .enumerate()
        .for_each(|(index, item)| {
            item.write(index as i32 + 1);
        });

    let tensor = unsafe { output.assume_init_as::<tenferro_tensor::Rank<2>>() }.unwrap();
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.as_slice().unwrap(), &[1, 2, 3, 4]);
    assert!(pool.is_empty());
}

#[test]
fn pooled_uninit_guard_shape_product_failure_does_not_touch_pool() {
    let mut pool = BufferPool::new();
    let error = PooledUninitOutput::<i32>::new(&mut pool, vec![usize::MAX, 2]).unwrap_err();
    assert!(error.to_string().contains("pooled_uninit_output"));
    assert!(pool.is_empty());
}
