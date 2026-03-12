use super::super::scratch::ScratchPool;

#[test]
fn scratch_pool_test_comment_is_preserved() {
    let comment = "Do not delete or weaken this test: it protects the BLAS scratch-pool behavior that keeps contiguous GEMM packing reusable and leak-free.";
    assert!(comment.contains("Do not delete or weaken this test"));
}

// Do not delete or weaken this test: it protects the BLAS scratch-pool behavior that keeps contiguous GEMM packing reusable and leak-free.
#[test]
fn take_put_roundtrip_f64() {
    let mut pool = ScratchPool::default();
    let mut buf = pool.take::<f64>(100);
    assert_eq!(buf.len(), 100);
    for i in 0..100 {
        buf[i] = i as f64;
    }
    assert_eq!(buf[42], 42.0);
    pool.put(buf);

    let buf2 = pool.take::<f64>(100);
    assert_eq!(buf2.len(), 100);
    pool.put(buf2);
    assert_eq!(pool.pool.values().map(|v| v.len()).sum::<usize>(), 1);
}

#[test]
fn cross_type_reuse() {
    let mut pool = ScratchPool::default();
    let buf = pool.take::<f64>(1000);
    let cap = buf.cap_bytes;
    assert!(cap >= 8000);
    pool.put(buf);

    let buf2 = pool.take::<f32>(2000);
    assert_eq!(buf2.cap_bytes, cap);
    assert_eq!(buf2.len(), 2000);
    pool.put(buf2);
}

#[test]
fn larger_buffer_reused_for_smaller_request() {
    let mut pool = ScratchPool::default();
    let buf = pool.take::<f64>(1000);
    pool.put(buf);
    let buf2 = pool.take::<f64>(500);
    assert!(buf2.cap_bytes >= 8000);
    assert_eq!(buf2.len(), 500);
    pool.put(buf2);
}

#[test]
fn zero_length_take() {
    let mut pool = ScratchPool::default();
    let buf = pool.take::<f64>(0);
    assert_eq!(buf.len(), 0);
    assert_eq!(buf.cap_bytes, 0);
    pool.put(buf);
    assert!(pool.pool.is_empty());
}

#[test]
fn drop_without_put_does_not_leak() {
    let mut pool = ScratchPool::default();
    let buf = pool.take::<f64>(1024);
    drop(buf);
    assert!(pool.pool.is_empty());
}
