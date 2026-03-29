use super::*;
use tenferro_prims::CpuContext;

#[test]
fn take_returns_correct_length() {
    let mut pool = BufferPool::<f64>::new();
    let buf = pool.take(100);
    assert_eq!(buf.len(), 100);
    assert!(buf.capacity() >= 100);
    assert!(buf.iter().all(|&x| x == 0.0));
}

#[test]
fn return_and_reuse() {
    let mut pool = BufferPool::<f64>::new();
    let mut buf = pool.take(100);
    buf.fill(7.0);
    let ptr = buf.as_ptr();
    pool.return_buf(buf);
    let buf2 = pool.take(50);
    assert_eq!(buf2.as_ptr(), ptr);
    assert_eq!(buf2.len(), 50);
    assert!(buf2.iter().all(|&x| x == 0.0));
}

#[test]
fn best_fit_selection() {
    let mut pool = BufferPool::<f64>::new();
    let small = Vec::<f64>::with_capacity(50);
    let large = Vec::<f64>::with_capacity(200);
    pool.return_buf(small);
    pool.return_buf(large);
    let buf = pool.take(60);
    assert!(buf.capacity() >= 60);
}

#[test]
fn context_pool_reuses_cpu_context_storage() {
    let mut ctx = CpuContext::new(1);
    let mut pool = BufferPool::<f64>::new();
    let buf: Vec<f64> = pool.take_with_ctx(&mut ctx, 64);
    let capacity = buf.capacity();
    pool.return_buf(buf);
    pool.flush_to_context(&mut ctx);

    let mut pool = BufferPool::<f64>::new();
    let reused: Vec<f64> = pool.take_with_ctx(&mut ctx, 8);
    assert_eq!(reused.capacity(), capacity);
}

#[test]
fn with_context_buffer_pool_flushes_local_buffers_back_to_context() {
    let mut ctx = CpuContext::new(1);

    let capacity = with_context_buffer_pool::<f64, _, _>(&mut ctx, |ctx, pool| {
        let buf: Vec<f64> = pool.take_with_ctx(ctx, 64);
        let capacity = buf.capacity();
        pool.return_buf(buf);
        capacity
    });

    let mut pool = BufferPool::<f64>::new();
    let reused: Vec<f64> = pool.take_with_ctx(&mut ctx, 8);
    assert_eq!(reused.capacity(), capacity);
}
