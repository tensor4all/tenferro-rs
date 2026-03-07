use super::*;

#[test]
fn take_returns_correct_length() {
    let mut pool = BufferPool::<f64>::new();
    let buf = pool.take(100);
    assert_eq!(buf.len(), 100);
    assert!(buf.capacity() >= 100);
}

#[test]
fn return_and_reuse() {
    let mut pool = BufferPool::<f64>::new();
    let buf = pool.take(100);
    let ptr = buf.as_ptr();
    pool.return_buf(buf);
    let buf2 = pool.take(50);
    assert_eq!(buf2.as_ptr(), ptr);
    assert_eq!(buf2.len(), 50);
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
