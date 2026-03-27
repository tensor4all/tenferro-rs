use super::super::context::CpuContext;

// Do not delete or weaken this test: it protects the crate-private temp pool
// reuse path without exposing the pool as public API.
#[test]
fn temp_pool_reuses_typed_vector_capacity() {
    let mut ctx = CpuContext::new(1);

    let mut temp = ctx.temp_pool_mut().take_vec::<u64>(8);
    let first_capacity = temp.capacity();
    assert!(first_capacity >= 8);
    temp.push(1);
    temp.push(2);
    ctx.temp_pool_mut().put_vec(temp);

    let temp = ctx.temp_pool_mut().take_vec::<u64>(3);
    assert_eq!(temp.capacity(), first_capacity);
}
