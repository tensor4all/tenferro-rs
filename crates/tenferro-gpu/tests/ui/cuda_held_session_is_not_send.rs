use tenferro_gpu::cuda::CudaHeldSession;

fn assert_send<T: Send>() {}

fn main() {
    // A held session binds a stream and its admission to the opening thread.
    assert_send::<CudaHeldSession<'static>>();
}
