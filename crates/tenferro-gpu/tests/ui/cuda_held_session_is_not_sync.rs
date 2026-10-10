use tenferro_gpu::cuda::CudaHeldSession;

fn assert_sync<T: Sync>() {}

fn main() {
    // A held session must not be shared across threads: its binding is thread-affine.
    assert_sync::<CudaHeldSession<'static>>();
}
