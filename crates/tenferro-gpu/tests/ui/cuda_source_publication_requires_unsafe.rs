use tenferro_gpu::cuda::CudaHeldSession;
use tenferro_tensor::Tensor;

/// Publishing asserts the caller's provenance claim, so it must not be callable from safe code.
fn publish_without_unsafe(session: &mut CudaHeldSession<'_>, value: Tensor) {
    let _publication = session.publish(value);
}

fn main() {
    let _ = publish_without_unsafe;
}
