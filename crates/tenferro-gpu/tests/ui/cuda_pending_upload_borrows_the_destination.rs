use tenferro_gpu::cuda::{upload_pending, CudaRuntime, PinnedHostBuffer};
use tenferro_tensor::Tensor;

/// The pending handle owns the destination's mutable borrow for the whole copy, so no second
/// borrow of the same tensor may exist while it is in flight.
fn borrow_twice(runtime: &CudaRuntime, device: &mut Tensor) {
    let source = PinnedHostBuffer::new(runtime, 8).expect("pinned buffer");
    let pending = upload_pending(runtime, source, device).expect("pending upload");
    let _second = &mut *device;
    drop(pending);
}

fn main() {
    let _ = borrow_twice;
}
