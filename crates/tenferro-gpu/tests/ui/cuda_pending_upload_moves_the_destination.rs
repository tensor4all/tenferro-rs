use tenferro_gpu::cuda::{upload_pending, CudaRuntime, PinnedHostBuffer};
use tenferro_tensor::Tensor;

/// The transfer takes ownership of the destination, so the caller no longer has it: nothing can read,
/// mutate or drop the device tensor while the copy may write it, and forgetting the handle cannot
/// restore access to the admitted package.
fn use_after_move(runtime: &CudaRuntime, source: PinnedHostBuffer, destination: Tensor) {
    let _pending = upload_pending(runtime, source, destination).expect("submitted");
    let _still_mine = destination.shape().len();
}

fn main() {
    let _ = use_after_move;
}
