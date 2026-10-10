use tenferro_gpu::cuda::{download_pending, CudaRuntime, PinnedHostBuffer};
use tenferro_tensor::Tensor;

/// The transfer takes ownership of the value, so the caller no longer has it: no read lease has to
/// be maintained by hand, and forgetting the handle cannot restore access to the admitted package.
fn use_after_move(runtime: &CudaRuntime, source: Tensor) {
    let bytes = source.shape().iter().product::<usize>() * 8;
    let buffer = PinnedHostBuffer::new(runtime, bytes).expect("pinned buffer");
    let _pending = download_pending(runtime, source, buffer).expect("submitted");
    let _still_mine = source.shape().len();
}

fn main() {
    let _ = use_after_move;
}
