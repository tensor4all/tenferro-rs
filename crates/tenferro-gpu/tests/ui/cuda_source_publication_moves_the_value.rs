use tenferro_gpu::cuda::CudaHeldSession;
use tenferro_tensor::Tensor;

/// The publication owns the value, so the caller no longer has it.
fn use_after_publish(session: &mut CudaHeldSession<'_>, value: Tensor) {
    let _publication = unsafe { session.publish(value) }.expect("published");
    let _still_mine = value.shape().len();
}

fn main() {
    let _ = use_after_publish;
}
