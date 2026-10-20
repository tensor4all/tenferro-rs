use tenferro_gpu::cuda::CudaEngine;
use tenferro_tensor::Tensor;

/// The publication owns the value, so the caller no longer has it.
fn use_after_publish(engine: &mut CudaEngine, value: Tensor) {
    let _publication = engine.publish(value).expect("published");
    let _still_mine = value.shape().len();
}

fn main() {
    let _ = use_after_publish;
}
