use tenferro_algebra::Standard;
use tenferro_device::Result;
use tenferro_tensor::Tensor;

use super::*;

#[test]
fn cuda_backend_feature_surface_matches_tensor_prims_contract() {
    let _plan_fn: fn(&mut CudaContext, &PrimDescriptor, &[&[usize]]) -> Result<CudaPlan<f64>> =
        <CudaBackend as TensorPrims<Standard<f64>>>::plan;
    let _execute_fn: fn(
        &mut CudaContext,
        &CudaPlan<f64>,
        f64,
        &[&Tensor<f64>],
        f64,
        &mut Tensor<f64>,
    ) -> Result<()> = <CudaBackend as TensorPrims<Standard<f64>>>::execute;

    assert!(<CudaBackend as TensorPrims<Standard<f64>>>::has_extension_for(Extension::Contract));
    assert!(
        <CudaBackend as TensorPrims<Standard<f64>>>::has_extension_for(Extension::ElementwiseMul)
    );
}
