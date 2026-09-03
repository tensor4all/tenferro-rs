#![cfg(all(feature = "autodiff", feature = "cuda"))]

use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_einsum::EagerEinsumExt;
use tenferro_gpu::cuda::{
    download_tensor, gpu_available, upload_tensor, CudaBackend, CudaDeviceId,
};
use tenferro_runtime::Tensor;

#[test]
#[ignore]
fn cuda_eager_three_operand_einsum_stays_resident() {
    if !gpu_available() {
        eprintln!("skipping CUDA eager einsum test: no CUDA device");
        return;
    }

    let backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let runtime = EagerRuntime::with_cuda_backend(backend.clone()).unwrap();
    let make_input = |shape: Vec<usize>, values: Vec<f64>| {
        let host = Tensor::from_vec_col_major(shape, values).unwrap();
        let device = upload_tensor(backend.runtime(), &host).unwrap();
        EagerTensor::from_tensor_in(device, runtime.clone()).unwrap()
    };
    let lhs = make_input(vec![2, 2, 2], vec![1.0; 8]);
    let middle = make_input(vec![2, 2, 2], vec![1.0; 8]);
    let rhs = make_input(vec![2, 2, 2], vec![1.0; 8]);

    // Three operands force the general extension path rather than the direct
    // binary-dot path.
    let output = [&lhs, &middle, &rhs].einsum("bij,bjk,bkl->bil").unwrap();

    assert_eq!(output.runtime().id(), runtime.id());
    assert_eq!(output.shape(), &[2, 2, 2]);
    let resident = output.to_tensor().unwrap();
    assert!(resident.as_slice::<f64>().is_err());
    let host = download_tensor(backend.runtime(), &resident).unwrap();
    assert_eq!(host.as_slice::<f64>().unwrap(), &[4.0; 8]);
}
