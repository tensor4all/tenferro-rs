use tenferro_gpu::{
    cuda::cuda_devices, cuda::download_tensor, cuda::upload_tensor, cuda::CudaBackend,
};
use tenferro_tensor::{Tensor, TensorElementwise, TensorRead, TensorStructural, TensorWrite};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(device) = cuda_devices()?.into_iter().next() else {
        return Ok(());
    };

    let mut backend = CudaBackend::new(device.id())?;
    let cpu_a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let cpu_b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();

    let gpu_a = upload_tensor(backend.runtime(), &cpu_a)?;
    let gpu_b = upload_tensor(backend.runtime(), &cpu_b)?;
    let gpu_c = backend.add(&gpu_a, &gpu_b)?;
    let cpu_c = download_tensor(backend.runtime(), &gpu_c)?;

    assert_eq!(cpu_c.as_slice::<f64>().unwrap(), &[4.0, 6.0]);

    let mut gpu_reuse = upload_tensor(backend.runtime(), &cpu_c)?;
    backend.copy_read_into(
        TensorRead::from_tensor(&gpu_c),
        TensorWrite::from_tensor(&mut gpu_reuse),
    )?;
    let copied = download_tensor(backend.runtime(), &gpu_reuse)?;
    assert_eq!(copied.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
    Ok(())
}
