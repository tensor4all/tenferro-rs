use tenferro::cuda::{download_tensor, upload_tensor, CudaBackend};
use tenferro::{Tensor, TensorBackend};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = CudaBackend::new(0)?;

    let a = Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]);
    let b = Tensor::from_vec(vec![3], vec![4.0_f64, 5.0, 6.0]);

    let gpu_a = upload_tensor(backend.runtime(), &a)?;
    let gpu_b = upload_tensor(backend.runtime(), &b)?;
    let gpu_c = backend.add(&gpu_a, &gpu_b)?;
    let c = download_tensor(backend.runtime(), &gpu_c)?;

    assert_eq!(c.shape(), &[3]);
    assert_eq!(c.as_slice::<f64>().unwrap(), &[5.0, 7.0, 9.0]);

    Ok(())
}
