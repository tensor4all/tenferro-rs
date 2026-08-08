use super::common::*;
use super::support;
use tenferro_fft::FftNorm;
use tenferro_gpu::cuda::gpu_available;
use tenferro_runtime::Tensor;

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_validation_rejects_host_foreign_integer_bool_and_invalid_lengths() {
    if !gpu_available() {
        return;
    }

    let mut cuda = support::cuda_backend();
    let host = real_f64(&[4], 0.5);
    assert_error(
        Operation::Fft.execute_cuda(&mut cuda, &host, None, -1, FftNorm::Backward),
        "device",
    );

    let other = support::cuda_backend();
    let gpu_input = support::upload_cuda(other.runtime(), &host);
    assert_error(
        Operation::Fft.execute_cuda(&mut cuda, &gpu_input, None, -1, FftNorm::Backward),
        "runtime",
    );

    for input in [
        Tensor::from_vec_col_major(vec![4], vec![1_i32, 2, 3, 4]).unwrap(),
        Tensor::from_vec_col_major(vec![4], vec![true, false, true, false]).unwrap(),
    ] {
        assert_error(
            Operation::Fft.execute_cuda(&mut cuda, &input, None, -1, FftNorm::Backward),
            "dtype",
        );
    }

    assert_error(
        Operation::Fft.execute_cuda(&mut cuda, &host, None, 4, FftNorm::Backward),
        "axis",
    );
    assert_error(
        Operation::Fft.execute_cuda(&mut cuda, &host, Some(0), -1, FftNorm::Backward),
        "n",
    );

    let spectrum = complex_f64(&[3], 0.5);
    assert_error(
        Operation::Irfft.execute_cuda(&mut cuda, &spectrum, Some(8), -1, FftNorm::Backward),
        "spectrum",
    );
}
