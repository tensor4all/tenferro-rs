use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_fft::prelude::*;

#[test]
fn prelude_calls_concrete_fft_operation() {
    let input = Tensor::from_vec_col_major([4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let mut backend = CpuEngine::serial();
    let spectrum = {
        let session: &mut dyn tenferro_tensor::Engine = &mut backend;
        input.fft(None, -1, FftNorm::Backward, session)
    }
    .unwrap();
    assert_eq!(
        spectrum.as_slice::<Complex64>().unwrap()[0],
        Complex64::new(10.0, 0.0)
    );
}

#[cfg(feature = "autodiff")]
#[test]
fn prelude_calls_eager_fft_operation() {
    let mut engine = CpuEngine::serial();
    let runtime = EagerRuntime::new().unwrap();
    let input = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major([4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
        runtime.clone(),
    )
    .unwrap();
    let spectrum = runtime
        .with_eager_session(&mut engine, |session| {
            session.rfft(&input, None, -1, FftNorm::Backward)
        })
        .unwrap();
    assert_eq!(spectrum.shape(), &[3]);
}
