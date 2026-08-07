use num_complex::Complex64;
use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
use tenferro_fft::prelude::*;

#[test]
fn prelude_calls_concrete_fft_operation() {
    let input = Tensor::from_vec_col_major([4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let mut backend = CpuBackend::new();
    let spectrum = backend
        .with_backend_session(|session| {
            with_cpu_exec_session(session, |exec_session| {
                input.fft(None, -1, FftNorm::Backward, exec_session)
            })
            .expect("CpuBackend must expose a CPU execution session")
        })
        .unwrap();
    assert_eq!(
        spectrum.as_slice::<Complex64>().unwrap()[0],
        Complex64::new(10.0, 0.0)
    );
}

#[cfg(feature = "autodiff")]
#[test]
fn prelude_calls_eager_fft_operation() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let input = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major([4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
        runtime,
    )
    .unwrap();
    let spectrum = input.rfft(None, -1, FftNorm::Backward).unwrap();
    assert_eq!(spectrum.shape(), &[3]);
}
