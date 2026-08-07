use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
use tenferro_linalg::prelude::*;
use tenferro_runtime::prelude::*;
use tenferro_runtime::{TensorRead, TensorView};

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error < 1.0e-12,
            "value {index}: actual={actual}, expected={expected}, error={error}"
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = CpuBackend::new();

    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![3.0, 0.0, 0.0, 1.0])?;
    let identity = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0])?;

    let product = a.matmul(&identity, &mut backend)?;
    assert_eq!(product.shape(), &[2, 2]);
    assert_close(product.host_data()?, &[3.0, 0.0, 0.0, 1.0]);

    let svd = backend.with_backend_session(|session| {
        with_cpu_exec_session(session, |exec_session| {
            exec_session.svd_read(TensorRead::from_view(TensorView::F64(product.as_view())))
        })
        .expect("CpuBackend must expose a CPU execution session")
    })?;
    assert_eq!(svd.len(), 3);
    assert_eq!(svd[0].shape(), &[2, 2]);
    assert_eq!(svd[1].shape(), &[2]);
    assert_eq!(svd[2].shape(), &[2, 2]);
    assert_close(svd[1].as_slice::<f64>().unwrap(), &[3.0, 1.0]);

    Ok(())
}
