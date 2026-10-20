// Import the parent's items by name rather than with a glob. The glob would
// also pull in `cubecl::prelude::ComplexCore`, whose default `conj(self)`
// body is `unexpanded!()`; because it takes the receiver by value it wins
// method resolution over `num_complex`'s inherent `conj(&self)`, and the
// host-side checks below then panic with "Unexpanded Cube functions should
// not be called" instead of computing a conjugate.
use super::super::{svd_typed, SvdDriver};
use num_complex::Complex64;
use tenferro_gpu::cuda::{download_tensor, upload_tensor, CudaDeviceId, CudaEngine};
use tenferro_tensor::Tensor;

#[test]
#[ignore = "requires CUDA; full factors have no public driver option"]
fn test_cubecl_svd_xgesvdp_full_and_economy() {
    let mut backend = CudaEngine::open(CudaDeviceId::from_ordinal(0)).unwrap();
    for (m, n) in [(7, 3), (3, 7)] {
        let data: Vec<_> = (0..m * n)
            .map(|i| Complex64::new((i as f64 * 0.7).cos(), (i as f64 * 0.31 + 0.4).sin()))
            .collect();
        let host = Tensor::from_vec_col_major([m, n], data.clone()).unwrap();
        let device = upload_tensor(backend.runtime(), &host).unwrap();
        for full in [false, true] {
            let (u, s, vt) = svd_typed(
                &mut backend,
                device.as_typed::<Complex64>().unwrap(),
                full,
                SvdDriver::Xgesvdp,
            )
            .unwrap();
            let (uc, vr, k) = (
                if full { m } else { m.min(n) },
                if full { n } else { m.min(n) },
                m.min(n),
            );
            assert_eq!(u.shape(), &[m, uc]);
            assert_eq!(vt.shape(), &[vr, n]);
            let u = download_tensor(backend.runtime(), &Tensor::from_typed(u)).unwrap();
            let s = download_tensor(backend.runtime(), &Tensor::from_typed(s)).unwrap();
            let vt = download_tensor(backend.runtime(), &Tensor::from_typed(vt)).unwrap();
            let u = u.as_slice::<Complex64>().unwrap();
            let s = s.as_slice::<f64>().unwrap();
            let vt = vt.as_slice::<Complex64>().unwrap();
            for i in 0..m {
                for j in 0..n {
                    let actual: Complex64 =
                        (0..k).map(|c| u[i + m * c] * s[c] * vt[c + vr * j]).sum();
                    assert!(
                        (actual - data[i + m * j]).norm() < 1e-12,
                        "reconstruction full={full} ({m},{n})"
                    );
                }
            }
            // Include the nullspace columns/rows, not only the thin factors.
            for i in 0..uc {
                for j in 0..uc {
                    let gram: Complex64 = (0..m).map(|r| u[r + m * i].conj() * u[r + m * j]).sum();
                    assert!(
                        (gram - Complex64::new(f64::from(i == j), 0.0)).norm() < 1e-12,
                        "U full={full}"
                    );
                }
            }
            for i in 0..vr {
                for j in 0..vr {
                    let gram: Complex64 =
                        (0..n).map(|c| vt[i + vr * c] * vt[j + vr * c].conj()).sum();
                    assert!(
                        (gram - Complex64::new(f64::from(i == j), 0.0)).norm() < 1e-12,
                        "V full={full}"
                    );
                }
            }
        }
    }
}
