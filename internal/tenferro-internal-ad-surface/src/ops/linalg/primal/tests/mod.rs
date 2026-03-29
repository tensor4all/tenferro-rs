use super::*;
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{set_default_runtime, RuntimeContext};

fn matrix_f64(values: &[f64], dims: &[usize]) -> Tensor<f64> {
    Tensor::from_slice(values, dims, MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn factorization_builders_cover_qr_and_lu_variants() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = matrix_f64(&[4.0, 1.0, 1.0, 3.0], &[2, 2]);

    let qr_out = qr(&a).run().unwrap();
    assert_eq!(qr_out.q.dims(), &[2, 2]);
    assert_eq!(qr_out.r.dims(), &[2, 2]);

    let lu_out = lu(&a).pivot(LuPivot::NoPivot).run().unwrap();
    assert_eq!(lu_out.p.dims(), &[0]);
    assert_eq!(lu_out.l.dims(), &[2, 2]);
    assert_eq!(lu_out.u.dims(), &[2, 2]);

    let lu_factor_out = lu_factor(&a).run().unwrap();
    assert_eq!(lu_factor_out.factors.dims(), &[2, 2]);
    assert_eq!(lu_factor_out.pivots.dims(), &[2]);

    let lu_factor_ex_out = lu_factor_ex(&a).run().unwrap();
    assert_eq!(lu_factor_ex_out.factors.dims(), &[2, 2]);
    assert_eq!(lu_factor_ex_out.pivots.dims(), &[2]);
    assert_eq!(lu_factor_ex_out.info.buffer().as_slice().unwrap(), &[0]);
}

#[test]
fn factorization_builders_cover_eigen_and_cholesky_variants() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let spd = matrix_f64(&[5.0, 1.0, 1.0, 4.0], &[2, 2]);

    let eigen_out = eigen(&spd).run().unwrap();
    assert_eq!(eigen_out.values.dims(), &[2]);
    assert_eq!(eigen_out.vectors.dims(), &[2, 2]);

    let cholesky_out = cholesky(&spd).run().unwrap();
    assert_eq!(cholesky_out.dims(), &[2, 2]);

    let cholesky_ex_out = cholesky_ex(&spd).run().unwrap();
    assert_eq!(cholesky_ex_out.l.dims(), &[2, 2]);
    assert_eq!(cholesky_ex_out.info.buffer().as_slice().unwrap(), &[0]);
}
