use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use tenferro_device::{Error, Result};

pub(crate) fn as_i32(name: &str, value: usize) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| Error::InvalidArgument(format!("{name} is too large for LAPACK i32 ABI")))
}

pub(crate) fn check_len(op: &str, field: &str, got: usize, need: usize) -> Result<()> {
    if got < need {
        return Err(Error::InvalidArgument(format!(
            "{op}: {field} slice length {got} < required {need}"
        )));
    }
    Ok(())
}

pub(crate) fn lwork_from_query_f64(op: &str, query: f64) -> Result<i32> {
    if !query.is_finite() {
        return Err(Error::InvalidArgument(format!(
            "{op}: invalid LAPACK work query result {query}"
        )));
    }
    let lwork = query.max(1.0).ceil();
    if lwork > i32::MAX as f64 {
        return Err(Error::InvalidArgument(format!(
            "{op}: requested work size {lwork} exceeds i32::MAX"
        )));
    }
    Ok(lwork as i32)
}

pub(crate) fn lwork_from_query_f32(op: &str, query: f32) -> Result<i32> {
    if !query.is_finite() {
        return Err(Error::InvalidArgument(format!(
            "{op}: invalid LAPACK work query result {query}"
        )));
    }
    let lwork = query.max(1.0).ceil();
    if lwork > i32::MAX as f32 {
        return Err(Error::InvalidArgument(format!(
            "{op}: requested work size {lwork} exceeds i32::MAX"
        )));
    }
    Ok(lwork as i32)
}

pub(crate) fn lwork_from_query_c64(op: &str, query: Complex64) -> Result<i32> {
    lwork_from_query_f64(op, query.re)
}

pub(crate) fn lwork_from_query_c32(op: &str, query: Complex32) -> Result<i32> {
    lwork_from_query_f32(op, query.re)
}

pub(crate) fn check_info_nonnegative(op: &str, info: i32) -> Result<()> {
    if info < 0 {
        return Err(Error::InvalidArgument(format!(
            "{op}: LAPACK reported invalid argument at position {}",
            -info
        )));
    }
    Ok(())
}

pub(crate) fn check_info_success(op: &str, info: i32) -> Result<()> {
    check_info_nonnegative(op, info)?;
    if info > 0 {
        return Err(Error::InvalidArgument(format!(
            "{op}: LAPACK failed with info={info}"
        )));
    }
    Ok(())
}

pub(crate) fn check_info_cholesky(op: &str, info: i32) -> Result<()> {
    check_info_nonnegative(op, info)?;
    if info > 0 {
        return Err(Error::InvalidArgument(format!(
            "{op}: matrix is not positive definite (minor {info})"
        )));
    }
    Ok(())
}

pub(crate) fn pivots_to_forward_perm(m: usize, pivots: &[i32]) -> Result<Vec<usize>> {
    let mut perm: Vec<usize> = (0..m).collect();
    for (i, &p) in pivots.iter().enumerate() {
        if p <= 0 {
            return Err(Error::InvalidArgument(
                "lu: LAPACK returned non-positive pivot index".into(),
            ));
        }
        let j = (p - 1) as usize;
        if j >= m {
            return Err(Error::InvalidArgument(format!(
                "lu: LAPACK pivot index {p} out of range for m={m}"
            )));
        }
        perm.swap(i, j);
    }
    Ok(perm)
}

pub(crate) fn split_lu<T: Copy + Zero + One>(
    lu: &[T],
    m: usize,
    n: usize,
    l: &mut [T],
    u_out: &mut [T],
) {
    let k = m.min(n);

    for j in 0..k {
        for i in 0..m {
            l[i + j * m] = if i > j {
                lu[i + j * m]
            } else if i == j {
                T::one()
            } else {
                T::zero()
            };
        }
    }

    for j in 0..n {
        for i in 0..k {
            u_out[i + j * k] = if i <= j { lu[i + j * m] } else { T::zero() };
        }
    }
}

pub(crate) fn fill_zero_upper<T: Copy + Zero>(mat: &mut [T], n: usize) {
    for j in 0..n {
        for i in 0..j {
            mat[i + j * n] = T::zero();
        }
    }
}

pub(crate) fn write_real_eig_general_output<T>(
    n: usize,
    wr: &[T],
    wi: &[T],
    vr: &[T],
    values_ri: &mut [T],
    vectors_ri: &mut [T],
) where
    T: Copy + Zero + PartialOrd + std::ops::Neg<Output = T>,
{
    for i in 0..n {
        values_ri[2 * i] = wr[i];
        values_ri[2 * i + 1] = wi[i];
    }

    for v in vectors_ri.iter_mut() {
        *v = T::zero();
    }

    let mut j = 0usize;
    while j < n {
        if wi[j] == T::zero() {
            for i in 0..n {
                vectors_ri[2 * (i + j * n)] = vr[i + j * n];
                vectors_ri[2 * (i + j * n) + 1] = T::zero();
            }
            j += 1;
        } else if wi[j] > T::zero() {
            for i in 0..n {
                let re = vr[i + j * n];
                let im = vr[i + (j + 1) * n];
                vectors_ri[2 * (i + j * n)] = re;
                vectors_ri[2 * (i + j * n) + 1] = im;
                vectors_ri[2 * (i + (j + 1) * n)] = re;
                vectors_ri[2 * (i + (j + 1) * n) + 1] = -im;
            }
            j += 2;
        } else {
            j += 1;
        }
    }
}
