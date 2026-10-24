//! The independent-use boundary of #1903: a user-owned host operation over the
//! unified tensor/view family.
//!
//! A downstream crate that stores its own scalar in `TypedTensor` owns the
//! numerical kernel. This module is that case reduced to its minimum: the
//! operation below reaches the tensor data through `ColMajorView` /
//! `ColMajorViewMut` only, and needs no tenferro session, no extension
//! registration, and no numerical or AD backend. The kernel is generic over the
//! element type, so one source line serves a preset scalar and a downstream one.

use tenferro_tensor::{ColMajorView, ColMajorViewMut, Rank, TypedTensor};

/// A downstream element type, deliberately not one of tenferro's preset scalars.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Weight(i32);

impl std::ops::Add for Weight {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl std::ops::Mul for Weight {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

/// `c = a * b` in column-major storage, against the validated views alone.
fn matmul_into<T>(
    a: &ColMajorView<'_, T, 2>,
    b: &ColMajorView<'_, T, 2>,
    c: &mut ColMajorViewMut<'_, T, 2>,
) where
    T: Copy + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    let [m, k] = *a.shape();
    let [k_other, n] = *b.shape();
    assert_eq!(k, k_other, "inner extents must agree");
    assert_eq!(c.shape(), &[m, n], "destination holds m x n");
    for j in 0..n {
        for i in 0..m {
            let mut acc = T::default();
            for p in 0..k {
                acc = acc + a[[i, p]] * b[[p, j]];
            }
            c[[i, j]] = acc;
        }
    }
}

#[test]
fn preset_scalar_runs_the_downstream_kernel() {
    let a = TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 2], vec![1., 2., 3., 4.]).unwrap();
    let b = TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 2], vec![5., 6., 7., 8.]).unwrap();
    let mut c = TypedTensor::<f64, Rank<2>>::zeros([2, 2]).unwrap();
    matmul_into(
        &a.host_col_major_view().unwrap(),
        &b.host_col_major_view().unwrap(),
        &mut c.host_col_major_view_mut().unwrap(),
    );
    assert_eq!(c.as_slice().unwrap(), &[23., 34., 31., 46.]);
}

#[test]
fn downstream_scalar_runs_the_same_kernel_unchanged() {
    let a = TypedTensor::<Weight, Rank<2>>::from_vec_col_major(
        [2, 2],
        vec![Weight(2), Weight(0), Weight(1), Weight(3)],
    )
    .unwrap();
    let b = TypedTensor::<Weight, Rank<2>>::from_vec_col_major(
        [2, 2],
        vec![Weight(1), Weight(4), Weight(2), Weight(5)],
    )
    .unwrap();
    let mut c =
        TypedTensor::<Weight, Rank<2>>::from_vec_col_major([2, 2], vec![Weight::default(); 4])
            .unwrap();
    matmul_into(
        &a.host_col_major_view().unwrap(),
        &b.host_col_major_view().unwrap(),
        &mut c.host_col_major_view_mut().unwrap(),
    );
    assert_eq!(
        c.as_slice().unwrap(),
        &[Weight(6), Weight(12), Weight(9), Weight(15)]
    );
}

#[test]
fn downstream_scalar_keeps_checked_access_without_a_backend() {
    let mut tensor = TypedTensor::<Weight, Rank<2>>::from_vec_col_major(
        [2, 2],
        vec![Weight(1), Weight(2), Weight(3), Weight(4)],
    )
    .unwrap();
    assert_eq!(tensor[[1, 0]], Weight(2));
    tensor[[1, 0]] = Weight(20);
    assert_eq!(tensor.get(&[1, 0]).ok(), Some(&Weight(20)));
    assert!(tensor.get(&[2, 0]).is_err());
}

#[test]
fn host_marked_owner_reaches_the_same_views_for_a_downstream_scalar() {
    let tensor = TypedTensor::<Weight, Rank<2>>::from_vec_col_major(
        [2, 2],
        vec![Weight(1), Weight(2), Weight(3), Weight(4)],
    )
    .unwrap();
    let mut host = tensor.into_host().unwrap();
    assert_eq!(host.host_col_major_view().unwrap()[[1, 0]], Weight(2));
    host.host_col_major_view_mut().unwrap()[[1, 0]] = Weight(20);
    assert_eq!(host.host_col_major_view().unwrap()[[1, 0]], Weight(20));
}
