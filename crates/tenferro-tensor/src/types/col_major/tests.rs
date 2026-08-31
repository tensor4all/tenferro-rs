use super::*;
use crate::{
    BackendStorageHandle, DeviceId, DeviceKind, GpuBackendKind, MemoryKind, Placement,
    StorageBuffer,
};

#[test]
fn owned_view_preserves_static_shape_and_column_major_access() {
    let tensor =
        TypedTensor::<i32, Rank<3>>::from_vec_col_major([2, 2, 2], (0..8).collect()).unwrap();
    let view = tensor.host_col_major().unwrap();

    assert_eq!(view.shape(), &[2, 2, 2]);
    assert_eq!(view[[1, 0, 0]], 1);
    assert_eq!(view[[0, 1, 0]], 2);
    assert_eq!(view[[0, 0, 1]], 4);
    assert_eq!(view.get([2, 0, 0]), None);
    assert_eq!(view.iter().copied().sum::<i32>(), 28);
    assert_eq!(
        view.axis0_lanes().collect::<Vec<_>>(),
        vec![&[0, 1][..], &[2, 3][..], &[4, 5][..], &[6, 7][..]]
    );

    // SAFETY: every index is below the corresponding extent, 2.
    assert_eq!(unsafe { view.get_unchecked([1, 1, 1]) }, &7);
}

#[test]
fn mutable_view_iterators_keep_element_borrows_disjoint() {
    let mut tensor =
        TypedTensor::<i32, Rank<2>>::from_vec_col_major([2, 2], vec![1, 2, 3, 4]).unwrap();
    {
        let mut view = tensor.host_col_major_mut().unwrap();
        *view.get_mut([1, 0]).unwrap() = 20;
        view[[0, 1]] = 30;
        for lane in view.axis0_lanes_mut() {
            lane[1] += 1;
        }
        for value in view.iter_mut() {
            *value *= 2;
        }
        assert_eq!(view.as_slice(), &[2, 42, 60, 10]);
    }
    assert_eq!(tensor.as_slice().unwrap(), &[2, 42, 60, 10]);
}

#[test]
fn scalar_empty_and_singleton_shapes_have_defined_lane_behavior() {
    let scalar = TypedTensor::<i32, Rank<0>>::from_vec_col_major([], vec![7]).unwrap();
    let scalar_view = scalar.host_col_major().unwrap();
    assert_eq!(scalar_view[[]], 7);
    assert_eq!(
        scalar_view.axis0_lanes().collect::<Vec<_>>(),
        vec![&[7][..]]
    );

    let empty = TypedTensor::<i32, Rank<2>>::from_vec_col_major([0, 3], vec![]).unwrap();
    assert_eq!(empty.host_col_major().unwrap().axis0_lanes().count(), 0);

    let empty_data: [i32; 0] = [];
    let late_empty = TypedTensorView::<_, Rank<3>>::from_slice_ranked(
        [usize::MAX, 2, 0],
        [1, 1, 1],
        0,
        &empty_data,
    )
    .unwrap();
    assert!(late_empty.host_col_major().unwrap().as_slice().is_empty());

    let singleton = TypedTensor::<i32, Rank<3>>::from_vec_col_major([1, 1, 2], vec![8, 9]).unwrap();
    assert_eq!(
        singleton
            .host_col_major()
            .unwrap()
            .axis0_lanes()
            .collect::<Vec<_>>(),
        vec![&[8][..], &[9][..]]
    );
}

#[test]
fn compact_offset_view_is_zero_copy_and_noncompact_view_is_rejected() {
    let data = [99_i32, 1, 2, 3, 4, 88];
    let validated = {
        let offset =
            TypedTensorView::<_, Rank<2>>::from_slice_ranked([2, 2], [1, 2], 1, &data).unwrap();
        offset.host_col_major().unwrap()
    };
    assert_eq!(validated.as_slice(), &[1, 2, 3, 4]);
    assert!(std::ptr::eq(
        validated.as_slice().as_ptr(),
        data[1..].as_ptr()
    ));

    let noncompact =
        TypedTensorView::<_, Rank<2>>::from_slice_ranked([2, 2], [2, 1], 0, &data[..4]).unwrap();
    assert!(matches!(
        noncompact.host_col_major(),
        Err(crate::Error::Validation { .. })
    ));
}

#[test]
fn constructor_rejects_overflow_and_length_mismatch() {
    assert!(matches!(
        ColMajorView::<u8, 2>::new(&[], [usize::MAX, 2], "test"),
        Err(crate::Error::Validation { .. })
    ));
    assert!(matches!(
        ColMajorView::<u8, 1>::new(&[0], [2], "test"),
        Err(crate::Error::Validation { .. })
    ));
    assert!(matches!(
        static_shape::<2>(&[1], "test"),
        Err(crate::Error::Validation { .. })
    ));
}

#[test]
fn mutable_view_access_surface_covers_checked_and_unsafe_paths() {
    let mut tensor =
        TypedTensor::<i32, Rank<2>>::from_vec_col_major([2, 2], vec![1, 2, 3, 4]).unwrap();
    {
        let mut view = tensor.host_col_major_mut().unwrap();

        assert_eq!(view.shape(), &[2, 2]);
        assert_eq!(view.iter().copied().sum::<i32>(), 10);
        assert_eq!(view.axis0_lanes().count(), 2);
        assert_eq!(view.get([0, 1]), Some(&3));
        assert_eq!(view.get([2, 0]), None);
        assert_eq!(view.get_mut([0, 2]), None);
        view.as_mut_slice()[0] = 10;
        // SAFETY: both indices are below extent 2 and no overlapping mutable
        // borrow remains active across these statements.
        assert_eq!(unsafe { view.get_unchecked([1, 0]) }, &2);
        // SAFETY: index [1, 1] is in bounds and this is the only active borrow.
        *unsafe { view.get_unchecked_mut([1, 1]) } = 40;
        assert_eq!(
            format!("{view:?}"),
            "ColMajorViewMut { shape: [2, 2], len: 4 }"
        );
        assert_eq!(view.as_slice(), &[10, 2, 3, 40]);
    }

    let shared_view = tensor.host_col_major().unwrap();
    assert_eq!(
        format!("{shared_view:?}"),
        "ColMajorView { shape: [2, 2], len: 4 }"
    );

    let shared = std::panic::catch_unwind(|| {
        let tensor = TypedTensor::<i32, Rank<1>>::from_vec_col_major([1], vec![1]).unwrap();
        let view = tensor.host_col_major().unwrap();
        let _ = view[[1]];
    });
    assert!(shared.is_err());

    let mutable = std::panic::catch_unwind(|| {
        let mut tensor = TypedTensor::<i32, Rank<1>>::from_vec_col_major([1], vec![1]).unwrap();
        let mut view = tensor.host_col_major_mut().unwrap();
        view[[1]] = 2;
    });
    assert!(mutable.is_err());
}

#[test]
fn backend_storage_is_rejected_before_scalar_access() {
    let placement = Placement {
        memory_kind: MemoryKind::Device,
        device: Some(DeviceId {
            kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
            ordinal: 0,
        }),
        cpu_affinity: None,
    };
    let mut tensor = TypedTensor::<f64, Rank<1>>::from_buffer_col_major(
        [2],
        StorageBuffer::Backend(Box::new(BackendStorageHandle::<f64>::new_with_len(1736, 2))),
        placement,
    )
    .unwrap();

    assert!(matches!(
        tensor.host_col_major(),
        Err(crate::Error::RuntimeState { .. })
    ));
    assert!(matches!(
        tensor.host_col_major_mut(),
        Err(crate::Error::RuntimeState { .. })
    ));
}

#[test]
fn poisson_jacobi_step_uses_first_axis_lanes() {
    const N: usize = 6;
    let u = TypedTensor::<f64, Rank<2>>::from_vec_col_major([N, N], vec![0.0; N * N]).unwrap();
    let rhs = TypedTensor::<f64, Rank<2>>::from_vec_col_major([N, N], vec![1.0; N * N]).unwrap();
    let mut next =
        TypedTensor::<f64, Rank<2>>::from_vec_col_major([N, N], vec![0.0; N * N]).unwrap();

    let u = u.host_col_major().unwrap();
    let rhs = rhs.host_col_major().unwrap();
    let u_lanes = u.axis0_lanes().collect::<Vec<_>>();
    let rhs_lanes = rhs.axis0_lanes().collect::<Vec<_>>();
    let mut next = next.host_col_major_mut().unwrap();
    let h2 = 0.04;

    for (j, next_lane) in next.axis0_lanes_mut().enumerate().skip(1).take(N - 2) {
        let left = u_lanes[j - 1];
        let center = u_lanes[j];
        let right = u_lanes[j + 1];
        let rhs_lane = rhs_lanes[j];
        for i in 1..N - 1 {
            next_lane[i] =
                0.25 * (center[i - 1] + center[i + 1] + left[i] + right[i] + h2 * rhs_lane[i]);
        }
    }

    for j in 0..N {
        for i in 0..N {
            let expected = if i == 0 || j == 0 || i + 1 == N || j + 1 == N {
                0.0
            } else {
                0.01
            };
            assert_eq!(next[[i, j]], expected);
        }
    }
}
