//! Downstream faer interoperability (issue #1602).
//!
//! A downstream scientific application can keep its tensors in tenferro and
//! hand tenferro-owned host storage to faer for operations that are not (yet)
//! exposed through tenferro's operation APIs. This binary is the executable
//! source of truth for the faer section of
//! `docs/guides/external-linalg-interop.md`; the guide quotes these regions
//! through `snippet-source` markers.
//!
//! Everything here uses only public tenferro APIs and a direct `faer`
//! dependency pinned to the same version the workspace uses. There is no
//! hidden materialization: a non-contiguous view or a non-host tensor is
//! rejected with an explicit error, and the caller decides when to copy or
//! transfer.

use faer::linalg::matmul::matmul;
use faer::linalg::solvers::Solve;
use faer::{Accum, Mat, MatMut, MatRef, Par};
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{
    BackendStorage, BackendStorageHandle, DeviceId, DeviceKind, MemoryKind, Placement,
    StorageBuffer, TensorViewCanonicalization, TypedTensor, TypedTensorView,
};

fn assert_close(actual: f64, expected: f64, context: &str) {
    let error = (actual - expected).abs();
    assert!(
        error < 1.0e-12,
        "{context}: actual={actual}, expected={expected}, error={error}"
    );
}

fn assert_slice_close(actual: &[f64], expected: &[f64], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: length mismatch");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_close(*actual, *expected, &format!("{context}[{index}]"));
    }
}

// snippet-start:faer-immutable
fn faer_immutable_view() -> tenferro_tensor::Result<()> {
    // Only a compact column-major host tensor can be borrowed without copying.
    // Check both properties explicitly before using the contiguous path.
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])?;
    assert!(a.is_col_major_contiguous()?);
    assert_eq!(a.placement().memory_kind, MemoryKind::UnpinnedHost);

    // Zero-copy faer view. Column-major compact storage means the leading
    // dimension is the number of rows and the flat slice is the whole buffer.
    let (rows, cols) = (a.shape()[0], a.shape()[1]);
    let a_view: MatRef<f64> = MatRef::from_column_major_slice(a.as_slice()?, rows, cols);
    assert_eq!(*a_view.get(1, 2), 6.0); // row 1, column 2

    let b =
        TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0])?;
    let b_view = MatRef::from_column_major_slice(b.as_slice()?, b.shape()[0], b.shape()[1]);

    // faer computes into its own output. Par::Seq keeps the call off faer's
    // global rayon pool; Par::Rayon(n) would use that ambient pool instead.
    let mut product = Mat::<f64>::zeros(rows, b.shape()[1]);
    matmul(
        product.as_mut(),
        Accum::Replace,
        a_view,
        b_view,
        1.0,
        Par::Seq,
    );

    // A * B for A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]:
    // C = [[58,64],[139,154]]. Positions double as the column-major check:
    // the buffer is [58,139,64,154] (column 0 first, rows fastest).
    let product = product.as_ref();
    assert_close(*product.get(0, 0), 58.0, "C00");
    assert_close(*product.get(1, 0), 139.0, "C10");
    assert_close(*product.get(0, 1), 64.0, "C01");
    assert_close(*product.get(1, 1), 154.0, "C11");
    Ok(())
}
// snippet-end:faer-immutable

// snippet-start:faer-mutable
fn faer_mutable_view() -> tenferro_tensor::Result<()> {
    let mut x = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;

    // Writable faer view over tenferro-owned host storage. The mutable borrow
    // is exclusive, so no other reference to the buffer can be live. Scoping
    // the view ends the borrow before the tensor is read again below.
    {
        let mut x_view: MatMut<f64> = MatMut::from_column_major_slice_mut(x.host_data_mut()?, 2, 2);
        for column in x_view.as_mut().col_iter_mut() {
            for element in column.iter_mut() {
                *element *= 2.0;
            }
        }
    }

    // The write happened in place: the tensor now owns the scaled buffer.
    assert_slice_close(x.as_slice()?, &[2.0, 4.0, 6.0, 8.0], "scaled x");
    Ok(())
}
// snippet-end:faer-mutable

// snippet-start:faer-solve
fn faer_solve_on_zero_copy_views() -> tenferro_tensor::Result<()> {
    // Solve A x = b with faer's LU, reading A and b straight from tenferro
    // storage and writing x into faer-owned memory.
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![3.0, 1.0, 1.0, 2.0])?;
    let b = TypedTensor::<f64>::from_vec_col_major(vec![2, 1], vec![5.0, 4.0])?;
    let a_view = MatRef::from_column_major_slice(a.as_slice()?, 2, 2);
    let b_view = MatRef::from_column_major_slice(b.as_slice()?, 2, 1);

    let lu = a_view.partial_piv_lu();
    let mut x = Mat::<f64>::zeros(2, 1);
    x.as_mut().copy_from(b_view);
    lu.solve_in_place(x.as_mut());

    // Verify A x == b. Exact solution: x = [1.2, 1.4] (3*1.2+1.4 == 5,
    // 1.2+2*1.4 == 4). Check both the solution and the reconstruction.
    assert_close(*x.get(0, 0), 1.2, "x0");
    assert_close(*x.get(1, 0), 1.4, "x1");
    let residual0 = *a_view.get(0, 0) * *x.get(0, 0) + *a_view.get(0, 1) * *x.get(1, 0);
    let residual1 = *a_view.get(1, 0) * *x.get(0, 0) + *a_view.get(1, 1) * *x.get(1, 0);
    assert_close(residual0, 5.0, "A x row 0");
    assert_close(residual1, 4.0, "A x row 1");
    Ok(())
}
// snippet-end:faer-solve

// snippet-start:faer-non-contiguous
fn faer_rejects_non_contiguous_without_materializing() -> tenferro_tensor::Result<()> {
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])?;
    // Transposing a 2x3 tensor produces a 3x2 view with non-column-major
    // strides. It is metadata only; nothing has been copied.
    let transposed: TypedTensorView<'_, f64> = a.as_view().transpose_view([1, 0])?;
    assert!(!transposed.is_col_major_contiguous()?);

    // as_slice rejects the view explicitly. There is no hidden materialization.
    let error = transposed.as_slice().unwrap_err();
    assert!(
        error.to_string().contains("not contiguous column-major"),
        "unexpected error: {error}"
    );

    // The caller decides to materialize through an explicit backend call.
    let mut backend = CpuBackend::new();
    let compact = backend.to_contiguous(&transposed)?;
    assert!(compact.is_col_major_contiguous()?);
    // 3x2 column-major copy of the transposed matrix [[1,4],[2,5],[3,6]]:
    // column 0 = [1,2,3], column 1 = [4,5,6].
    assert_slice_close(
        compact.as_slice()?,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "compact",
    );
    Ok(())
}
// snippet-end:faer-non-contiguous

// snippet-start:faer-non-host
fn faer_rejects_non_host_storage() -> tenferro_tensor::Result<()> {
    // A backend-placed tensor (for example GPU memory, or an external provider
    // allocation) cannot be borrowed as a host slice. Construct one with the
    // public from_buffer_col_major API to show the error contract.
    let backend_buffer: Box<dyn BackendStorage<f64>> =
        Box::new(BackendStorageHandle::<f64>::new_with_len(7, 4));
    let device_tensor = TypedTensor::<f64>::from_buffer_col_major(
        vec![2, 2],
        StorageBuffer::Backend(backend_buffer),
        Placement {
            memory_kind: MemoryKind::Device,
            device: Some(DeviceId {
                kind: DeviceKind::Other("example".into()),
                ordinal: 0,
            }),
            cpu_affinity: None,
        },
    )?;
    assert_eq!(device_tensor.placement().memory_kind, MemoryKind::Device);

    // Host access is rejected explicitly; tenferro never downloads on borrow.
    let error = device_tensor.as_slice().unwrap_err();
    assert!(
        error
            .to_string()
            .contains("backend storage cannot be borrowed as host data"),
        "unexpected error: {error}"
    );

    // The required transfer is an explicit download through the owning
    // backend (CpuBackend rejects foreign backend buffers); only then can faer
    // read the data.
    Ok(())
}
// snippet-end:faer-non-host

fn main() -> Result<(), Box<dyn std::error::Error>> {
    faer_immutable_view()?;
    faer_mutable_view()?;
    faer_solve_on_zero_copy_views()?;
    faer_rejects_non_contiguous_without_materializing()?;
    faer_rejects_non_host_storage()?;
    println!("faer_interop: all checks passed");
    Ok(())
}
