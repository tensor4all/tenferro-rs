//! Tests for tenferro-tensor: constructors, metadata, view ops, data ops,
//! Differentiable impl.

use tenferro_device::{ComputeDevice, LogicalMemorySpace, OpKind};
use tenferro_tensor::{DataBuffer, MemoryOrder, Tensor};

const MEM: LogicalMemorySpace = LogicalMemorySpace::MainMemory;
const COL: MemoryOrder = MemoryOrder::ColumnMajor;
const ROW: MemoryOrder = MemoryOrder::RowMajor;

// ============================================================================
// DataBuffer
// ============================================================================

#[test]
fn databuffer_from_vec() {
    let buf = DataBuffer::from_vec(vec![1.0, 2.0, 3.0]);
    assert_eq!(buf.len(), 3);
    assert!(buf.is_owned());
    assert!(!buf.is_gpu());
    assert!(buf.is_unique());
}

#[test]
fn databuffer_as_slice() {
    let buf = DataBuffer::from_vec(vec![1.0_f64, 2.0, 3.0]);
    assert_eq!(buf.as_slice(), Some(&[1.0, 2.0, 3.0][..]));
}

#[test]
fn databuffer_as_mut_slice() {
    let mut buf = DataBuffer::from_vec(vec![1.0_f64, 2.0]);
    buf.as_mut_slice().unwrap()[0] = 42.0;
    assert_eq!(buf.as_slice().unwrap()[0], 42.0);
}

#[test]
fn databuffer_shared_no_mut() {
    let buf = DataBuffer::from_vec(vec![1.0_f64]);
    let _buf2 = buf.clone();
    let mut buf = buf;
    assert!(buf.as_mut_slice().is_none());
}

#[test]
fn databuffer_empty() {
    let buf = DataBuffer::<f64>::from_vec(vec![]);
    assert!(buf.is_empty());
    assert_eq!(buf.len(), 0);
}

#[test]
fn databuffer_clone_shares() {
    let buf = DataBuffer::from_vec(vec![1.0_f64]);
    assert!(buf.is_unique());
    let _buf2 = buf.clone();
    assert!(!buf.is_unique());
}

// ============================================================================
// Tensor constructors
// ============================================================================

#[test]
fn zeros_column_major() {
    let t = Tensor::<f64>::zeros(&[3, 4], MEM, COL);
    assert_eq!(t.dims(), &[3, 4]);
    assert_eq!(t.strides(), &[1, 3]);
    assert_eq!(t.offset(), 0);
    assert_eq!(t.len(), 12);
    let data = t.buffer().as_slice().unwrap();
    assert!(data.iter().all(|&x| x == 0.0));
}

#[test]
fn zeros_row_major() {
    let t = Tensor::<f64>::zeros(&[3, 4], MEM, ROW);
    assert_eq!(t.dims(), &[3, 4]);
    assert_eq!(t.strides(), &[4, 1]);
    let data = t.buffer().as_slice().unwrap();
    assert!(data.iter().all(|&x| x == 0.0));
}

#[test]
fn zeros_scalar() {
    let t = Tensor::<f64>::zeros(&[], MEM, COL);
    assert_eq!(t.dims(), &[] as &[usize]);
    assert_eq!(t.len(), 1);
    assert_eq!(t.ndim(), 0);
}

#[test]
fn zeros_empty_dim() {
    let t = Tensor::<f64>::zeros(&[0, 4], MEM, COL);
    assert_eq!(t.len(), 0);
    assert!(t.is_empty());
}

#[test]
fn ones_basic() {
    let t = Tensor::<f64>::ones(&[2, 3], MEM, COL);
    let data = t.buffer().as_slice().unwrap();
    assert!(data.iter().all(|&x| x == 1.0));
    assert_eq!(t.len(), 6);
}

#[test]
fn from_slice_column_major() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::<f64>::from_slice(&data, &[2, 3], COL).unwrap();
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.strides(), &[1, 2]);
    // Column-major: data[0]=t(0,0), data[1]=t(1,0), data[2]=t(0,1), ...
    let buf = t.buffer().as_slice().unwrap();
    assert_eq!(buf, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn from_slice_row_major() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::<f64>::from_slice(&data, &[2, 3], ROW).unwrap();
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.strides(), &[3, 1]);
}

#[test]
fn from_slice_length_mismatch() {
    let data = [1.0, 2.0, 3.0];
    let result = Tensor::<f64>::from_slice(&data, &[2, 3], COL);
    assert!(result.is_err());
}

#[test]
fn from_vec_basic() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::<f64>::from_vec(data, &[2, 3], &[1, 2], 0).unwrap();
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.strides(), &[1, 2]);
}

#[test]
fn from_vec_with_offset() {
    let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::<f64>::from_vec(data, &[2, 3], &[1, 2], 1).unwrap();
    assert_eq!(t.offset(), 1);
}

#[test]
fn from_vec_invalid_layout() {
    let data = vec![1.0, 2.0, 3.0];
    let result = Tensor::<f64>::from_vec(data, &[2, 3], &[1, 2], 0);
    assert!(result.is_err());
}

#[test]
fn from_vec_strides_length_mismatch() {
    let data = vec![1.0; 6];
    let result = Tensor::<f64>::from_vec(data, &[2, 3], &[1], 0);
    assert!(result.is_err());
}

#[test]
fn eye_3x3_col_major() {
    let t = Tensor::<f64>::eye(3, MEM, COL);
    assert_eq!(t.dims(), &[3, 3]);
    let data = t.buffer().as_slice().unwrap();
    // Column-major: [1,0,0, 0,1,0, 0,0,1]
    assert_eq!(data, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn eye_3x3_row_major() {
    let t = Tensor::<f64>::eye(3, MEM, ROW);
    assert_eq!(t.dims(), &[3, 3]);
    let data = t.buffer().as_slice().unwrap();
    // Row-major: [1,0,0, 0,1,0, 0,0,1]
    assert_eq!(data, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn eye_1x1() {
    let t = Tensor::<f64>::eye(1, MEM, COL);
    assert_eq!(t.buffer().as_slice().unwrap(), &[1.0]);
}

// ============================================================================
// Metadata
// ============================================================================

#[test]
fn ndim() {
    let t = Tensor::<f64>::zeros(&[2, 3, 4], MEM, COL);
    assert_eq!(t.ndim(), 3);
}

#[test]
fn len_3d() {
    let t = Tensor::<f64>::zeros(&[2, 3, 4], MEM, COL);
    assert_eq!(t.len(), 24);
}

#[test]
fn is_empty_nonempty() {
    let t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    assert!(!t.is_empty());
}

#[test]
fn logical_memory_space() {
    let t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    assert_eq!(t.logical_memory_space(), MEM);
}

#[test]
fn preferred_compute_device_default_none() {
    let t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    assert!(t.preferred_compute_device().is_none());
}

#[test]
fn set_preferred_compute_device() {
    let mut t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    let dev = ComputeDevice::Cpu { device_id: 0 };
    t.set_preferred_compute_device(Some(dev));
    assert_eq!(t.preferred_compute_device(), Some(dev));
}

#[test]
fn effective_compute_devices_default() {
    let t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    let devs = t.effective_compute_devices(OpKind::BatchedGemm).unwrap();
    assert_eq!(devs, vec![ComputeDevice::Cpu { device_id: 0 }]);
}

#[test]
fn effective_compute_devices_override() {
    let mut t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    let dev = ComputeDevice::Cpu { device_id: 1 };
    t.set_preferred_compute_device(Some(dev));
    let devs = t.effective_compute_devices(OpKind::Contract).unwrap();
    assert_eq!(devs, vec![dev]);
}

#[test]
fn is_conjugated_default_false() {
    let t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    assert!(!t.is_conjugated());
}

#[test]
fn is_ready_cpu() {
    let t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    assert!(t.is_ready());
}

// ============================================================================
// Clone (shallow)
// ============================================================================

#[test]
fn clone_shares_buffer() {
    let t = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3], COL).unwrap();
    let t2 = t.clone();
    assert!(!t.buffer().is_unique());
    assert!(!t2.buffer().is_unique());
    assert_eq!(t.buffer().as_ptr(), t2.buffer().as_ptr());
}

// ============================================================================
// Conjugation
// ============================================================================

#[test]
fn conj_toggles_flag() {
    let t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    let tc = t.conj();
    assert!(tc.is_conjugated());
    let tcc = tc.conj();
    assert!(!tcc.is_conjugated());
}

#[test]
fn into_conj() {
    let t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    let tc = t.into_conj();
    assert!(tc.is_conjugated());
}

// ============================================================================
// is_contiguous
// ============================================================================

#[test]
fn contiguous_col_major() {
    let t = Tensor::<f64>::zeros(&[3, 4], MEM, COL);
    assert!(t.is_contiguous());
}

#[test]
fn contiguous_row_major() {
    let t = Tensor::<f64>::zeros(&[3, 4], MEM, ROW);
    assert!(t.is_contiguous());
}

#[test]
fn not_contiguous_after_permute() {
    let t = Tensor::<f64>::zeros(&[3, 4], MEM, COL);
    let tp = t.permute(&[1, 0]).unwrap();
    // Transposed column-major: strides [3, 1] with dims [4, 3]
    // This is actually row-major contiguous! So it IS contiguous.
    // Let's test with a 3D case instead.
    let t3 = Tensor::<f64>::zeros(&[2, 3, 4], MEM, COL);
    let tp3 = t3.permute(&[2, 0, 1]).unwrap();
    // dims [4, 2, 3], strides [6, 1, 2] — not contiguous in either order
    assert!(!tp3.is_contiguous());
    // But the 2D transpose IS contiguous (row-major)
    assert!(tp.is_contiguous());
}

// ============================================================================
// Permute
// ============================================================================

#[test]
fn permute_transpose() {
    let t = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], COL).unwrap();
    let tp = t.permute(&[1, 0]).unwrap();
    assert_eq!(tp.dims(), &[3, 2]);
    assert_eq!(tp.strides(), &[2, 1]);
}

#[test]
fn permute_identity() {
    let t = Tensor::<f64>::zeros(&[3, 4, 5], MEM, COL);
    let tp = t.permute(&[0, 1, 2]).unwrap();
    assert_eq!(tp.dims(), &[3, 4, 5]);
    assert_eq!(tp.strides(), t.strides());
}

#[test]
fn permute_3d() {
    let t = Tensor::<f64>::zeros(&[2, 3, 4], MEM, COL);
    let tp = t.permute(&[2, 0, 1]).unwrap();
    assert_eq!(tp.dims(), &[4, 2, 3]);
    // Col-major strides: [1, 2, 6] → permuted: [6, 1, 2]
    assert_eq!(tp.strides(), &[6, 1, 2]);
}

#[test]
fn permute_invalid_length() {
    let t = Tensor::<f64>::zeros(&[3, 4], MEM, COL);
    assert!(t.permute(&[0]).is_err());
}

#[test]
fn permute_out_of_range() {
    let t = Tensor::<f64>::zeros(&[3, 4], MEM, COL);
    assert!(t.permute(&[0, 5]).is_err());
}

#[test]
fn permute_duplicate() {
    let t = Tensor::<f64>::zeros(&[3, 4], MEM, COL);
    assert!(t.permute(&[0, 0]).is_err());
}

// ============================================================================
// Broadcast
// ============================================================================

#[test]
fn broadcast_expand_dim1() {
    let t = Tensor::<f64>::ones(&[3, 1], MEM, COL);
    let b = t.broadcast(&[3, 4]).unwrap();
    assert_eq!(b.dims(), &[3, 4]);
    assert_eq!(b.strides()[1], 0);
}

#[test]
fn broadcast_same_shape() {
    let t = Tensor::<f64>::ones(&[3, 4], MEM, COL);
    let b = t.broadcast(&[3, 4]).unwrap();
    assert_eq!(b.dims(), &[3, 4]);
    assert_eq!(b.strides(), t.strides());
}

#[test]
fn broadcast_incompatible() {
    let t = Tensor::<f64>::ones(&[3, 2], MEM, COL);
    assert!(t.broadcast(&[3, 4]).is_err());
}

// ============================================================================
// Diagonal
// ============================================================================

#[test]
fn diagonal_2d() {
    let t = Tensor::<f64>::eye(3, MEM, COL);
    let d = t.diagonal(&[(0, 1)]).unwrap();
    assert_eq!(d.dims(), &[3]);
    // Col-major strides [1, 3] → diagonal stride 1+3=4
    assert_eq!(d.strides(), &[4]);
}

#[test]
fn diagonal_data_access() {
    let data = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
    let t = Tensor::<f64>::from_slice(&data, &[3, 3], COL).unwrap();
    let d = t.diagonal(&[(0, 1)]).unwrap();
    // Diagonal elements: d(0)=t(0,0)=1, d(1)=t(1,1)=2, d(2)=t(2,2)=3
    let buf = d.buffer().as_slice().unwrap();
    let off = d.offset();
    let s = d.strides()[0];
    assert_eq!(buf[(off + 0 * s) as usize], 1.0);
    assert_eq!(buf[(off + 1 * s) as usize], 2.0);
    assert_eq!(buf[(off + 2 * s) as usize], 3.0);
}

#[test]
fn diagonal_mismatched_dims() {
    let t = Tensor::<f64>::zeros(&[3, 4], MEM, COL);
    assert!(t.diagonal(&[(0, 1)]).is_err());
}

#[test]
fn diagonal_same_axis() {
    let t = Tensor::<f64>::zeros(&[3, 3], MEM, COL);
    assert!(t.diagonal(&[(0, 0)]).is_err());
}

// ============================================================================
// Reshape
// ============================================================================

#[test]
fn reshape_flatten() {
    let t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    let r = t.reshape(&[6]).unwrap();
    assert_eq!(r.dims(), &[6]);
    assert_eq!(r.strides(), &[1]);
}

#[test]
fn reshape_same_shape() {
    let t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    let r = t.reshape(&[2, 3]).unwrap();
    assert_eq!(r.dims(), &[2, 3]);
}

#[test]
fn reshape_different_shape() {
    let t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    let r = t.reshape(&[3, 2]).unwrap();
    assert_eq!(r.dims(), &[3, 2]);
}

#[test]
fn reshape_incompatible_size() {
    let t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    assert!(t.reshape(&[5]).is_err());
}

#[test]
fn reshape_non_contiguous_fails() {
    let t = Tensor::<f64>::zeros(&[2, 3, 4], MEM, COL);
    let tp = t.permute(&[2, 0, 1]).unwrap();
    assert!(tp.reshape(&[24]).is_err());
}

// ============================================================================
// Select
// ============================================================================

#[test]
fn select_dim0() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::<f64>::from_slice(&data, &[2, 3], COL).unwrap();
    let s = t.select(0, 1).unwrap();
    assert_eq!(s.dims(), &[3]);
    // Row 1: t(1,0)=2, t(1,1)=4, t(1,2)=6
    let buf = s.buffer().as_slice().unwrap();
    assert_eq!(buf[(s.offset() + 0 * s.strides()[0]) as usize], 2.0);
    assert_eq!(buf[(s.offset() + 1 * s.strides()[0]) as usize], 4.0);
    assert_eq!(buf[(s.offset() + 2 * s.strides()[0]) as usize], 6.0);
}

#[test]
fn select_dim1() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::<f64>::from_slice(&data, &[2, 3], COL).unwrap();
    let s = t.select(1, 2).unwrap();
    assert_eq!(s.dims(), &[2]);
    // Col 2: t(0,2)=5, t(1,2)=6
    let buf = s.buffer().as_slice().unwrap();
    assert_eq!(buf[(s.offset() + 0 * s.strides()[0]) as usize], 5.0);
    assert_eq!(buf[(s.offset() + 1 * s.strides()[0]) as usize], 6.0);
}

#[test]
fn select_out_of_range_dim() {
    let t = Tensor::<f64>::zeros(&[3, 4], MEM, COL);
    assert!(t.select(2, 0).is_err());
}

#[test]
fn select_out_of_range_index() {
    let t = Tensor::<f64>::zeros(&[3, 4], MEM, COL);
    assert!(t.select(0, 3).is_err());
}

// ============================================================================
// Narrow
// ============================================================================

#[test]
fn narrow_basic() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::<f64>::from_slice(&data, &[2, 3], COL).unwrap();
    let n = t.narrow(1, 1, 2).unwrap();
    assert_eq!(n.dims(), &[2, 2]);
    // Cols 1..3: t(0,1)=3, t(1,1)=4, t(0,2)=5, t(1,2)=6
    let buf = n.buffer().as_slice().unwrap();
    let off = n.offset();
    assert_eq!(buf[(off + 0) as usize], 3.0); // t(0,1)
    assert_eq!(buf[(off + 1) as usize], 4.0); // t(1,1)
}

#[test]
fn narrow_full_range() {
    let t = Tensor::<f64>::zeros(&[3, 4], MEM, COL);
    let n = t.narrow(0, 0, 3).unwrap();
    assert_eq!(n.dims(), &[3, 4]);
    assert_eq!(n.offset(), 0);
}

#[test]
fn narrow_out_of_range() {
    let t = Tensor::<f64>::zeros(&[3, 4], MEM, COL);
    assert!(t.narrow(1, 3, 2).is_err()); // 3+2=5 > 4
}

// ============================================================================
// Contiguous / into_contiguous
// ============================================================================

#[test]
fn contiguous_already_contiguous() {
    let t = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let c = t.contiguous(COL);
    assert_eq!(c.buffer().as_slice(), t.buffer().as_slice());
}

#[test]
fn contiguous_from_permuted() {
    let t = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], COL).unwrap();
    let tp = t.permute(&[1, 0]).unwrap(); // [3, 2], strides [2, 1]
    let c = tp.contiguous(COL);
    assert_eq!(c.dims(), &[3, 2]);
    assert_eq!(c.strides(), &[1, 3]);
    assert_eq!(c.offset(), 0);
    assert!(c.is_contiguous());
    // Verify data: tp(0,0)=t(0,0)=1, tp(0,1)=t(1,0)=2,
    //              tp(1,0)=t(0,1)=3, tp(1,1)=t(1,1)=4,
    //              tp(2,0)=t(0,2)=5, tp(2,1)=t(1,2)=6
    // Col-major [3,2]: layout is [tp(0,0),tp(1,0),tp(2,0), tp(0,1),tp(1,1),tp(2,1)]
    //                            = [1, 3, 5, 2, 4, 6]
    assert_eq!(
        c.buffer().as_slice().unwrap(),
        &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    );
}

#[test]
fn into_contiguous_passthrough() {
    let t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    let ptr = t.buffer().as_ptr();
    let c = t.into_contiguous(COL);
    assert_eq!(c.buffer().as_ptr(), ptr);
}

#[test]
fn into_contiguous_copies_when_needed() {
    let t = Tensor::<f64>::zeros(&[2, 3, 4], MEM, COL);
    let tp = t.permute(&[2, 0, 1]).unwrap();
    let c = tp.into_contiguous(COL);
    assert!(c.is_contiguous());
    assert_eq!(c.offset(), 0);
}

// ============================================================================
// Tril / Triu
// ============================================================================

#[test]
fn tril_3x3() {
    let t = Tensor::<f64>::ones(&[3, 3], MEM, COL);
    let lower = t.tril(0);
    assert_eq!(lower.dims(), &[3, 3]);
    let data = lower.buffer().as_slice().unwrap();
    // Col-major: col 0 = [1,1,1], col 1 = [0,1,1], col 2 = [0,0,1]
    assert_eq!(data, &[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn triu_3x3() {
    let t = Tensor::<f64>::ones(&[3, 3], MEM, COL);
    let upper = t.triu(0);
    assert_eq!(upper.dims(), &[3, 3]);
    let data = upper.buffer().as_slice().unwrap();
    // Col-major: col 0 = [1,0,0], col 1 = [1,1,0], col 2 = [1,1,1]
    assert_eq!(data, &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
}

#[test]
fn tril_with_diagonal_offset() {
    let t = Tensor::<f64>::ones(&[3, 3], MEM, COL);
    let lower = t.tril(1);
    let data = lower.buffer().as_slice().unwrap();
    // tril(1): keep where j - i <= 1
    // Col 0: all kept [1,1,1]; Col 1: row 0,1,2 where j=1: 1-0=1<=1, 1-1=0<=1, 1-2=-1<=1 → all kept [1,1,1]
    // Col 2: 2-0=2>1 → 0; 2-1=1<=1 → 1; 2-2=0<=1 → 1 → [0,1,1]
    assert_eq!(data, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
}

#[test]
fn triu_with_negative_diagonal() {
    let t = Tensor::<f64>::ones(&[3, 3], MEM, COL);
    let upper = t.triu(-1);
    let data = upper.buffer().as_slice().unwrap();
    // triu(-1): keep where j - i >= -1
    // Col 0: 0-0=0>=-1, 0-1=-1>=-1, 0-2=-2<-1 → [1,1,0]
    // Col 1: 1-0=1>=-1, 1-1=0>=-1, 1-2=-1>=-1 → [1,1,1]
    // Col 2: all >= -1 → [1,1,1]
    assert_eq!(data, &[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn tril_triu_complementary() {
    let t = Tensor::<f64>::ones(&[3, 3], MEM, COL);
    let lower = t.tril(0);
    let upper = t.triu(1);
    // tril(0) + triu(1) should reconstruct the original
    let l_data = lower.buffer().as_slice().unwrap();
    let u_data = upper.buffer().as_slice().unwrap();
    for i in 0..9 {
        assert_eq!(l_data[i] + u_data[i], 1.0, "mismatch at index {i}");
    }
}

// ============================================================================
// to_memory_space_async
// ============================================================================

#[test]
fn to_memory_space_same_space() {
    let t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    let t2 = t.to_memory_space_async(MEM).unwrap();
    assert_eq!(t2.dims(), &[2, 3]);
}

#[test]
fn to_memory_space_gpu_fails() {
    let t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    assert!(t
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .is_err());
}

// ============================================================================
// TensorView operations
// ============================================================================

#[test]
fn tensor_view_metadata() {
    let t = Tensor::<f64>::zeros(&[3, 4], MEM, COL);
    let v = t.tensor_view();
    assert_eq!(v.dims(), &[3, 4]);
    assert_eq!(v.strides(), t.strides());
    assert_eq!(v.ndim(), 2);
    assert_eq!(v.offset(), 0);
    assert!(!v.is_conjugated());
    assert_eq!(v.logical_memory_space(), MEM);
    assert!(v.preferred_compute_device().is_none());
}

#[test]
fn tensor_view_permute() {
    let t = Tensor::<f64>::zeros(&[3, 4], MEM, COL);
    let v = t.tensor_view();
    let vp = v.permute(&[1, 0]).unwrap();
    assert_eq!(vp.dims(), &[4, 3]);
}

#[test]
fn tensor_view_broadcast() {
    let t = Tensor::<f64>::ones(&[1, 3], MEM, COL);
    let v = t.tensor_view();
    let vb = v.broadcast(&[4, 3]).unwrap();
    assert_eq!(vb.dims(), &[4, 3]);
    assert_eq!(vb.strides()[0], 0);
}

#[test]
fn tensor_view_select() {
    let t = Tensor::<f64>::zeros(&[3, 4, 5], MEM, COL);
    let v = t.tensor_view();
    let vs = v.select(2, 2).unwrap();
    assert_eq!(vs.dims(), &[3, 4]);
}

#[test]
fn tensor_view_narrow() {
    let t = Tensor::<f64>::zeros(&[3, 10], MEM, COL);
    let v = t.tensor_view();
    let vn = v.narrow(1, 2, 3).unwrap();
    assert_eq!(vn.dims(), &[3, 3]);
}

#[test]
fn tensor_view_to_tensor() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::<f64>::from_slice(&data, &[2, 3], COL).unwrap();
    let v = t.tensor_view();
    let vp = v.permute(&[1, 0]).unwrap();
    let owned = vp.to_tensor(COL);
    assert_eq!(owned.dims(), &[3, 2]);
    assert!(owned.is_contiguous());
}

#[test]
fn tensor_view_conj() {
    let t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    let v = t.tensor_view();
    let vc = v.conj();
    assert!(vc.is_conjugated());
}

// ============================================================================
// Differentiable
// ============================================================================

#[test]
fn zero_tangent() {
    use chainrules_core::Differentiable;

    let t = Tensor::<f64>::ones(&[2, 3], MEM, COL);
    let zt = t.zero_tangent();
    assert_eq!(zt.dims(), &[2, 3]);
    let data = zt.buffer().as_slice().unwrap();
    assert!(data.iter().all(|&x| x == 0.0));
}

#[test]
fn accumulate_tangent_basic() {
    use chainrules_core::Differentiable;

    let a = Tensor::<f64>::ones(&[2, 3], MEM, COL);
    let b = Tensor::<f64>::ones(&[2, 3], MEM, COL);
    let result = Tensor::<f64>::accumulate_tangent(a, &b);
    assert_eq!(result.dims(), &[2, 3]);
    let data = result.buffer().as_slice().unwrap();
    assert!(data.iter().all(|&x| x == 2.0));
}

#[test]
fn accumulate_tangent_with_zero() {
    use chainrules_core::Differentiable;

    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3], COL).unwrap();
    let b = a.zero_tangent();
    let result = Tensor::<f64>::accumulate_tangent(a, &b);
    let data = result.buffer().as_slice().unwrap();
    assert_eq!(data, &[1.0, 2.0, 3.0]);
}
