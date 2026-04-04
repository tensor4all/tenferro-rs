use crate::v2::types::{DType, Tensor, TypedTensor};
use computegraph::Operand;

#[test]
fn test_zeros_ones() {
    let z = TypedTensor::<f64>::zeros(vec![2, 3]);
    assert_eq!(z.shape, vec![2, 3]);
    assert_eq!(z.n_elements(), 6);
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(*z.get(&[i, j]), 0.0);
        }
    }

    let o = TypedTensor::<f64>::ones(vec![2, 3]);
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(*o.get(&[i, j]), 1.0);
        }
    }
}

#[test]
fn test_from_vec_col_major() {
    // Column-major: data = [1,2,3,4,5,6] with shape [2,3]
    // Column-major strides: [1, 2]
    // (0,0)->0=1, (1,0)->1=2, (0,1)->2=3, (1,1)->3=4, (0,2)->4=5, (1,2)->5=6
    let t = TypedTensor::<f64>::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(*t.get(&[0, 0]), 1.0);
    assert_eq!(*t.get(&[1, 0]), 2.0);
    assert_eq!(*t.get(&[0, 1]), 3.0);
    assert_eq!(*t.get(&[1, 1]), 4.0);
    assert_eq!(*t.get(&[0, 2]), 5.0);
    assert_eq!(*t.get(&[1, 2]), 6.0);
}

#[test]
fn test_reshape() {
    let t = Tensor::F64(TypedTensor::from_vec(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let r = t.reshape(&[3, 2]);
    assert_eq!(r.shape(), &[3, 2]);
    // After reshape from [2,3] col-major to [3,2] col-major:
    // The data bytes are preserved. Col-major [2,3] data = [1,2,3,4,5,6]
    // reinterpreted as [3,2] col-major: strides [1,3]
    // (0,0)->0=1, (1,0)->1=2, (2,0)->2=3, (0,1)->3=4, (1,1)->4=5, (2,1)->5=6
    match &r {
        Tensor::F64(inner) => {
            assert_eq!(*inner.get(&[0, 0]), 1.0);
            assert_eq!(*inner.get(&[1, 0]), 2.0);
            assert_eq!(*inner.get(&[2, 0]), 3.0);
            assert_eq!(*inner.get(&[0, 1]), 4.0);
            assert_eq!(*inner.get(&[1, 1]), 5.0);
            assert_eq!(*inner.get(&[2, 1]), 6.0);
        }
        _ => panic!("unexpected dtype"),
    }
}

#[test]
fn test_add_mul() {
    let a = Tensor::F64(TypedTensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
    let b = Tensor::F64(TypedTensor::from_vec(
        vec![2, 2],
        vec![10.0, 20.0, 30.0, 40.0],
    ));
    let sum = a.add(&b);
    let prod = a.multiply(&b);

    match &sum {
        Tensor::F64(t) => {
            assert_eq!(*t.get(&[0, 0]), 11.0);
            assert_eq!(*t.get(&[1, 0]), 22.0);
            assert_eq!(*t.get(&[0, 1]), 33.0);
            assert_eq!(*t.get(&[1, 1]), 44.0);
        }
        _ => panic!("unexpected dtype"),
    }

    match &prod {
        Tensor::F64(t) => {
            assert_eq!(*t.get(&[0, 0]), 10.0);
            assert_eq!(*t.get(&[1, 0]), 40.0);
            assert_eq!(*t.get(&[0, 1]), 90.0);
            assert_eq!(*t.get(&[1, 1]), 160.0);
        }
        _ => panic!("unexpected dtype"),
    }
}

#[test]
fn test_reduce_sum_2d() {
    // Shape [2,3], reduce axis 0 -> shape [3]
    let t = Tensor::F64(TypedTensor::from_vec(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let r = t.reduce_sum(&[0]);
    assert_eq!(r.shape(), &[3]);
    match &r {
        Tensor::F64(inner) => {
            // col-major [2,3]: col0=[1,2], col1=[3,4], col2=[5,6]
            // reduce axis 0: [3, 7, 11]
            assert_eq!(*inner.get(&[0]), 3.0);
            assert_eq!(*inner.get(&[1]), 7.0);
            assert_eq!(*inner.get(&[2]), 11.0);
        }
        _ => panic!("unexpected dtype"),
    }
}

#[test]
fn test_reduce_sum_all() {
    let t = Tensor::F64(TypedTensor::from_vec(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let r = t.reduce_sum(&[0, 1]);
    assert_eq!(r.shape(), &[1]);
    match &r {
        Tensor::F64(inner) => {
            assert_eq!(*inner.get(&[0]), 21.0);
        }
        _ => panic!("unexpected dtype"),
    }
}

#[test]
fn test_dot_general_matmul() {
    // A: [2,3], B: [3,4] -> C: [2,4]
    // lhs_contracting=[1], rhs_contracting=[0], no batch
    // Col-major A [2,3]: data = [a00,a10, a01,a11, a02,a12]
    let a_data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    // A = [[1,2,3],[4,5,6]]
    let a = Tensor::F64(TypedTensor::from_vec(vec![2, 3], a_data));

    // Col-major B [3,4]: data by columns
    // B = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
    let b_data = vec![
        1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
    ];
    let b = Tensor::F64(TypedTensor::from_vec(vec![3, 4], b_data));

    let c = a.dot_general(&b, &[1], &[0], &[], &[]);
    assert_eq!(c.shape(), &[2, 4]);

    match &c {
        Tensor::F64(t) => {
            // C[i,j] = sum_k A[i,k]*B[k,j]
            // C[0,0] = 1*1 + 2*5 + 3*9 = 38
            assert_eq!(*t.get(&[0, 0]), 38.0);
            // C[1,0] = 4*1 + 5*5 + 6*9 = 83
            assert_eq!(*t.get(&[1, 0]), 83.0);
            // C[0,1] = 1*2 + 2*6 + 3*10 = 44
            assert_eq!(*t.get(&[0, 1]), 44.0);
            // C[1,1] = 4*2 + 5*6 + 6*10 = 98
            assert_eq!(*t.get(&[1, 1]), 98.0);
            // C[0,3] = 1*4 + 2*8 + 3*12 = 56
            assert_eq!(*t.get(&[0, 3]), 56.0);
            // C[1,3] = 4*4 + 5*8 + 6*12 = 128
            assert_eq!(*t.get(&[1, 3]), 128.0);
        }
        _ => panic!("unexpected dtype"),
    }
}

#[test]
fn test_transpose() {
    // Shape [2,3] -> perm [1,0] -> shape [3,2]
    let t = Tensor::F64(TypedTensor::from_vec(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let tr = t.transpose_perm(&[1, 0]);
    assert_eq!(tr.shape(), &[3, 2]);
    match &tr {
        Tensor::F64(inner) => {
            // Original col-major [2,3]: (0,0)=1, (1,0)=2, (0,1)=3, (1,1)=4, (0,2)=5, (1,2)=6
            // Transposed [3,2]: (j,i) = original(i,j)
            assert_eq!(*inner.get(&[0, 0]), 1.0);
            assert_eq!(*inner.get(&[0, 1]), 2.0);
            assert_eq!(*inner.get(&[1, 0]), 3.0);
            assert_eq!(*inner.get(&[1, 1]), 4.0);
            assert_eq!(*inner.get(&[2, 0]), 5.0);
            assert_eq!(*inner.get(&[2, 1]), 6.0);
        }
        _ => panic!("unexpected dtype"),
    }
}

#[test]
fn test_broadcast_in_dim() {
    // Scalar-like [1] -> [3]
    let t = Tensor::F64(TypedTensor::from_vec(vec![1], vec![5.0]));
    let b = t.broadcast_in_dim(&[3], &[0]);
    assert_eq!(b.shape(), &[3]);
    match &b {
        Tensor::F64(inner) => {
            assert_eq!(*inner.get(&[0]), 5.0);
            assert_eq!(*inner.get(&[1]), 5.0);
            assert_eq!(*inner.get(&[2]), 5.0);
        }
        _ => panic!("unexpected dtype"),
    }

    // Vector [3] -> matrix [3,2], dims=[0] (input dim 0 maps to output dim 0)
    let v = Tensor::F64(TypedTensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]));
    let m = v.broadcast_in_dim(&[3, 2], &[0]);
    assert_eq!(m.shape(), &[3, 2]);
    match &m {
        Tensor::F64(inner) => {
            for j in 0..2 {
                assert_eq!(*inner.get(&[0, j]), 1.0);
                assert_eq!(*inner.get(&[1, j]), 2.0);
                assert_eq!(*inner.get(&[2, j]), 3.0);
            }
        }
        _ => panic!("unexpected dtype"),
    }
}

#[test]
fn test_neg() {
    let t = Tensor::F64(TypedTensor::from_vec(vec![2], vec![3.0, -7.0]));
    let n = t.neg();
    match &n {
        Tensor::F64(inner) => {
            assert_eq!(*inner.get(&[0]), -3.0);
            assert_eq!(*inner.get(&[1]), 7.0);
        }
        _ => panic!("unexpected dtype"),
    }
}

#[test]
fn test_conj_real() {
    let t = Tensor::F64(TypedTensor::from_vec(vec![2], vec![3.0, -7.0]));
    let c = t.conj();
    match &c {
        Tensor::F64(inner) => {
            assert_eq!(*inner.get(&[0]), 3.0);
            assert_eq!(*inner.get(&[1]), -7.0);
        }
        _ => panic!("unexpected dtype"),
    }
}

#[test]
fn test_conj_complex() {
    use num_complex::Complex64;
    let t = Tensor::C64(TypedTensor::from_vec(
        vec![2],
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)],
    ));
    let c = t.conj();
    match &c {
        Tensor::C64(inner) => {
            assert_eq!(*inner.get(&[0]), Complex64::new(1.0, -2.0));
            assert_eq!(*inner.get(&[1]), Complex64::new(3.0, 4.0));
        }
        _ => panic!("unexpected dtype"),
    }
}

#[test]
fn test_zero_one_factory() {
    let z = Tensor::zero(&[2, 3]);
    assert_eq!(z.shape(), &[2, 3]);
    assert_eq!(z.dtype(), DType::F64);

    let o = Tensor::one(&[2, 3]);
    assert_eq!(o.shape(), &[2, 3]);
    assert_eq!(o.dtype(), DType::F64);
}

#[test]
fn test_extract_diagonal_2d() {
    // 3x3 matrix -> [3] diagonal
    // Col-major data: [1,2,3,4,5,6,7,8,9]
    // a[0,0]=1, a[1,0]=2, a[2,0]=3, a[0,1]=4, a[1,1]=5, a[2,1]=6, a[0,2]=7, a[1,2]=8, a[2,2]=9
    // Diagonal: [1, 5, 9]
    let t = Tensor::F64(TypedTensor::from_vec(
        vec![3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ));
    let d = t.extract_diagonal(0, 1);
    assert_eq!(d.shape(), &[3]);
    match &d {
        Tensor::F64(inner) => {
            assert_eq!(*inner.get(&[0]), 1.0);
            assert_eq!(*inner.get(&[1]), 5.0);
            assert_eq!(*inner.get(&[2]), 9.0);
        }
        _ => panic!("unexpected dtype"),
    }
}

#[test]
fn test_extract_diagonal_3d() {
    // [2,3,3] tensor -> [2,3] tensor (diagonal along axes 1,2)
    // Col-major: strides [1,2,6]
    // t[b,i,j] with b=0..2, i=0..3, j=0..3
    let data: Vec<f64> = (1..=18).map(|x| x as f64).collect();
    let t = Tensor::F64(TypedTensor::from_vec(vec![2, 3, 3], data));
    let d = t.extract_diagonal(1, 2);
    assert_eq!(d.shape(), &[2, 3]);
    // d[b,i] = t[b,i,i]
    match &d {
        Tensor::F64(inner) => match &t {
            Tensor::F64(orig) => {
                for b in 0..2 {
                    for i in 0..3 {
                        assert_eq!(
                            *inner.get(&[b, i]),
                            *orig.get(&[b, i, i]),
                            "d[{b},{i}] != t[{b},{i},{i}]"
                        );
                    }
                }
            }
            _ => panic!("unexpected dtype"),
        },
        _ => panic!("unexpected dtype"),
    }
}
