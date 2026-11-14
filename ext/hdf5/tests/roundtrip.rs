//! Round trips: every supported dtype, every shape class, and the two entry points.

use hdf5_rt::Extents;
use num_complex::{Complex32, Complex64};
use tenferro_hdf5::hdf5;
use tenferro_hdf5::{
    read_info, read_tensor, read_typed, read_typed_with, write_tensor, write_typed, ReadOptions,
};
use tenferro_tensor::{
    DType, DynRank, Dynamic, Host, Rank, Tensor, TensorScalar, TypedTensor, TypedTensorView,
};

fn fixture() -> (tempfile::TempDir, hdf5::File, hdf5::Group) {
    let dir = tempfile::tempdir().unwrap();
    let file = hdf5::File::create(dir.path().join("tensors.h5")).unwrap();
    let group = file.create_group("tensors").unwrap();
    (dir, file, group)
}

fn shapes() -> Vec<Vec<usize>> {
    vec![vec![], vec![5], vec![3, 4], vec![2, 0, 3], vec![2, 3, 4, 5]]
}

fn stored_extents(group: &hdf5::Group, name: &str) -> Vec<usize> {
    let dataset = group.dataset(name).unwrap();
    match dataset.space().unwrap().extents().unwrap() {
        Extents::Scalar => Vec::new(),
        Extents::Simple(simple) => simple.iter().map(|extent| extent.dim).collect(),
        other => panic!("unexpected dataspace {other:?}"),
    }
}

fn roundtrip<T>(shape: &[usize], values: Vec<T>)
where
    T: TensorScalar + hdf5::H5Type + PartialEq + core::fmt::Debug,
{
    let (_dir, _file, group) = fixture();
    let tensor =
        TypedTensor::<T, DynRank, Host>::from_host_vec_col_major(shape.to_vec(), values.clone())
            .unwrap();
    write_typed(&group, "a", tensor.as_view()).unwrap();

    let info = read_info(&group, "a").unwrap();
    assert_eq!(info.dtype, T::dtype());
    assert_eq!(info.shape, shape.to_vec());
    assert_eq!(info.version, 1);

    let back: TypedTensor<T, DynRank, Host> = read_typed(&group, "a").unwrap();
    assert_eq!(back.shape(), shape);
    assert_eq!(back.as_slice(), values.as_slice());

    let expected_extents: Vec<usize> = shape.iter().rev().copied().collect();
    assert_eq!(stored_extents(&group, "a"), expected_extents);
}

macro_rules! dtype_cases {
    ($test:ident, $ty:ty, $gen:expr) => {
        #[test]
        fn $test() {
            for shape in shapes() {
                let n: usize = shape.iter().product();
                roundtrip::<$ty>(&shape, $gen(n));
            }
        }
    };
}

dtype_cases!(roundtrip_f32, f32, |n| (0..n)
    .map(|i| i as f32 * 0.25 - 1.5)
    .collect());
dtype_cases!(roundtrip_f64, f64, |n| (0..n)
    .map(|i| i as f64 * -0.5 + 3.25)
    .collect());
dtype_cases!(roundtrip_i32, i32, |n| (0..n)
    .map(|i| i as i32 * 7 - 11)
    .collect());
dtype_cases!(roundtrip_i64, i64, |n| (0..n)
    .map(|i| i as i64 * 1_000_003 - 5)
    .collect());
dtype_cases!(roundtrip_bool, bool, |n| (0..n)
    .map(|i| i % 3 == 1)
    .collect());
dtype_cases!(roundtrip_c32, Complex32, |n| (0..n)
    .map(|i| Complex32::new(i as f32, -(i as f32) * 0.5))
    .collect());
dtype_cases!(roundtrip_c64, Complex64, |n| (0..n)
    .map(|i| Complex64::new(i as f64 * 0.125, i as f64))
    .collect());

#[test]
fn float_bit_patterns_survive() {
    let values = vec![
        f64::from_bits(0x7ff8_dead_beef_0001),
        -0.0,
        0.0,
        f64::from_bits(1),
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MAX,
    ];
    let (_dir, _file, group) = fixture();
    let tensor = TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(
        vec![values.len()],
        values.clone(),
    )
    .unwrap();
    write_typed(&group, "a", tensor.as_view()).unwrap();
    let back: TypedTensor<f64, DynRank, Host> = read_typed(&group, "a").unwrap();
    let bits: Vec<u64> = back
        .as_slice()
        .iter()
        .map(|value| value.to_bits())
        .collect();
    let expected: Vec<u64> = values.iter().map(|value| value.to_bits()).collect();
    assert_eq!(bits, expected);
}

#[test]
fn complex_bit_patterns_survive() {
    let values = vec![
        Complex64::new(f64::from_bits(0x7ff8_dead_beef_0001), -0.0),
        Complex64::new(f64::from_bits(1), f64::NEG_INFINITY),
    ];
    let (_dir, _file, group) = fixture();
    let tensor = TypedTensor::<Complex64, DynRank, Host>::from_host_vec_col_major(
        vec![values.len()],
        values.clone(),
    )
    .unwrap();
    write_typed(&group, "a", tensor.as_view()).unwrap();
    let back: TypedTensor<Complex64, DynRank, Host> = read_typed(&group, "a").unwrap();
    for (found, expected) in back.as_slice().iter().zip(values.iter()) {
        assert_eq!(found.re.to_bits(), expected.re.to_bits());
        assert_eq!(found.im.to_bits(), expected.im.to_bits());
    }
}

#[test]
fn static_rank_round_trips() {
    let (_dir, _file, group) = fixture();
    let tensor =
        TypedTensor::<f32, Rank<2>, Host>::from_host_vec_col_major([2, 3], vec![1.0; 6]).unwrap();
    write_typed(&group, "a", tensor.as_view()).unwrap();
    let back: TypedTensor<f32, Rank<2>, Host> = read_typed(&group, "a").unwrap();
    assert_eq!(back.shape(), &[2, 3]);
    assert_eq!(back.as_slice(), tensor.as_slice());
    let flat: TypedTensor<f32, DynRank, Host> = read_typed(&group, "a").unwrap();
    assert_eq!(flat.shape(), &[2, 3]);
}

#[test]
fn zero_sized_axis_writes_no_payload() {
    let (_dir, _file, group) = fixture();
    let tensor =
        TypedTensor::<i64, DynRank, Host>::from_host_vec_col_major(vec![3, 0, 2], vec![]).unwrap();
    write_typed(&group, "a", tensor.as_view()).unwrap();
    assert_eq!(stored_extents(&group, "a"), vec![2, 0, 3]);
    let back: TypedTensor<i64, DynRank, Host> = read_typed(&group, "a").unwrap();
    assert_eq!(back.shape(), &[3, 0, 2]);
    assert!(back.as_slice().is_empty());
}

#[test]
fn dynamic_representations_round_trip() {
    let (_dir, _file, group) = fixture();
    let host =
        TypedTensor::<i32, DynRank, Host>::from_host_vec_col_major(vec![2, 2], vec![1, 2, 3, 4])
            .unwrap();

    let plain: TypedTensor<i32, DynRank, Dynamic> =
        TypedTensor::from_vec_col_major(vec![2, 2], vec![5, 6, 7, 8]).unwrap();
    write_typed(&group, "plain", plain.as_view()).unwrap();
    let back: TypedTensor<i32, DynRank, Host> = read_typed(&group, "plain").unwrap();
    assert_eq!(back.as_slice(), &[5, 6, 7, 8]);

    let promoted = host.promote().unwrap().into_dynamic();
    write_typed(&group, "promoted", promoted.as_view()).unwrap();
    let back: TypedTensor<i32, DynRank, Host> = read_typed(&group, "promoted").unwrap();
    assert_eq!(back.as_slice(), &[1, 2, 3, 4]);
}

#[test]
fn compact_offset_view_writes_only_its_region() {
    let (_dir, _file, group) = fixture();
    let storage: Vec<f64> = (0..10).map(f64::from).collect();
    let view: TypedTensorView<'_, f64, DynRank, Dynamic> =
        TypedTensorView::from_slice(vec![3], vec![1], 2, &storage).unwrap();
    write_typed(&group, "a", view).unwrap();
    let back: TypedTensor<f64, DynRank, Host> = read_typed(&group, "a").unwrap();
    assert_eq!(back.as_slice(), &[2.0, 3.0, 4.0]);
}

#[test]
fn dtype_erased_dispatch_round_trips_every_dtype() {
    let (_dir, _file, group) = fixture();
    let cases: Vec<Tensor> = vec![
        Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap(),
        Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap(),
        Tensor::from_vec_col_major(vec![2], vec![-1_i32, 2]).unwrap(),
        Tensor::from_vec_col_major(vec![2], vec![-1_i64, 2]).unwrap(),
        Tensor::from_vec_col_major(vec![2], vec![true, false]).unwrap(),
        Tensor::from_vec_col_major(vec![2], vec![Complex32::new(1.0, 2.0); 2]).unwrap(),
        Tensor::from_vec_col_major(vec![2], vec![Complex64::new(1.0, 2.0); 2]).unwrap(),
    ];
    for (index, tensor) in cases.iter().enumerate() {
        let name = format!("t{index}");
        write_tensor(&group, &name, tensor).unwrap();
        let back = read_tensor(&group, &name).unwrap();
        assert_eq!(back.dtype(), tensor.dtype());
        assert_eq!(back.shape(), tensor.shape());
        // The erased path must return the stored values, not just the metadata.
        macro_rules! same_payload {
            ($element:ty) => {
                assert_eq!(
                    back.as_slice::<$element>().unwrap(),
                    tensor.as_slice::<$element>().unwrap(),
                    "{:?}",
                    tensor.dtype()
                )
            };
        }
        match tensor.dtype() {
            DType::F32 => same_payload!(f32),
            DType::F64 => same_payload!(f64),
            DType::I32 => same_payload!(i32),
            DType::I64 => same_payload!(i64),
            DType::Bool => same_payload!(bool),
            DType::C32 => same_payload!(Complex32),
            DType::C64 => same_payload!(Complex64),
            DType::External(_) => unreachable!("no external dtype in this case list"),
        }
        let info = read_info(&group, &name).unwrap();
        assert_eq!(info.dtype, tensor.dtype());
    }
}

#[test]
fn read_options_limit_elements() {
    let (_dir, _file, group) = fixture();
    let tensor =
        TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![8], vec![0.0; 8]).unwrap();
    write_typed(&group, "a", tensor.as_view()).unwrap();

    let limited: Result<TypedTensor<f64, DynRank, Host>, _> =
        read_typed_with(&group, "a", &ReadOptions::default().max_elements(4));
    assert!(matches!(
        limited,
        Err(tenferro_hdf5::Error::ElementLimit {
            elements: 8,
            limit: 4
        })
    ));

    let generous: Result<TypedTensor<f64, DynRank, Host>, _> =
        read_typed_with(&group, "a", &ReadOptions::default().max_elements(8));
    assert_eq!(generous.unwrap().shape(), &[8]);
}
