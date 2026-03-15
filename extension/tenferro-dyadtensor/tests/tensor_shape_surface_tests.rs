use num_complex::{Complex32, Complex64};
use tenferro_dyadtensor::{StructuredTensor, Tensor};
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

fn vector(values: &[f64]) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn matrix2(values: &[f64; 4]) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(values, &[2, 2], MemoryOrder::ColumnMajor).unwrap()
}

fn matrix2_f32(values: &[f32; 4]) -> DenseTensor<f32> {
    DenseTensor::<f32>::from_slice(values, &[2, 2], MemoryOrder::ColumnMajor).unwrap()
}

fn matrix2_c32(values: &[Complex32; 4]) -> DenseTensor<Complex32> {
    DenseTensor::<Complex32>::from_slice(values, &[2, 2], MemoryOrder::ColumnMajor).unwrap()
}

fn matrix2_c64(values: &[Complex64; 4]) -> DenseTensor<Complex64> {
    DenseTensor::<Complex64>::from_slice(values, &[2, 2], MemoryOrder::ColumnMajor).unwrap()
}

fn vector_f32(values: &[f32]) -> DenseTensor<f32> {
    DenseTensor::<f32>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn vector_c32(values: &[Complex32]) -> DenseTensor<Complex32> {
    DenseTensor::<Complex32>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn vector_c64(values: &[Complex64]) -> DenseTensor<Complex64> {
    DenseTensor::<Complex64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn as_slice(tensor: &DenseTensor<f64>) -> &[f64] {
    tensor.buffer().as_slice().unwrap()
}

#[test]
fn tensor_shape_accessors_cover_dense_diag_and_empty_layouts() {
    let dense = Tensor::from_tensor(matrix2(&[1.0, 2.0, 3.0, 4.0]));
    assert_eq!(dense.dims(), &[2, 2]);
    assert_eq!(dense.axis_classes(), &[0, 1]);
    assert!(dense.is_dense());
    assert!(!dense.is_diag());
    assert_eq!(dense.ndim(), 2);
    assert_eq!(dense.len(), 4);
    assert!(!dense.is_empty());

    let empty = Tensor::from_slice::<f64>(&[], &[0]).unwrap();
    assert_eq!(empty.dims(), &[0]);
    assert_eq!(empty.ndim(), 1);
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());

    let diag = Tensor::from_structured(
        StructuredTensor::from_diagonal_vector(vector(&[3.0, 4.0]), 2).unwrap(),
    );
    assert_eq!(diag.dims(), &[2, 2]);
    assert_eq!(diag.axis_classes(), &[0, 0]);
    assert!(!diag.is_dense());
    assert!(diag.is_diag());
    assert_eq!(diag.ndim(), 2);
    assert_eq!(diag.len(), 4);
}

#[test]
fn tensor_shape_transform_methods_cover_dense_paths() {
    let matrix = Tensor::from_tensor(matrix2(&[1.0, 2.0, 3.0, 4.0]));

    let reshaped = matrix.reshape(&[4]).unwrap();
    assert_eq!(reshaped.dims(), &[4]);
    assert_eq!(
        as_slice(reshaped.as_f64().unwrap().primal()),
        &[1.0, 2.0, 3.0, 4.0]
    );

    let prefix = matrix.take_prefix(1, 1).unwrap();
    assert_eq!(prefix.dims(), &[2, 1]);
    assert_eq!(as_slice(prefix.as_f64().unwrap().primal()), &[1.0, 2.0]);

    let contiguous = matrix.contiguous(MemoryOrder::RowMajor).unwrap();
    assert_eq!(contiguous.dims(), &[2, 2]);
    assert_eq!(
        as_slice(contiguous.as_f64().unwrap().primal()),
        &[1.0, 3.0, 2.0, 4.0]
    );
}

#[test]
fn tensor_diag_embed_promotes_dense_vector_to_structured_diagonal() {
    let vector = Tensor::from_tensor(vector(&[5.0, -1.0]));
    let diag = vector.diag_embed(2).unwrap();

    assert!(diag.is_diag());
    assert_eq!(diag.dims(), &[2, 2]);
    assert_eq!(diag.axis_classes(), &[0, 0]);
}

#[test]
fn tensor_shape_methods_cover_non_f64_runtime_variants() {
    let f32_matrix = Tensor::from_tensor(matrix2_f32(&[1.0, 2.0, 3.0, 4.0]));
    assert_eq!(f32_matrix.axis_classes(), &[0, 1]);
    assert!(f32_matrix.is_dense());
    assert!(!f32_matrix.is_diag());
    let f32_reshaped = f32_matrix.reshape(&[4]).unwrap();
    assert_eq!(
        f32_reshaped.scalar_type(),
        tenferro_dyadtensor::ScalarType::F32
    );
    assert_eq!(f32_reshaped.dims(), &[4]);
    let f32_prefix = f32_matrix.take_prefix(1, 1).unwrap();
    assert_eq!(f32_prefix.dims(), &[2, 1]);
    let f32_contiguous = f32_matrix.contiguous(MemoryOrder::RowMajor).unwrap();
    assert_eq!(f32_contiguous.dims(), &[2, 2]);
    let f32_diag = Tensor::from_tensor(vector_f32(&[5.0, -1.0]))
        .diag_embed(2)
        .unwrap();
    assert_eq!(f32_diag.axis_classes(), &[0, 0]);
    assert!(f32_diag.is_diag());

    let c32_matrix = Tensor::from_tensor(matrix2_c32(&[
        Complex32::new(1.0, 0.0),
        Complex32::new(2.0, 0.5),
        Complex32::new(3.0, -1.0),
        Complex32::new(4.0, 0.25),
    ]));
    assert_eq!(c32_matrix.axis_classes(), &[0, 1]);
    assert!(c32_matrix.is_dense());
    assert!(!c32_matrix.is_diag());
    let c32_reshaped = c32_matrix.reshape(&[4]).unwrap();
    assert_eq!(c32_reshaped.dims(), &[4]);
    let c32_prefix = c32_matrix.take_prefix(1, 1).unwrap();
    assert_eq!(
        c32_prefix.scalar_type(),
        tenferro_dyadtensor::ScalarType::C32
    );
    assert_eq!(c32_prefix.dims(), &[2, 1]);
    let c32_contiguous = c32_matrix.contiguous(MemoryOrder::RowMajor).unwrap();
    assert_eq!(
        c32_contiguous.scalar_type(),
        tenferro_dyadtensor::ScalarType::C32
    );
    let c32_diag = Tensor::from_tensor(vector_c32(&[
        Complex32::new(5.0, -1.0),
        Complex32::new(-2.0, 0.5),
    ]))
    .diag_embed(2)
    .unwrap();
    assert_eq!(c32_diag.axis_classes(), &[0, 0]);
    assert!(c32_diag.is_diag());

    let c64_matrix = Tensor::from_tensor(matrix2_c64(&[
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.5),
        Complex64::new(3.0, -1.0),
        Complex64::new(4.0, 0.25),
    ]));
    assert_eq!(c64_matrix.axis_classes(), &[0, 1]);
    assert!(c64_matrix.is_dense());
    assert!(!c64_matrix.is_diag());
    let c64_reshaped = c64_matrix.reshape(&[4]).unwrap();
    assert_eq!(c64_reshaped.dims(), &[4]);
    let c64_prefix = c64_matrix.take_prefix(1, 1).unwrap();
    assert_eq!(c64_prefix.dims(), &[2, 1]);
    let c64_contiguous = c64_matrix.contiguous(MemoryOrder::RowMajor).unwrap();
    assert_eq!(c64_contiguous.dims(), &[2, 2]);

    let c64_diag = Tensor::from_tensor(vector_c64(&[
        Complex64::new(5.0, -1.0),
        Complex64::new(-2.0, 0.5),
    ]))
    .diag_embed(2)
    .unwrap();
    assert_eq!(c64_diag.scalar_type(), tenferro_dyadtensor::ScalarType::C64);
    assert!(c64_diag.is_diag());
    assert_eq!(c64_diag.axis_classes(), &[0, 0]);
}
