use std::ptr;

use num_complex::Complex64;
use tenferro_algebra::Standard;
use tenferro_device::{Generator, LogicalMemorySpace};
use tenferro_tensor::MemoryOrder;

use super::*;
use crate::{
    ComplexRealPrimsDescriptor, ComplexRealUnaryOp, ComplexScalePrimsDescriptor,
    MetadataCastPrimsDescriptor, MetadataConstantValue, MetadataDType, MetadataGenerateOp,
    MetadataPrimsDescriptor, MetadataScalarTensorRef, MetadataTensorMut, MetadataTensorRef,
    RngPrimsDescriptor, SemiringBinaryOp, TensorComplexRealPrims, TensorComplexScalePrims,
    TensorMetadataCastPrims, TensorMetadataPrims, TensorResolveConjContextFor, TensorRngPrims,
};

fn dummy_real_tensor() -> Tensor<f64> {
    Tensor::from_slice(&[2.0, 3.0], &[2], MemoryOrder::ColumnMajor).unwrap()
}

fn dummy_complex_tensor() -> Tensor<Complex64> {
    Tensor::from_slice(
        &[Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap()
}

#[cfg(not(feature = "cuda"))]
fn assert_send<T: Send>() {}

#[cfg(not(feature = "cuda"))]
#[test]
fn cuda_backend_is_send() {
    assert_send::<CudaBackend>();
}

#[cfg(not(feature = "cuda"))]
#[test]
fn rocm_backend_is_send() {
    assert_send::<RocmBackend>();
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cuda_backend_drop_clears_handle() {
    use std::mem;

    let backend = CudaBackend {
        _handle: 0x1234 as *mut c_void,
        _lib: unsafe { libloading::Library::new("libm.so.6").unwrap() },
    };

    let handle_ptr = &backend._handle as *const *mut c_void as usize;
    let handle_before = unsafe { *(handle_ptr as *const *mut c_void) };
    assert!(!handle_before.is_null());

    mem::drop(backend);
}

#[test]
fn rocm_backend_drop_clears_handle() {
    use std::mem;

    let backend = RocmBackend {
        _handle: 0x1234 as *mut c_void,
        _lib: unsafe { libloading::Library::new("libm.so.6").unwrap() },
    };

    let handle_ptr = &backend._handle as *const *mut c_void as usize;
    let handle_before = unsafe { *(handle_ptr as *const *mut c_void) };
    assert!(!handle_before.is_null());

    mem::drop(backend);
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cuda_stub_reports_errors_and_resolves_conj() {
    let mut ctx = CudaContext::default();
    let plan = CudaPlan::<f64> {
        _handle: ptr::null_mut(),
        _workspace_size: 0,
        _marker: PhantomData,
    };
    let input = Tensor::<f64>::ones(
        &[1],
        tenferro_device::LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let mut output = Tensor::<f64>::zeros(
        &[1],
        tenferro_device::LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let desc = SemiringCoreDescriptor::MakeContiguous;

    let plan_result =
        <CudaBackend as TensorSemiringCore<Standard<f64>>>::plan(&mut ctx, &desc, &[&[1], &[1]]);
    assert!(matches!(plan_result, Err(Error::DeviceError(_))));
    let fast_plan_result = <CudaBackend as TensorSemiringFastPath<Standard<f64>>>::plan(
        &mut ctx,
        &SemiringFastPathDescriptor::ElementwiseBinary {
            op: SemiringBinaryOp::Mul,
        },
        &[&[1], &[1], &[1]],
    );
    assert!(matches!(fast_plan_result, Err(Error::DeviceError(_))));

    let exec_result = <CudaBackend as TensorSemiringCore<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        1.0,
        &[&input],
        0.0,
        &mut output,
    );
    assert!(matches!(exec_result, Err(Error::DeviceError(_))));
    let fast_exec_result = <CudaBackend as TensorSemiringFastPath<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        1.0,
        &[&input, &input],
        0.0,
        &mut output,
    );
    assert!(matches!(fast_exec_result, Err(Error::DeviceError(_))));
    assert!(
        !<CudaBackend as TensorSemiringFastPath<Standard<f64>>>::has_fast_path(
            SemiringFastPathDescriptor::Contract {
                modes_a: vec![0],
                modes_b: vec![0],
                modes_c: vec![0],
            }
        )
    );

    let complex = Tensor::<Complex64>::from_slice(
        &[Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let complex_conj = complex.conj();
    let resolved = CudaBackend::resolve_conj(&mut ctx, &complex_conj);
    assert!(!resolved.is_conjugated());

    let data = resolved.buffer().as_slice().unwrap();
    assert_eq!(data[0], Complex64::new(1.0, -2.0));
    assert_eq!(data[1], Complex64::new(-3.0, -4.0));

    let passthrough = CudaBackend::resolve_conj(&mut ctx, &complex);
    assert_eq!(
        passthrough.buffer().as_slice().unwrap(),
        complex.buffer().as_slice().unwrap()
    );
}

#[test]
fn rocm_stub_reports_errors_and_resolves_conj() {
    let mut ctx = RocmContext::default();
    let plan = RocmPlan::<f64> {
        _handle: ptr::null_mut(),
        _workspace_size: 0,
        _marker: PhantomData,
    };
    let input = Tensor::<f64>::ones(
        &[1],
        tenferro_device::LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let mut output = Tensor::<f64>::zeros(
        &[1],
        tenferro_device::LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let desc = SemiringCoreDescriptor::MakeContiguous;

    let plan_result =
        <RocmBackend as TensorSemiringCore<Standard<f64>>>::plan(&mut ctx, &desc, &[&[1], &[1]]);
    assert!(matches!(plan_result, Err(Error::DeviceError(_))));
    let fast_plan_result = <RocmBackend as TensorSemiringFastPath<Standard<f64>>>::plan(
        &mut ctx,
        &SemiringFastPathDescriptor::ElementwiseBinary {
            op: SemiringBinaryOp::Mul,
        },
        &[&[1], &[1], &[1]],
    );
    assert!(matches!(fast_plan_result, Err(Error::DeviceError(_))));

    let exec_result = <RocmBackend as TensorSemiringCore<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        1.0,
        &[&input],
        0.0,
        &mut output,
    );
    assert!(matches!(exec_result, Err(Error::DeviceError(_))));
    let fast_exec_result = <RocmBackend as TensorSemiringFastPath<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        1.0,
        &[&input, &input],
        0.0,
        &mut output,
    );
    assert!(matches!(fast_exec_result, Err(Error::DeviceError(_))));
    assert!(
        !<RocmBackend as TensorSemiringFastPath<Standard<f64>>>::has_fast_path(
            SemiringFastPathDescriptor::ElementwiseBinary {
                op: SemiringBinaryOp::Mul,
            }
        )
    );

    let complex = Tensor::<Complex64>::from_slice(
        &[Complex64::new(2.0, -1.0), Complex64::new(0.5, 3.0)],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let complex_conj = complex.conj();
    let resolved = RocmBackend::resolve_conj(&mut ctx, &complex_conj);
    assert!(!resolved.is_conjugated());

    let data = resolved.buffer().as_slice().unwrap();
    assert_eq!(data[0], Complex64::new(2.0, 1.0));
    assert_eq!(data[1], Complex64::new(0.5, -3.0));

    let passthrough = RocmBackend::resolve_conj(&mut ctx, &complex);
    assert_eq!(
        passthrough.buffer().as_slice().unwrap(),
        complex.buffer().as_slice().unwrap()
    );
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cuda_stub_context_metadata_and_family_protocols_reject_execution() {
    let mut ctx = CudaContext::new();
    assert_eq!(ctx.device_id(), 0);
    assert!(matches!(ctx.bind_to_device(), Err(Error::DeviceError(_))));

    let metadata_desc = MetadataPrimsDescriptor::Generate {
        op: MetadataGenerateOp::Constant(MetadataConstantValue::I32(3)),
        output_dtype: MetadataDType::I32,
    };
    let mut metadata_out = Tensor::<i32>::zeros(
        &[2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let err = <CudaBackend as TensorMetadataPrims>::plan(
        &mut ctx,
        &metadata_desc,
        &[],
        MetadataTensorMut::I32(&mut metadata_out),
    )
    .unwrap_err();
    assert!(err.to_string().contains("stub CudaBackend"));
    let err = <CudaBackend as TensorMetadataPrims>::execute(
        &mut ctx,
        &metadata_desc,
        &[],
        MetadataTensorMut::I32(&mut metadata_out),
    )
    .unwrap_err();
    assert!(err.to_string().contains("stub CudaBackend"));
    assert!(!<CudaBackend as TensorMetadataPrims>::has_metadata_support(
        metadata_desc.clone()
    ));

    let complex = dummy_complex_tensor();
    let resolved = <CudaContext as TensorResolveConjContextFor<Complex64>>::resolve_conj(
        &mut ctx,
        &complex.conj(),
    );
    assert!(!resolved.is_conjugated());

    let mut real_output = Tensor::<f64>::zeros(
        &[2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let complex_real_desc = ComplexRealPrimsDescriptor::PointwiseUnary {
        op: ComplexRealUnaryOp::Abs,
    };
    let err = <CudaBackend as TensorComplexRealPrims<Complex64>>::plan(
        &mut ctx,
        &complex_real_desc,
        &[complex.dims(), real_output.dims()],
    )
    .unwrap_err();
    assert!(err.to_string().contains("CudaBackend"));
    let err = <CudaBackend as TensorComplexRealPrims<Complex64>>::execute(
        &mut ctx,
        &(),
        1.0,
        &[&complex],
        0.0,
        &mut real_output,
    )
    .unwrap_err();
    assert!(err.to_string().contains("CudaBackend"));
    assert!(
        !<CudaBackend as TensorComplexRealPrims<Complex64>>::has_complex_real_support(
            complex_real_desc.clone()
        )
    );

    let real = dummy_real_tensor();
    let mut complex_output = Tensor::<Complex64>::zeros(
        &[2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let err = <CudaBackend as TensorComplexScalePrims<Complex64>>::plan(
        &mut ctx,
        &ComplexScalePrimsDescriptor::PointwiseMul,
        &[complex.dims(), real.dims(), complex_output.dims()],
    )
    .unwrap_err();
    assert!(err.to_string().contains("CudaBackend"));
    let err = <CudaBackend as TensorComplexScalePrims<Complex64>>::execute(
        &mut ctx,
        &(),
        Complex64::new(1.0, 0.0),
        &complex,
        &real,
        Complex64::new(0.0, 0.0),
        &mut complex_output,
    )
    .unwrap_err();
    assert!(err.to_string().contains("CudaBackend"));
    assert!(
        !<CudaBackend as TensorComplexScalePrims<Complex64>>::has_complex_scale_support(
            ComplexScalePrimsDescriptor::PointwiseMul
        )
    );

    let rng_desc = RngPrimsDescriptor::Uniform;
    let mut generator = Generator::cpu(7);
    let mut rng_output = Tensor::<f64>::zeros(
        &[2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let err = <CudaBackend as TensorRngPrims<Standard<f64>>>::plan(&mut ctx, &rng_desc, &[&[2]])
        .unwrap_err();
    assert!(err.to_string().contains("CudaBackend"));
    let err = <CudaBackend as TensorRngPrims<Standard<f64>>>::execute(
        &mut ctx,
        &(rng_desc.clone(), vec![2]),
        &mut generator,
        &mut rng_output,
    )
    .unwrap_err();
    assert!(err.to_string().contains("CudaBackend"));
    assert!(!<CudaBackend as TensorRngPrims<Standard<f64>>>::has_rng_support(rng_desc));

    let int_desc = RngPrimsDescriptor::Integer { low: -1, high: 3 };
    let err = <CudaBackend as TensorRngPrims<Standard<i32>>>::plan(&mut ctx, &int_desc, &[&[2]])
        .unwrap_err();
    assert!(err.to_string().contains("CudaBackend"));
    assert!(!<CudaBackend as TensorRngPrims<Standard<i32>>>::has_rng_support(int_desc));
}

#[test]
fn rocm_stub_context_metadata_and_family_protocols_reject_execution() {
    let mut ctx = RocmContext::new();

    let metadata_desc = MetadataPrimsDescriptor::Generate {
        op: MetadataGenerateOp::Constant(MetadataConstantValue::Bool(true)),
        output_dtype: MetadataDType::Bool,
    };
    let metadata_input = Tensor::<u8>::zeros(
        &[2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let mut metadata_out = Tensor::<u8>::zeros(
        &[2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let err = <RocmBackend as TensorMetadataPrims>::plan(
        &mut ctx,
        &metadata_desc,
        &[],
        MetadataTensorMut::Bool(&mut metadata_out),
    )
    .unwrap_err();
    assert!(err.to_string().contains("RocmBackend"));
    let err = <RocmBackend as TensorMetadataPrims>::execute(
        &mut ctx,
        &metadata_desc,
        &[MetadataTensorRef::Bool(&metadata_input)],
        MetadataTensorMut::Bool(&mut metadata_out),
    )
    .unwrap_err();
    assert!(err.to_string().contains("RocmBackend"));
    assert!(!<RocmBackend as TensorMetadataPrims>::has_metadata_support(
        metadata_desc.clone()
    ));

    let complex = dummy_complex_tensor();
    let resolved = <RocmContext as TensorResolveConjContextFor<Complex64>>::resolve_conj(
        &mut ctx,
        &complex.conj(),
    );
    assert!(!resolved.is_conjugated());

    let mut real_output = Tensor::<f64>::zeros(
        &[2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let complex_real_desc = ComplexRealPrimsDescriptor::PointwiseUnary {
        op: ComplexRealUnaryOp::Abs,
    };
    let err = <RocmBackend as TensorComplexRealPrims<Complex64>>::plan(
        &mut ctx,
        &complex_real_desc,
        &[complex.dims(), real_output.dims()],
    )
    .unwrap_err();
    assert!(err.to_string().contains("RocmBackend"));
    let err = <RocmBackend as TensorComplexRealPrims<Complex64>>::execute(
        &mut ctx,
        &(),
        1.0,
        &[&complex],
        0.0,
        &mut real_output,
    )
    .unwrap_err();
    assert!(err.to_string().contains("RocmBackend"));
    assert!(
        !<RocmBackend as TensorComplexRealPrims<Complex64>>::has_complex_real_support(
            complex_real_desc.clone()
        )
    );

    let real = dummy_real_tensor();
    let mut complex_output = Tensor::<Complex64>::zeros(
        &[2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let err = <RocmBackend as TensorComplexScalePrims<Complex64>>::plan(
        &mut ctx,
        &ComplexScalePrimsDescriptor::PointwiseMul,
        &[complex.dims(), real.dims(), complex_output.dims()],
    )
    .unwrap_err();
    assert!(err.to_string().contains("RocmBackend"));
    let err = <RocmBackend as TensorComplexScalePrims<Complex64>>::execute(
        &mut ctx,
        &(),
        Complex64::new(1.0, 0.0),
        &complex,
        &real,
        Complex64::new(0.0, 0.0),
        &mut complex_output,
    )
    .unwrap_err();
    assert!(err.to_string().contains("RocmBackend"));
    assert!(
        !<RocmBackend as TensorComplexScalePrims<Complex64>>::has_complex_scale_support(
            ComplexScalePrimsDescriptor::PointwiseMul
        )
    );

    let rng_desc = RngPrimsDescriptor::Uniform;
    let mut generator = Generator::cpu(7);
    let mut rng_output = Tensor::<f64>::zeros(
        &[2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let err = <RocmBackend as TensorRngPrims<Standard<f64>>>::plan(&mut ctx, &rng_desc, &[&[2]])
        .unwrap_err();
    assert!(err.to_string().contains("RocmBackend"));
    let err = <RocmBackend as TensorRngPrims<Standard<f64>>>::execute(
        &mut ctx,
        &(rng_desc.clone(), vec![2]),
        &mut generator,
        &mut rng_output,
    )
    .unwrap_err();
    assert!(err.to_string().contains("RocmBackend"));
    assert!(!<RocmBackend as TensorRngPrims<Standard<f64>>>::has_rng_support(rng_desc));

    let int_desc = RngPrimsDescriptor::Integer { low: -1, high: 3 };
    let err = <RocmBackend as TensorRngPrims<Standard<i32>>>::plan(&mut ctx, &int_desc, &[&[2]])
        .unwrap_err();
    assert!(err.to_string().contains("RocmBackend"));
    assert!(!<RocmBackend as TensorRngPrims<Standard<i32>>>::has_rng_support(int_desc));
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cuda_metadata_cast_stub_rejects_pointwise_and_where_protocols() {
    let mut ctx = CudaContext::new();
    let mask = Tensor::<u8>::zeros(
        &[2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let scalar = Tensor::<f64>::ones(
        &[2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let mut output = Tensor::<f64>::zeros(
        &[2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let pointwise = MetadataCastPrimsDescriptor::PointwiseCast {
        input_dtype: MetadataDType::Bool,
    };
    let err =
        <CudaBackend as TensorMetadataCastPrims<f64>>::plan(&mut ctx, &pointwise, &[&[2], &[2]])
            .unwrap_err();
    assert!(err.to_string().contains("CudaBackend"));
    let err = <CudaBackend as TensorMetadataCastPrims<f64>>::execute(
        &mut ctx,
        &pointwise,
        1.0,
        &[MetadataScalarTensorRef::Metadata(MetadataTensorRef::Bool(
            &mask,
        ))],
        0.0,
        &mut output,
    )
    .unwrap_err();
    assert!(err.to_string().contains("CudaBackend"));
    assert!(
        !<CudaBackend as TensorMetadataCastPrims<f64>>::has_metadata_cast_support(
            pointwise.clone()
        )
    );

    let where_desc = MetadataCastPrimsDescriptor::Where {
        cond_dtype: MetadataDType::Bool,
    };
    let err = <CudaBackend as TensorMetadataCastPrims<f64>>::plan(
        &mut ctx,
        &where_desc,
        &[&[2], &[2], &[2], &[2]],
    )
    .unwrap_err();
    assert!(err.to_string().contains("CudaBackend"));
    let err = <CudaBackend as TensorMetadataCastPrims<f64>>::execute(
        &mut ctx,
        &where_desc,
        1.0,
        &[
            MetadataScalarTensorRef::Metadata(MetadataTensorRef::Bool(&mask)),
            MetadataScalarTensorRef::Scalar(&scalar),
            MetadataScalarTensorRef::Scalar(&scalar),
        ],
        0.0,
        &mut output,
    )
    .unwrap_err();
    assert!(err.to_string().contains("CudaBackend"));
    assert!(!<CudaBackend as TensorMetadataCastPrims<f64>>::has_metadata_cast_support(where_desc));
}

#[test]
fn rocm_metadata_cast_stub_rejects_pointwise_and_where_protocols() {
    let mut ctx = RocmContext::new();
    let mask = Tensor::<u8>::zeros(
        &[2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let scalar = Tensor::<f64>::ones(
        &[2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let mut output = Tensor::<f64>::zeros(
        &[2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let pointwise = MetadataCastPrimsDescriptor::PointwiseCast {
        input_dtype: MetadataDType::Bool,
    };
    let err =
        <RocmBackend as TensorMetadataCastPrims<f64>>::plan(&mut ctx, &pointwise, &[&[2], &[2]])
            .unwrap_err();
    assert!(err.to_string().contains("RocmBackend"));
    let err = <RocmBackend as TensorMetadataCastPrims<f64>>::execute(
        &mut ctx,
        &pointwise,
        1.0,
        &[MetadataScalarTensorRef::Metadata(MetadataTensorRef::Bool(
            &mask,
        ))],
        0.0,
        &mut output,
    )
    .unwrap_err();
    assert!(err.to_string().contains("RocmBackend"));
    assert!(
        !<RocmBackend as TensorMetadataCastPrims<f64>>::has_metadata_cast_support(
            pointwise.clone()
        )
    );

    let where_desc = MetadataCastPrimsDescriptor::Where {
        cond_dtype: MetadataDType::Bool,
    };
    let err = <RocmBackend as TensorMetadataCastPrims<f64>>::plan(
        &mut ctx,
        &where_desc,
        &[&[2], &[2], &[2], &[2]],
    )
    .unwrap_err();
    assert!(err.to_string().contains("RocmBackend"));
    let err = <RocmBackend as TensorMetadataCastPrims<f64>>::execute(
        &mut ctx,
        &where_desc,
        1.0,
        &[
            MetadataScalarTensorRef::Metadata(MetadataTensorRef::Bool(&mask)),
            MetadataScalarTensorRef::Scalar(&scalar),
            MetadataScalarTensorRef::Scalar(&scalar),
        ],
        0.0,
        &mut output,
    )
    .unwrap_err();
    assert!(err.to_string().contains("RocmBackend"));
    assert!(!<RocmBackend as TensorMetadataCastPrims<f64>>::has_metadata_cast_support(where_desc));
}
