use std::ptr;

use num_complex::Complex64;
use tenferro_tensor::MemoryOrder;

use super::*;

#[cfg(not(feature = "cuda"))]
#[test]
fn cuda_stub_reports_errors_and_resolves_conj() {
    let mut ctx = CudaContext {
        _stream: std::ptr::null_mut(),
        _workspace: Vec::new(),
        _plan_cache: PlanCache::new(),
    };
    let plan = CudaPlan::<f64> {
        _handle: ptr::null_mut(),
        _workspace_size: 0,
        _marker: PhantomData,
    };
    let input = Tensor::<f64>::ones(
        &[1],
        tenferro_device::LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );
    let mut output = Tensor::<f64>::zeros(
        &[1],
        tenferro_device::LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );
    let desc = PrimDescriptor::MakeContiguous;

    let plan_result =
        <CudaBackend as TensorPrims<Standard<f64>>>::plan(&mut ctx, &desc, &[&[1], &[1]]);
    assert!(matches!(plan_result, Err(Error::DeviceError(_))));

    let exec_result = <CudaBackend as TensorPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        1.0,
        &[&input],
        0.0,
        &mut output,
    );
    assert!(matches!(exec_result, Err(Error::DeviceError(_))));
    assert!(!<CudaBackend as TensorPrims<Standard<f64>>>::has_extension_for(Extension::Contract));

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
}

#[test]
fn rocm_stub_reports_errors_and_resolves_conj() {
    let mut ctx = RocmContext {
        _stream: std::ptr::null_mut(),
        _workspace: Vec::new(),
        _plan_cache: PlanCache::new(),
    };
    let plan = RocmPlan::<f64> {
        _handle: ptr::null_mut(),
        _workspace_size: 0,
        _marker: PhantomData,
    };
    let input = Tensor::<f64>::ones(
        &[1],
        tenferro_device::LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );
    let mut output = Tensor::<f64>::zeros(
        &[1],
        tenferro_device::LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );
    let desc = PrimDescriptor::MakeContiguous;

    let plan_result =
        <RocmBackend as TensorPrims<Standard<f64>>>::plan(&mut ctx, &desc, &[&[1], &[1]]);
    assert!(matches!(plan_result, Err(Error::DeviceError(_))));

    let exec_result = <RocmBackend as TensorPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        1.0,
        &[&input],
        0.0,
        &mut output,
    );
    assert!(matches!(exec_result, Err(Error::DeviceError(_))));
    assert!(
        !<RocmBackend as TensorPrims<Standard<f64>>>::has_extension_for(Extension::ElementwiseMul)
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
}
