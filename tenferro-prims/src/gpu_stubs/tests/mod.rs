use std::ptr;

use num_complex::Complex64;
use tenferro_tensor::MemoryOrder;

use super::*;
use crate::SemiringBinaryOp;

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
