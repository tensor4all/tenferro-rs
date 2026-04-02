use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tenferro_device::LogicalMemorySpace;

use super::DataBuffer;

#[test]
fn owned_buffer_paths_cover_uniqueness_mutation_and_vec_extraction() {
    let mut buffer = DataBuffer::from_vec(vec![1_i32, 2, 3]);
    assert!(buffer.is_unique());
    assert_eq!(buffer.as_mut_slice().unwrap(), &mut [1, 2, 3]);

    buffer.as_mut_slice().unwrap()[1] = 9;
    assert_eq!(buffer.as_slice().unwrap(), &[1, 9, 3]);

    let shared = buffer.clone();
    assert!(!buffer.is_unique());
    assert!(buffer.as_mut_slice().is_none());
    assert!(buffer.clone().try_into_vec().is_none());

    drop(shared);
    assert_eq!(buffer.try_into_vec(), Some(vec![1, 9, 3]));
}

#[test]
fn external_buffer_paths_cover_cpu_access_and_release_callback() {
    let released = Arc::new(AtomicBool::new(false));
    let data = vec![1.0_f64, 2.0, 3.0];
    let ptr = data.as_ptr();
    let len = data.len();

    let mut buffer = unsafe {
        DataBuffer::from_external(ptr, len, {
            let released = Arc::clone(&released);
            move || {
                released.store(true, Ordering::SeqCst);
                drop(data);
            }
        })
    };

    assert_eq!(buffer.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    assert_eq!(buffer.len(), 3);
    assert!(!buffer.is_owned());
    assert!(!buffer.is_gpu());
    assert!(buffer.as_mut_slice().is_none());
    assert!(buffer.as_ptr().is_some());
    assert!(buffer.try_into_vec().is_none());
    assert!(released.load(Ordering::SeqCst));
}

#[test]
fn gpu_buffer_paths_cover_queries_and_release_callback() {
    let released = Arc::new(AtomicBool::new(false));
    let device_ptr = NonNull::<f64>::dangling().as_ptr();
    let space = LogicalMemorySpace::GpuMemory { device_id: 0 };

    let mut buffer = unsafe {
        DataBuffer::from_gpu_parts(device_ptr, 4, space, {
            let released = Arc::clone(&released);
            move || {
                released.store(true, Ordering::SeqCst);
            }
        })
    };

    assert!(buffer.as_slice().is_none());
    assert!(buffer.as_mut_slice().is_none());
    assert_eq!(buffer.len(), 4);
    assert!(!buffer.is_owned());
    assert!(buffer.is_gpu());
    assert!(buffer.as_ptr().is_none());
    assert_eq!(buffer.as_device_ptr(), Some(device_ptr as *const f64));
    assert_eq!(buffer.gpu_memory_space(), Some(space));

    drop(buffer);
    assert!(released.load(Ordering::SeqCst));
}

#[test]
fn reinterpret_as_covers_owned_external_and_gpu_buffers() {
    let owned = DataBuffer::from_vec(vec![1_u32, 2, 3]);
    let owned_reinterpreted = owned.reinterpret_as::<u32>(3).unwrap();
    assert_eq!(owned_reinterpreted.as_slice().unwrap(), &[1, 2, 3]);
    assert_eq!(owned_reinterpreted.as_ptr(), owned.as_ptr());
    assert!(!owned_reinterpreted.is_owned());
    assert!(!owned_reinterpreted.is_gpu());

    let external_released = Arc::new(AtomicBool::new(false));
    let external_data = vec![4_u32, 5, 6];
    let external_ptr = external_data.as_ptr();
    let external = unsafe {
        DataBuffer::from_external(external_ptr, external_data.len(), {
            let external_released = Arc::clone(&external_released);
            move || {
                external_released.store(true, Ordering::SeqCst);
                drop(external_data);
            }
        })
    };
    let external_reinterpreted = external.reinterpret_as::<u32>(3).unwrap();
    assert_eq!(external_reinterpreted.as_slice().unwrap(), &[4, 5, 6]);
    assert_eq!(external_reinterpreted.as_ptr(), Some(external_ptr));

    drop(external_reinterpreted);
    assert!(!external_released.load(Ordering::SeqCst));
    drop(external);
    assert!(external_released.load(Ordering::SeqCst));

    let gpu_released = Arc::new(AtomicBool::new(false));
    let device_ptr = NonNull::<u32>::dangling().as_ptr();
    let space = LogicalMemorySpace::GpuMemory { device_id: 1 };
    let gpu = unsafe {
        DataBuffer::from_gpu_parts(device_ptr, 3, space, {
            let gpu_released = Arc::clone(&gpu_released);
            move || {
                gpu_released.store(true, Ordering::SeqCst);
            }
        })
    };
    let gpu_reinterpreted = gpu.reinterpret_as::<u32>(3).unwrap();
    assert!(gpu_reinterpreted.is_gpu());
    assert_eq!(
        gpu_reinterpreted.as_device_ptr(),
        Some(device_ptr as *const u32)
    );
    assert_eq!(gpu_reinterpreted.gpu_memory_space(), Some(space));

    drop(gpu_reinterpreted);
    assert!(!gpu_released.load(Ordering::SeqCst));
    drop(gpu);
    assert!(gpu_released.load(Ordering::SeqCst));
}

#[test]
fn reinterpret_as_rejects_oversized_and_misaligned_views() {
    let owned = DataBuffer::from_vec(vec![1_u32, 2]);
    let oversize = match owned.reinterpret_as::<u64>(2) {
        Ok(_) => panic!("oversized reinterpretation should fail"),
        Err(err) => err,
    };
    assert!(format!("{oversize}").contains("exceeds source byte size"));

    let misaligned_released = Arc::new(AtomicBool::new(false));
    let bytes = vec![0_u8; 4];
    let ptr = unsafe { bytes.as_ptr().add(1) };
    let misaligned = unsafe {
        DataBuffer::from_external(ptr, 3, {
            let misaligned_released = Arc::clone(&misaligned_released);
            move || {
                misaligned_released.store(true, Ordering::SeqCst);
                drop(bytes);
            }
        })
    };
    let err = match misaligned.reinterpret_as::<u16>(1) {
        Ok(_) => panic!("misaligned reinterpretation should fail"),
        Err(err) => err,
    };
    assert!(format!("{err}").contains("alignment"));
    drop(misaligned);
    assert!(misaligned_released.load(Ordering::SeqCst));
}
