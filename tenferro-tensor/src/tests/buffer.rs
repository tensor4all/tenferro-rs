use std::ptr::NonNull;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use tenferro_device::LogicalMemorySpace;

use super::DataBuffer;

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
