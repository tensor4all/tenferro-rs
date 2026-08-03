use std::any::Any;
use std::fmt;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use num_complex::{Complex32, Complex64};
use tenferro_cpu::{with_cpu_exec_session, CpuBackend, CpuExecSession};
use tenferro_tensor::{
    AllocationDomainId, AllocationId, BackendSessionHost, BackendStorage, DType, HostAccessError,
    HostReadGuard, HostWriteGuard, MemoryKind, Placement, SharedTensorAllocationDomain,
    StorageBuffer, Tensor, TypedTensor,
};

use crate::{FftNorm, TensorFftExt};

#[derive(Debug, Default)]
struct AccessCounts {
    reads: AtomicUsize,
    writes: AtomicUsize,
}

struct FakeManagedBuffer<T> {
    values: Mutex<Vec<T>>,
    domain: AllocationDomainId,
    allocation: AllocationId,
    counts: Arc<AccessCounts>,
}

impl<T> fmt::Debug for FakeManagedBuffer<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FakeManagedBuffer")
            .field("domain", &self.domain)
            .field("allocation", &self.allocation)
            .finish_non_exhaustive()
    }
}

impl<T: Copy + Send + Sync + 'static> BackendStorage<T> for FakeManagedBuffer<T> {
    fn backend_family(&self) -> &'static str {
        "fake-managed"
    }

    fn len(&self) -> usize {
        self.values.lock().map_or(0, |values| values.len())
    }

    fn allocation_domain(&self) -> Option<AllocationDomainId> {
        Some(self.domain)
    }

    fn allocation_id(&self) -> Option<AllocationId> {
        Some(self.allocation)
    }

    fn map_read(&self) -> Result<HostReadGuard<'_, T>, HostAccessError> {
        let guard = self
            .values
            .lock()
            .map_err(|_| HostAccessError::BackendFailure {
                message: "fake read lock poisoned".to_string(),
            })?;
        self.counts.reads.fetch_add(1, Ordering::Relaxed);
        Ok(HostReadGuard::new(guard))
    }

    fn map_write(&self) -> Result<HostWriteGuard<'_, T>, HostAccessError> {
        let mut guard = self
            .values
            .lock()
            .map_err(|_| HostAccessError::BackendFailure {
                message: "fake write lock poisoned".to_string(),
            })?;
        self.counts.writes.fetch_add(1, Ordering::Relaxed);
        Ok(HostWriteGuard::new(guard.len(), move |source: &[T]| {
            guard.copy_from_slice(source);
            Ok(())
        }))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug)]
struct FakeDomain {
    id: AllocationDomainId,
    next_allocation: AtomicU64,
    counts: Arc<AccessCounts>,
}

impl FakeDomain {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            id: AllocationDomainId::fresh(),
            next_allocation: AtomicU64::new(1),
            counts: Arc::new(AccessCounts::default()),
        })
    }

    fn tensor<T: Copy + Send + Sync + 'static>(
        &self,
        shape: &[usize],
        values: Vec<T>,
    ) -> TypedTensor<T> {
        let buffer = FakeManagedBuffer {
            values: Mutex::new(values),
            domain: self.id,
            allocation: AllocationId::from_backend_id(
                self.next_allocation.fetch_add(1, Ordering::Relaxed),
            ),
            counts: Arc::clone(&self.counts),
        };
        TypedTensor::from_buffer_col_major(
            shape.to_vec(),
            StorageBuffer::Backend(Arc::new(buffer)),
            Placement {
                memory_kind: MemoryKind::Managed,
                device: None,
                cpu_affinity: None,
            },
        )
        .unwrap()
    }

    fn element_count(shape: &[usize]) -> tenferro_tensor::Result<usize> {
        shape.iter().try_fold(1_usize, |count, &dim| {
            count.checked_mul(dim).ok_or_else(|| {
                tenferro_tensor::Error::invalid_argument(
                    "FakeDomain::allocate",
                    "shape",
                    "element count overflow",
                )
            })
        })
    }
}

impl SharedTensorAllocationDomain for FakeDomain {
    fn id(&self) -> AllocationDomainId {
        self.id
    }

    fn allocate(&self, dtype: DType, shape: &[usize]) -> tenferro_tensor::Result<Tensor> {
        let len = Self::element_count(shape)?;
        Ok(match dtype {
            DType::F32 => Tensor::F32(self.tensor(shape, vec![0.0_f32; len])),
            DType::F64 => Tensor::F64(self.tensor(shape, vec![0.0_f64; len])),
            DType::C32 => Tensor::C32(self.tensor(shape, vec![Complex32::default(); len])),
            DType::C64 => Tensor::C64(self.tensor(shape, vec![Complex64::default(); len])),
            other => {
                return Err(tenferro_tensor::Error::unsupported_dtype(
                    "fft",
                    other,
                    "fake domain supports FFT floating and complex outputs",
                ));
            }
        })
    }
}

fn backend(domain: &Arc<FakeDomain>) -> CpuBackend {
    let erased: Arc<dyn SharedTensorAllocationDomain> = domain.clone();
    CpuBackend::new().with_allocation_domain(erased)
}

fn with_cpu_session<R>(
    backend: &mut CpuBackend,
    f: impl for<'a> FnOnce(&'a mut CpuExecSession<'a>) -> R + Send,
) -> R
where
    R: Send,
{
    backend.with_backend_session(|session| {
        with_cpu_exec_session(session, f).expect("CpuBackend must expose a CPU execution session")
    })
}

fn assert_managed_output(output: &Tensor, domain: &FakeDomain, input_id: Option<AllocationId>) {
    let (output_domain, output_id) = match output {
        Tensor::F32(output) => (output.allocation_domain(), output.allocation_id()),
        Tensor::F64(output) => (output.allocation_domain(), output.allocation_id()),
        Tensor::C32(output) => (output.allocation_domain(), output.allocation_id()),
        Tensor::C64(output) => (output.allocation_domain(), output.allocation_id()),
        other => panic!("unexpected managed FFT output dtype {:?}", other.dtype()),
    };
    assert_eq!(output_domain, Some(domain.id));
    assert_ne!(output_id, input_id);
    assert_eq!(output.placement().memory_kind, MemoryKind::Managed);
}

#[test]
fn managed_cpu_fft_covers_all_supported_scalar_and_operation_paths() {
    let domain = FakeDomain::new();
    let mut backend = backend(&domain);

    with_cpu_session(&mut backend, |backend| {
        let real32 = domain.tensor(&[4], vec![1.0_f32, 2.0, 3.0, 4.0]);
        let real32_id = real32.allocation_id();
        let full32 = Tensor::F32(real32)
            .fft(None, 0, FftNorm::Backward, backend)
            .unwrap();
        assert_managed_output(&full32, &domain, real32_id);
        let real32 = domain.tensor(&[4], vec![1.0_f32, 2.0, 3.0, 4.0]);
        let one_sided32 = Tensor::F32(real32)
            .rfft(None, 0, FftNorm::Backward, backend)
            .unwrap();
        let Tensor::C32(one_sided32) = one_sided32 else {
            unreachable!()
        };
        let recovered32 = Tensor::C32(one_sided32)
            .irfft(Some(4), 0, FftNorm::Backward, backend)
            .unwrap();
        assert_managed_output(&recovered32, &domain, None);

        let real64 = domain.tensor(&[4], vec![1.0_f64, 2.0, 3.0, 4.0]);
        let real64_id = real64.allocation_id();
        let one_sided64 = Tensor::F64(real64)
            .rfft(None, 0, FftNorm::Backward, backend)
            .unwrap();
        assert_managed_output(&one_sided64, &domain, real64_id);
        let Tensor::C64(one_sided64) = one_sided64 else {
            unreachable!()
        };
        let recovered64 = Tensor::C64(one_sided64)
            .irfft(Some(4), 0, FftNorm::Backward, backend)
            .unwrap();
        assert_managed_output(&recovered64, &domain, None);

        let complex32 = domain.tensor(&[4], vec![Complex32::new(1.0, 0.0); 4]);
        let transformed32 = Tensor::C32(complex32)
            .fft(None, 0, FftNorm::Backward, backend)
            .unwrap();
        let inverted32 = transformed32
            .ifft(None, 0, FftNorm::Backward, backend)
            .unwrap();
        assert_managed_output(&inverted32, &domain, None);

        let complex64 = domain.tensor(&[4], vec![Complex64::new(1.0, 0.0); 4]);
        let transformed64 = Tensor::C64(complex64)
            .fft(None, 0, FftNorm::Backward, backend)
            .unwrap();
        let inverted64 = transformed64
            .ifft(None, 0, FftNorm::Backward, backend)
            .unwrap();
        assert_managed_output(&inverted64, &domain, None);
    });

    assert_eq!(domain.counts.reads.load(Ordering::Relaxed), 9);
    assert_eq!(domain.counts.writes.load(Ordering::Relaxed), 9);
}
