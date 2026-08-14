use std::any::Any;
use std::fmt;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{
    AllocationDomainId, AllocationId, BackendSessionHost, BackendStorage, DType, HostAccessError,
    HostReadGuard, HostWriteGuard, MemoryKind, Placement, SharedTensorAllocationDomain,
    StorageBuffer, Tensor, TensorRead, TensorScalar, TypedTensor,
};

use crate::{FftNorm, TensorFftExt, TensorReadFftExt};

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

    fn map_write(&mut self) -> Result<HostWriteGuard<'_, T>, HostAccessError> {
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

    fn tensor<T: TensorScalar + Copy + Send + Sync + 'static>(
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
            StorageBuffer::Backend(Box::new(buffer)),
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
fn domain_bound_backend_accepts_host_owned_fft() {
    let domain = FakeDomain::new();
    let host = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let mut backend = backend(&domain);

    let output = backend
        .with_backend_session(|session| host.rfft(None, 0, FftNorm::Backward, session))
        .unwrap();
    let Tensor::C64(output) = output else {
        panic!("expected C64 host FFT output")
    };

    assert_eq!(
        output.as_slice().unwrap(),
        &[
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
        ]
    );
    assert!(matches!(output.buffer(), StorageBuffer::Host(_)));
    assert_eq!(output.allocation_domain(), None);
    assert_eq!(output.placement().memory_kind, MemoryKind::UnpinnedHost);
    assert_eq!(domain.counts.reads.load(Ordering::Relaxed), 0);
    assert_eq!(domain.counts.writes.load(Ordering::Relaxed), 0);
}

#[test]
fn domain_bound_backend_accepts_host_read_fft() {
    let domain = FakeDomain::new();
    let host = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let input = TensorRead::from_tensor(&host);
    let mut backend = backend(&domain);

    let output = backend
        .with_backend_session(|session| input.rfft_read(None, 0, FftNorm::Backward, session))
        .unwrap();
    let Tensor::C64(output) = output else {
        panic!("expected C64 host FFT output")
    };

    assert_eq!(
        output.as_slice().unwrap(),
        &[
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
        ]
    );
    assert!(matches!(output.buffer(), StorageBuffer::Host(_)));
    assert_eq!(output.allocation_domain(), None);
    assert_eq!(output.placement().memory_kind, MemoryKind::UnpinnedHost);
    assert_eq!(domain.counts.reads.load(Ordering::Relaxed), 0);
    assert_eq!(domain.counts.writes.load(Ordering::Relaxed), 0);
}

#[test]
fn domain_bound_backend_rejects_foreign_fft_input() {
    let domain = FakeDomain::new();
    let foreign = FakeDomain::new();
    let input = Tensor::F64(foreign.tensor(&[4], vec![1.0_f64, 2.0, 3.0, 4.0]));
    let mut backend = backend(&domain);

    let error = backend
        .with_backend_session(|session| input.rfft(None, 0, FftNorm::Backward, session))
        .unwrap_err();

    assert!(matches!(
        error,
        tenferro_tensor::Error::HostAccess {
            source: HostAccessError::ForeignDomain { .. },
            ..
        }
    ));
}

#[test]
fn managed_cpu_fft_covers_all_supported_scalar_and_operation_paths() {
    let domain = FakeDomain::new();
    let mut backend = backend(&domain);

    backend.with_backend_session(|session| {
        let real32 = domain.tensor(&[4], vec![1.0_f32, 2.0, 3.0, 4.0]);
        let real32_id = real32.allocation_id();
        let full32 = Tensor::F32(real32)
            .fft(None, 0, FftNorm::Backward, session)
            .unwrap();
        assert_managed_output(&full32, &domain, real32_id);
        let real32 = domain.tensor(&[4], vec![1.0_f32, 2.0, 3.0, 4.0]);
        let one_sided32 = Tensor::F32(real32)
            .rfft(None, 0, FftNorm::Backward, session)
            .unwrap();
        let Tensor::C32(one_sided32) = one_sided32 else {
            unreachable!()
        };
        let recovered32 = Tensor::C32(one_sided32)
            .irfft(Some(4), 0, FftNorm::Backward, session)
            .unwrap();
        assert_managed_output(&recovered32, &domain, None);

        let real64 = domain.tensor(&[4], vec![1.0_f64, 2.0, 3.0, 4.0]);
        let real64_id = real64.allocation_id();
        let one_sided64 = Tensor::F64(real64)
            .rfft(None, 0, FftNorm::Backward, session)
            .unwrap();
        assert_managed_output(&one_sided64, &domain, real64_id);
        let Tensor::C64(one_sided64) = one_sided64 else {
            unreachable!()
        };
        let recovered64 = Tensor::C64(one_sided64)
            .irfft(Some(4), 0, FftNorm::Backward, session)
            .unwrap();
        assert_managed_output(&recovered64, &domain, None);

        let complex32 = domain.tensor(&[4], vec![Complex32::new(1.0, 0.0); 4]);
        let transformed32 = Tensor::C32(complex32)
            .fft(None, 0, FftNorm::Backward, session)
            .unwrap();
        let inverted32 = transformed32
            .ifft(None, 0, FftNorm::Backward, session)
            .unwrap();
        assert_managed_output(&inverted32, &domain, None);

        let complex64 = domain.tensor(&[4], vec![Complex64::new(1.0, 0.0); 4]);
        let transformed64 = Tensor::C64(complex64)
            .fft(None, 0, FftNorm::Backward, session)
            .unwrap();
        let inverted64 = transformed64
            .ifft(None, 0, FftNorm::Backward, session)
            .unwrap();
        assert_managed_output(&inverted64, &domain, None);
    });

    assert_eq!(domain.counts.reads.load(Ordering::Relaxed), 9);
    assert_eq!(domain.counts.writes.load(Ordering::Relaxed), 9);
}
