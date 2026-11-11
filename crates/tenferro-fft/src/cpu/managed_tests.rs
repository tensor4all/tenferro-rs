use crate::{FftOptions, FftnOptions};
use std::any::Any;
use std::fmt;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuEngine;
use tenferro_tensor::{
    AllocationDomainId, AllocationId, BackendStorage, DType, HostAccessError, HostReadGuard,
    HostWriteGuard, MemoryKind, Placement, SharedTensorAllocationDomain, StorageBuffer, Tensor,
    TensorRead, TensorScalar, TypedTensor,
};

use crate::{FftExecutor, FftOps};

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
            DType::F32 => Tensor::from_typed::<f32>(self.tensor(shape, vec![0.0_f32; len])),
            DType::F64 => Tensor::from_typed::<f64>(self.tensor(shape, vec![0.0_f64; len])),
            DType::C32 => Tensor::from_typed::<tenferro_tensor::Complex32>(
                self.tensor(shape, vec![Complex32::default(); len]),
            ),
            DType::C64 => Tensor::from_typed::<tenferro_tensor::Complex64>(
                self.tensor(shape, vec![Complex64::default(); len]),
            ),
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

#[derive(Clone, Copy, Debug)]
enum OutputFault {
    AllocationError,
    WrongDtype,
    ForeignDomain,
    WrongLength,
}

#[derive(Debug)]
struct FaultyOutputDomain {
    input_domain: Arc<FakeDomain>,
    fault: OutputFault,
}

impl SharedTensorAllocationDomain for FaultyOutputDomain {
    fn id(&self) -> AllocationDomainId {
        self.input_domain.id
    }

    fn allocate(&self, dtype: DType, shape: &[usize]) -> tenferro_tensor::Result<Tensor> {
        match self.fault {
            OutputFault::AllocationError => Err(tenferro_tensor::Error::runtime_state(
                "FaultyOutputDomain::allocate",
                "allocation unavailable",
            )),
            OutputFault::WrongDtype => self.input_domain.allocate(DType::F32, shape),
            OutputFault::ForeignDomain => FakeDomain::new().allocate(dtype, shape),
            OutputFault::WrongLength => self.input_domain.allocate(dtype, &[1]),
        }
    }
}

#[test]
fn managed_output_failures_return_errors_and_leave_input_usable() {
    for fault in [
        OutputFault::AllocationError,
        OutputFault::WrongDtype,
        OutputFault::ForeignDomain,
        OutputFault::WrongLength,
    ] {
        let domain = FakeDomain::new();
        let input = Tensor::from_typed::<f64>(domain.tensor(&[4], vec![1., 2., 3., 4.]));
        let mut failing =
            CpuEngine::serial().with_allocation_domain(Arc::new(FaultyOutputDomain {
                input_domain: domain.clone(),
                fault,
            }));
        let error = {
            let s: &mut dyn tenferro_tensor::Engine = &mut failing;
            input.rfft_with(&FftOptions::default().axis(0), s)
        }
        .unwrap_err();
        match fault {
            OutputFault::WrongLength => {
                assert!(matches!(error, tenferro_tensor::Error::HostAccess { .. }))
            }
            _ => assert!(matches!(error, tenferro_tensor::Error::RuntimeState { .. })),
        }
        let mut valid = backend(&domain);
        let result = {
            let s: &mut dyn tenferro_tensor::Engine = &mut valid;
            input.rfft_with(&FftOptions::default().axis(0), s)
        }
        .unwrap();
        let Some(result) = result.as_typed::<Complex64>() else {
            panic!("expected complex result")
        };
        let guard = result.backend_buffer().unwrap().map_read().unwrap();
        assert_eq!(
            &*guard,
            &[
                Complex64::new(10., 0.),
                Complex64::new(-2., 2.),
                Complex64::new(-2., 0.)
            ]
        );
    }
}

fn backend(domain: &Arc<FakeDomain>) -> CpuEngine {
    let erased: Arc<dyn SharedTensorAllocationDomain> = domain.clone();
    CpuEngine::serial().with_allocation_domain(erased)
}

fn assert_managed_output(output: &Tensor, domain: &FakeDomain, input_id: Option<AllocationId>) {
    let (output_domain, output_id) = match output.dtype() {
        DType::F32 => {
            let output = output.as_typed::<f32>().expect("managed FFT output");
            (output.allocation_domain(), output.allocation_id())
        }
        DType::F64 => {
            let output = output.as_typed::<f64>().expect("managed FFT output");
            (output.allocation_domain(), output.allocation_id())
        }
        DType::C32 => {
            let output = output
                .as_typed::<num_complex::Complex32>()
                .expect("managed FFT output");
            (output.allocation_domain(), output.allocation_id())
        }
        DType::C64 => {
            let output = output
                .as_typed::<num_complex::Complex64>()
                .expect("managed FFT output");
            (output.allocation_domain(), output.allocation_id())
        }
        other => panic!("unexpected managed FFT output dtype {:?}", other),
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

    let output = {
        let session: &mut dyn tenferro_tensor::Engine = &mut backend;
        host.rfft_with(&FftOptions::default().axis(0), session)
    }
    .unwrap();
    let Some(output) = output.as_typed::<Complex64>() else {
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

    let output = {
        let session: &mut dyn tenferro_tensor::Engine = &mut backend;
        input.rfft_with(&FftOptions::default().axis(0), session)
    }
    .unwrap();
    let Some(output) = output.as_typed::<Complex64>() else {
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
    let input = Tensor::from_typed::<f64>(foreign.tensor(&[4], vec![1.0_f64, 2.0, 3.0, 4.0]));
    let mut backend = backend(&domain);

    let error = {
        let session: &mut dyn tenferro_tensor::Engine = &mut backend;
        input.rfft_with(&FftOptions::default().axis(0), session)
    }
    .unwrap_err();

    assert!(matches!(
        error,
        tenferro_tensor::Error::HostAccess {
            source: HostAccessError::ForeignDomain { .. },
            ..
        }
    ));
}

/// A managed CPU tensor is legal input for the one-axis transform, so it must
/// stay legal for the n-dimensional entry point, including for its owned
/// per-axis intermediates.
#[test]
fn managed_cpu_fftn_matches_the_one_axis_transform() {
    let domain = FakeDomain::new();
    let mut backend = backend(&domain);
    let mut executor = FftExecutor::default();
    let values: Vec<Complex64> = (0..6)
        .map(|index| Complex64::new(index as f64, (index % 3) as f64))
        .collect();

    let input = Tensor::from_typed::<Complex64>(domain.tensor(&[2, 3], values));
    let n_dimensional = executor
        .fftn_with(
            &input,
            &FftnOptions::default().axes(vec![0, 1]),
            &mut backend,
        )
        .unwrap();
    assert_managed_output(&n_dimensional, &domain, None);

    let mut sequential = input;
    for axis in 0..2 {
        sequential = executor
            .fft_with(&sequential, &FftOptions::default().axis(axis), &mut backend)
            .unwrap();
    }

    let read = |tensor: &Tensor| {
        let typed = tensor.as_typed::<Complex64>().unwrap();
        let guard = typed.backend_buffer().unwrap().map_read().unwrap();
        guard.to_vec()
    };
    assert_eq!(read(&n_dimensional), read(&sequential));
}

#[test]
fn managed_cpu_fft_covers_all_supported_scalar_and_operation_paths() {
    let domain = FakeDomain::new();
    let mut backend = backend(&domain);

    {
        let session: &mut dyn tenferro_tensor::Engine = &mut backend;
        let real32 = domain.tensor(&[4], vec![1.0_f32, 2.0, 3.0, 4.0]);
        let real32_id = real32.allocation_id();
        let full32 = Tensor::from_typed::<f32>(real32)
            .fft_with(&FftOptions::default().axis(0), session)
            .unwrap();
        assert_managed_output(&full32, &domain, real32_id);
        let real32 = domain.tensor(&[4], vec![1.0_f32, 2.0, 3.0, 4.0]);
        let one_sided32 = Tensor::from_typed::<f32>(real32)
            .rfft_with(&FftOptions::default().axis(0), session)
            .unwrap();
        let Ok(one_sided32) = one_sided32.into_typed::<Complex32>() else {
            unreachable!()
        };
        let recovered32 = Tensor::from_typed::<tenferro_tensor::Complex32>(one_sided32)
            .irfft_with(&FftOptions::default().n(Some(4)).axis(0), session)
            .unwrap();
        assert_managed_output(&recovered32, &domain, None);

        let real64 = domain.tensor(&[4], vec![1.0_f64, 2.0, 3.0, 4.0]);
        let real64_id = real64.allocation_id();
        let one_sided64 = Tensor::from_typed::<f64>(real64)
            .rfft_with(&FftOptions::default().axis(0), session)
            .unwrap();
        assert_managed_output(&one_sided64, &domain, real64_id);
        let Ok(one_sided64) = one_sided64.into_typed::<Complex64>() else {
            unreachable!()
        };
        let recovered64 = Tensor::from_typed::<tenferro_tensor::Complex64>(one_sided64)
            .irfft_with(&FftOptions::default().n(Some(4)).axis(0), session)
            .unwrap();
        assert_managed_output(&recovered64, &domain, None);

        let complex32 = domain.tensor(&[4], vec![Complex32::new(1.0, 0.0); 4]);
        let transformed32 = Tensor::from_typed::<tenferro_tensor::Complex32>(complex32)
            .fft_with(&FftOptions::default().axis(0), session)
            .unwrap();
        let inverted32 = transformed32
            .ifft_with(&FftOptions::default().axis(0), session)
            .unwrap();
        assert_managed_output(&inverted32, &domain, None);

        let complex64 = domain.tensor(&[4], vec![Complex64::new(1.0, 0.0); 4]);
        let transformed64 = Tensor::from_typed::<tenferro_tensor::Complex64>(complex64)
            .fft_with(&FftOptions::default().axis(0), session)
            .unwrap();
        let inverted64 = transformed64
            .ifft_with(&FftOptions::default().axis(0), session)
            .unwrap();
        assert_managed_output(&inverted64, &domain, None);
    };

    assert_eq!(domain.counts.reads.load(Ordering::Relaxed), 9);
    assert_eq!(domain.counts.writes.load(Ordering::Relaxed), 9);
}
