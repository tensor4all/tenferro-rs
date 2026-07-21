use std::any::Any;
use std::cell::Cell;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{
    AllocationDomainId, AllocationId, BackendBuffer, Buffer, DType, HostAccessError, HostReadGuard,
    HostWriteGuard, MemoryKind, Placement, SharedTensorAllocationDomain, Tensor, TensorRead,
    TypedTensor,
};

use crate::LinalgBackend;

thread_local! {
    static OBSERVED_OPERATION_ENTRY_DEPTH: Cell<usize> = const { Cell::new(0) };
}

pub(super) struct ObservedOperationEntryGuard;

pub(super) fn enter_observed_operation_scope() -> ObservedOperationEntryGuard {
    OBSERVED_OPERATION_ENTRY_DEPTH.with(|depth| depth.set(depth.get() + 1));
    ObservedOperationEntryGuard
}

impl Drop for ObservedOperationEntryGuard {
    fn drop(&mut self) {
        OBSERVED_OPERATION_ENTRY_DEPTH.with(|depth| depth.set(depth.get() - 1));
    }
}

#[derive(Debug, Default)]
pub(super) struct AccessCounts {
    pub(super) reads: AtomicUsize,
    pub(super) writes: AtomicUsize,
    pub(super) allocations: AtomicUsize,
    pub(super) outside_entry: AtomicUsize,
}

impl AccessCounts {
    fn observe_entry(&self) {
        let outside = OBSERVED_OPERATION_ENTRY_DEPTH.with(|depth| depth.get() == 0);
        if outside {
            self.outside_entry.fetch_add(1, Ordering::Relaxed);
        }
    }
}

struct FakeManagedBuffer<T> {
    values: Mutex<Vec<T>>,
    domain: Option<AllocationDomainId>,
    allocation: AllocationId,
    gpu_busy: AtomicBool,
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

impl<T: Copy + Send + Sync + 'static> BackendBuffer<T> for FakeManagedBuffer<T> {
    fn backend_family(&self) -> &'static str {
        "fake-managed"
    }

    fn len(&self) -> usize {
        self.values.lock().map_or(0, |values| values.len())
    }

    fn allocation_domain(&self) -> Option<AllocationDomainId> {
        self.domain
    }

    fn allocation_id(&self) -> Option<AllocationId> {
        Some(self.allocation)
    }

    fn map_read(&self) -> Result<HostReadGuard<'_, T>, HostAccessError> {
        self.counts.observe_entry();
        if self.gpu_busy.load(Ordering::Relaxed) {
            return Err(HostAccessError::GpuAccessInProgress);
        }
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
        self.counts.observe_entry();
        if self.gpu_busy.load(Ordering::Relaxed) {
            return Err(HostAccessError::GpuAccessInProgress);
        }
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
pub(super) struct FakeDomain {
    id: AllocationDomainId,
    next_allocation: AtomicU64,
    pub(super) counts: Arc<AccessCounts>,
}

impl FakeDomain {
    pub(super) fn new() -> Arc<Self> {
        Arc::new(Self {
            id: AllocationDomainId::fresh(),
            next_allocation: AtomicU64::new(1),
            counts: Arc::new(AccessCounts::default()),
        })
    }

    fn next_id(&self) -> AllocationId {
        AllocationId::from_backend_id(self.next_allocation.fetch_add(1, Ordering::Relaxed))
    }

    pub(super) fn tensor<T: Copy + Send + Sync + 'static>(
        &self,
        shape: &[usize],
        values: Vec<T>,
    ) -> TypedTensor<T> {
        self.tensor_with_domain(shape, values, Some(self.id), false, MemoryKind::Managed)
    }

    fn tensor_with_domain<T: Copy + Send + Sync + 'static>(
        &self,
        shape: &[usize],
        values: Vec<T>,
        domain: Option<AllocationDomainId>,
        gpu_busy: bool,
        memory_kind: MemoryKind,
    ) -> TypedTensor<T> {
        let buffer = FakeManagedBuffer {
            values: Mutex::new(values),
            domain,
            allocation: self.next_id(),
            gpu_busy: AtomicBool::new(gpu_busy),
            counts: Arc::clone(&self.counts),
        };
        TypedTensor::from_buffer_col_major(
            shape.to_vec(),
            Buffer::Backend(Arc::new(buffer)),
            Placement {
                memory_kind,
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
        self.counts.observe_entry();
        self.counts.allocations.fetch_add(1, Ordering::Relaxed);
        let len = Self::element_count(shape)?;
        Ok(match dtype {
            DType::F32 => Tensor::F32(self.tensor(shape, vec![0.0_f32; len])),
            DType::F64 => Tensor::F64(self.tensor(shape, vec![0.0_f64; len])),
            DType::C32 => Tensor::C32(self.tensor(shape, vec![Complex32::new(0.0, 0.0); len])),
            DType::C64 => Tensor::C64(self.tensor(shape, vec![Complex64::new(0.0, 0.0); len])),
            other => {
                return Err(tenferro_tensor::Error::unsupported_dtype(
                    "cholesky",
                    other,
                    "fake domain supports floating and complex Cholesky outputs",
                ));
            }
        })
    }
}

fn backend(domain: &Arc<FakeDomain>) -> CpuBackend {
    let erased: Arc<dyn SharedTensorAllocationDomain> = domain.clone();
    CpuBackend::new().with_allocation_domain(erased)
}

fn assert_real_factor(values: &[f64]) {
    let expected = [2.0_f64, 1.0, 0.0, 2.0_f64.sqrt()];
    for (actual, expected) in values.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1.0e-5,
            "expected {expected}, got {actual}"
        );
    }
}

#[test]
fn domain_bound_backend_preserves_host_owned_and_read_cholesky() {
    let domain = FakeDomain::new();
    let host = Tensor::from_vec_col_major([2, 2], vec![4.0_f64, 2.0, 2.0, 3.0]).unwrap();
    let expected = CpuBackend::new().cholesky(&host).unwrap();
    let mut backend = backend(&domain);

    let direct = backend.cholesky(&host).unwrap();
    let read = backend
        .cholesky_read(TensorRead::from_tensor(&host))
        .unwrap();

    assert_eq!(
        direct.as_slice::<f64>().unwrap(),
        expected.as_slice::<f64>().unwrap()
    );
    assert_eq!(
        read.as_slice::<f64>().unwrap(),
        expected.as_slice::<f64>().unwrap()
    );
    assert_eq!(domain.counts.reads.load(Ordering::Relaxed), 0);
    assert_eq!(domain.counts.writes.load(Ordering::Relaxed), 0);
}

#[test]
fn fake_managed_cholesky_covers_all_cpu_dtypes_and_guarded_output() {
    let domain = FakeDomain::new();
    let mut backend = backend(&domain);

    macro_rules! check_real {
        ($scalar:ty, $variant:ident) => {{
            let input = domain.tensor(
                &[2, 2],
                vec![
                    4.0 as $scalar,
                    2.0 as $scalar,
                    2.0 as $scalar,
                    3.0 as $scalar,
                ],
            );
            let input_id = input.allocation_id();
            let output = backend.cholesky(&Tensor::$variant(input)).unwrap();
            let Tensor::$variant(output) = output else {
                unreachable!()
            };
            assert_eq!(output.allocation_domain(), Some(domain.id));
            assert_ne!(output.allocation_id(), input_id);
            let Buffer::Backend(buffer) = output.buffer() else {
                panic!("expected backend output")
            };
            let mapped = buffer.map_read().unwrap();
            assert_real_factor(&mapped.iter().map(|&value| value as f64).collect::<Vec<_>>());
        }};
    }

    macro_rules! check_complex {
        ($scalar:ty, $variant:ident, $real:ty) => {{
            let value = |real| <$scalar>::new(real as $real, 0.0);
            let input = domain.tensor(
                &[2, 2],
                vec![value(4.0), value(2.0), value(2.0), value(3.0)],
            );
            let input_id = input.allocation_id();
            let output = backend.cholesky(&Tensor::$variant(input)).unwrap();
            let Tensor::$variant(output) = output else {
                unreachable!()
            };
            assert_eq!(output.allocation_domain(), Some(domain.id));
            assert_ne!(output.allocation_id(), input_id);
            let Buffer::Backend(buffer) = output.buffer() else {
                panic!("expected backend output")
            };
            let mapped = buffer.map_read().unwrap();
            assert!(mapped.iter().all(|value| value.im.abs() <= 1.0e-5));
            assert_real_factor(
                &mapped
                    .iter()
                    .map(|value| value.re as f64)
                    .collect::<Vec<_>>(),
            );
        }};
    }

    check_real!(f32, F32);
    check_real!(f64, F64);
    check_complex!(Complex32, C32, f32);
    check_complex!(Complex64, C64, f64);
    assert_eq!(domain.counts.writes.load(Ordering::Relaxed), 4);
}

#[test]
fn fake_managed_cholesky_rejects_foreign_device_local_and_busy_buffers() {
    let domain = FakeDomain::new();
    let foreign = FakeDomain::new();
    let mut backend = backend(&domain);
    let values = vec![4.0_f32, 2.0, 2.0, 3.0];

    let foreign_tensor = foreign.tensor(&[2, 2], values.clone());
    let error = backend.cholesky(&Tensor::F32(foreign_tensor)).unwrap_err();
    assert!(matches!(
        error,
        tenferro_tensor::Error::HostAccess {
            source: HostAccessError::ForeignDomain { .. },
            ..
        }
    ));

    let device_local =
        domain.tensor_with_domain(&[2, 2], values.clone(), None, false, MemoryKind::Device);
    let error = backend.cholesky(&Tensor::F32(device_local)).unwrap_err();
    assert!(matches!(
        error,
        tenferro_tensor::Error::HostAccess {
            source: HostAccessError::Unsupported { .. },
            ..
        }
    ));

    let busy =
        domain.tensor_with_domain(&[2, 2], values, Some(domain.id), true, MemoryKind::Managed);
    let error = backend.cholesky(&Tensor::F32(busy)).unwrap_err();
    assert!(matches!(
        error,
        tenferro_tensor::Error::HostAccess {
            source: HostAccessError::GpuAccessInProgress,
            ..
        }
    ));
    assert_eq!(domain.counts.writes.load(Ordering::Relaxed), 0);
}
