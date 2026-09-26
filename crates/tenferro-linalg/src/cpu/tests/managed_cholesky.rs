use std::any::Any;
use std::cell::Cell;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{
    AllocationDomainId, AllocationId, BackendSessionHost, BackendStorage, DType, HostAccessError,
    HostReadGuard, HostWriteGuard, MemoryKind, Placement, SharedTensorAllocationDomain,
    StorageBuffer, Tensor, TensorRead, TensorScalar, TypedTensor,
};

use super::with_cpu_linalg;
use crate::LinalgBackend;

/// The Rust scalar type behind a preset variant name a macro received.
macro_rules! preset_scalar {
    (F32) => {
        f32
    };
    (F64) => {
        f64
    };
    (I32) => {
        i32
    };
    (I64) => {
        i64
    };
    (Bool) => {
        bool
    };
    (C32) => {
        num_complex::Complex32
    };
    (C64) => {
        num_complex::Complex64
    };
}
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

impl<T: Copy + Send + Sync + 'static> BackendStorage<T> for FakeManagedBuffer<T> {
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

    fn map_write(&mut self) -> Result<HostWriteGuard<'_, T>, HostAccessError> {
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

    pub(super) fn tensor<T: TensorScalar + Copy + Send + Sync + 'static>(
        &self,
        shape: &[usize],
        values: Vec<T>,
    ) -> TypedTensor<T> {
        self.tensor_with_domain(shape, values, Some(self.id), false, MemoryKind::Managed)
    }

    fn tensor_with_domain<T: TensorScalar + Copy + Send + Sync + 'static>(
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
            StorageBuffer::Backend(Box::new(buffer)),
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
            DType::F32 => Tensor::from_typed::<f32>(self.tensor(shape, vec![0.0_f32; len])),
            DType::F64 => Tensor::from_typed::<f64>(self.tensor(shape, vec![0.0_f64; len])),
            DType::C32 => Tensor::from_typed::<tenferro_tensor::Complex32>(
                self.tensor(shape, vec![Complex32::new(0.0, 0.0); len]),
            ),
            DType::C64 => Tensor::from_typed::<tenferro_tensor::Complex64>(
                self.tensor(shape, vec![Complex64::new(0.0, 0.0); len]),
            ),
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
    let mut expected_backend = CpuBackend::new();
    let expected =
        with_cpu_linalg(&mut expected_backend, |backend| backend.cholesky(&host)).unwrap();
    let mut backend = backend(&domain);

    let (direct, read) = with_cpu_linalg(&mut backend, |backend| {
        let direct = backend.cholesky(&host)?;
        let read = backend.cholesky_read(TensorRead::from_tensor(&host))?;
        Ok::<_, tenferro_tensor::Error>((direct, read))
    })
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
    let selected = backend.execution_info().domain_id();

    with_cpu_linalg(&mut backend, |backend| {
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
                let output = backend
                    .cholesky(&Tensor::from_typed::<preset_scalar!($variant)>(input))
                    .unwrap();
                let Ok(output) = output.into_typed::<preset_scalar!($variant)>() else {
                    unreachable!()
                };
                assert_eq!(output.allocation_domain(), Some(domain.id));
                assert_ne!(output.allocation_id(), input_id);
                assert_eq!(output.placement().memory_kind, MemoryKind::Managed);
                assert_eq!(output.placement().device, None);
                assert_eq!(output.placement().cpu_affinity, Some(selected));
                let StorageBuffer::Backend(buffer) = output.buffer() else {
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
                let output = backend
                    .cholesky(&Tensor::from_typed::<preset_scalar!($variant)>(input))
                    .unwrap();
                let Ok(output) = output.into_typed::<preset_scalar!($variant)>() else {
                    unreachable!()
                };
                assert_eq!(output.allocation_domain(), Some(domain.id));
                assert_ne!(output.allocation_id(), input_id);
                assert_eq!(output.placement().memory_kind, MemoryKind::Managed);
                assert_eq!(output.placement().device, None);
                assert_eq!(output.placement().cpu_affinity, Some(selected));
                let StorageBuffer::Backend(buffer) = output.buffer() else {
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
    });
}

#[test]
fn fake_managed_cholesky_rejects_foreign_device_local_and_busy_buffers() {
    let domain = FakeDomain::new();
    let foreign = FakeDomain::new();
    let mut backend = backend(&domain);
    let values = vec![4.0_f32, 2.0, 2.0, 3.0];

    with_cpu_linalg(&mut backend, |backend| {
        let foreign_tensor = foreign.tensor(&[2, 2], values.clone());
        let error = backend
            .cholesky(&Tensor::from_typed::<f32>(foreign_tensor))
            .unwrap_err();
        assert!(matches!(
            error,
            tenferro_tensor::Error::HostAccess {
                source: HostAccessError::ForeignDomain { .. },
                ..
            }
        ));

        let device_local = domain.tensor_with_domain(
            &[2, 2],
            values.clone(),
            Some(domain.id),
            false,
            MemoryKind::Device,
        );
        let error = backend
            .cholesky(&Tensor::from_typed::<f32>(device_local))
            .unwrap_err();
        assert!(matches!(
            error,
            tenferro_tensor::Error::HostAccess {
                source: HostAccessError::Unsupported { .. },
                ..
            }
        ));

        let busy =
            domain.tensor_with_domain(&[2, 2], values, Some(domain.id), true, MemoryKind::Managed);
        let error = backend
            .cholesky(&Tensor::from_typed::<f32>(busy))
            .unwrap_err();
        assert!(matches!(
            error,
            tenferro_tensor::Error::HostAccess {
                source: HostAccessError::GpuAccessInProgress,
                ..
            }
        ));
        assert_eq!(domain.counts.writes.load(Ordering::Relaxed), 0);
    });
}

#[test]
fn managed_snapshot_is_independent_and_rejects_foreign_storage() {
    use tenferro_tensor::{TensorStructural, TensorView};
    let domain = FakeDomain::new();
    let mut cpu = backend(&domain);
    let mut input = domain.tensor(&[2, 2], vec![4.0_f64, 2.0, 2.0, 3.0]);
    let snapshot = with_cpu_linalg(&mut cpu, |session| {
        session.to_contiguous_read(TensorRead::from_view(TensorView::F64(input.as_view())))
    })
    .unwrap();
    let snapshot = snapshot.as_typed::<f64>().unwrap();
    assert_eq!(snapshot.allocation_domain(), Some(domain.id()));
    assert_ne!(snapshot.allocation_id(), input.allocation_id());
    input
        .backend_buffer_mut()
        .unwrap()
        .map_write()
        .unwrap()
        .copy_from_slice(&[99.0; 4])
        .unwrap();
    assert_eq!(
        snapshot.with_host_read(|data| data.to_vec()).unwrap(),
        vec![4.0, 2.0, 2.0, 3.0]
    );

    let other = FakeDomain::new();
    let foreign = Tensor::from_typed(other.tensor(&[1], vec![2.0_f64]));
    let before = domain.counts.allocations.load(Ordering::Relaxed);
    let error = cpu
        .with_backend_session(|__s| __s.to_contiguous_read(TensorRead::from_tensor(&foreign)))
        .unwrap_err();
    assert!(matches!(
        error,
        tenferro_tensor::Error::HostAccess {
            source: HostAccessError::ForeignDomain { .. },
            ..
        }
    ));
    assert_eq!(other.counts.reads.load(Ordering::Relaxed), 0);
    assert_eq!(domain.counts.allocations.load(Ordering::Relaxed), before);
    for (memory_kind, busy) in [(MemoryKind::Device, false), (MemoryKind::Managed, true)] {
        let invalid = Tensor::from_typed(domain.tensor_with_domain(
            &[1],
            vec![1.0_f64],
            Some(domain.id()),
            busy,
            memory_kind,
        ));
        assert!(matches!(
            cpu.with_backend_session(
                |__s| __s.to_contiguous_read(TensorRead::from_tensor(&invalid))
            ),
            Err(tenferro_tensor::Error::HostAccess { .. })
        ));
        assert_eq!(domain.counts.allocations.load(Ordering::Relaxed), before);
    }
    let transposed = input.as_view().transpose_view([1, 0]).unwrap();
    assert!(cpu
        .with_backend_session(
            |__s| __s.to_contiguous_read(TensorRead::from_view(TensorView::F64(transposed)))
        )
        .is_err());
    assert_eq!(domain.counts.allocations.load(Ordering::Relaxed), before);
}

#[test]
fn managed_borrowed_cholesky_respects_offset_and_rejects_strides() {
    let domain = FakeDomain::new();
    let input = domain.tensor(&[2, 3], vec![-99.0_f64, -99.0, 4.0, 2.0, 2.0, 3.0]);
    let view = input
        .as_view()
        .try_slice_axis(1, tenferro_tensor::StridedSliceSpec::new(1, Some(3), 1))
        .unwrap();
    with_cpu_linalg(&mut backend(&domain), |session| {
        let output = session
            .cholesky_read(TensorRead::from_view(tenferro_tensor::TensorView::F64(
                view.clone(),
            )))
            .unwrap();
        output
            .as_typed::<f64>()
            .unwrap()
            .with_host_read(assert_real_factor)
            .unwrap();
        let strided = view.transpose_view([1, 0]).unwrap();
        assert!(session
            .cholesky_read(TensorRead::from_view(tenferro_tensor::TensorView::F64(
                strided
            )))
            .is_err());
    });
}

#[cfg(feature = "autodiff")]
#[test]
fn managed_eager_cholesky_preserves_domain_and_values() {
    use crate::EagerTensorLinalgExt;
    use tenferro_ad::{EagerRuntime, EagerTensor};
    let domain = FakeDomain::new();
    let input = domain.tensor(&[2, 2], vec![4.0_f64, 2.0, 2.0, 3.0]);
    let input_id = input.allocation_id();
    let runtime = EagerRuntime::with_cpu_backend(backend(&domain)).unwrap();
    let eager = EagerTensor::from_tensor_in(Tensor::from_typed(input), runtime).unwrap();
    let output = eager.cholesky().unwrap().to_tensor().unwrap();
    let output = output.as_typed::<f64>().unwrap();
    assert_eq!(output.allocation_domain(), Some(domain.id()));
    assert_ne!(output.allocation_id(), input_id);
    output.with_host_read(assert_real_factor).unwrap();
}

#[test]
fn managed_traced_cholesky_preserves_domain_and_values() {
    use crate::TracedTensorLinalgExt;
    use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
    let domain = FakeDomain::new();
    let input = domain.tensor(&[2, 2], vec![4.0_f64, 2.0, 2.0, 3.0]);
    let input_id = input.allocation_id();
    let traced = TracedTensor::from_tensor_concrete_shape(Tensor::from_typed(input)).unwrap();
    let program = GraphCompiler::new()
        .compile(&traced.cholesky().unwrap())
        .unwrap();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&backend(&domain)).unwrap())
        .unwrap();
    builder
        .install_extension_module(
            crate::extension_module::<CpuBackend>(tenferro_cpu::runtime_engine_id().unwrap())
                .unwrap(),
        )
        .unwrap();
    let output = builder
        .build()
        .unwrap()
        .run_compiled(&program, &[])
        .unwrap()
        .remove(0);
    let output = output.as_typed::<f64>().unwrap();
    assert_eq!(output.allocation_domain(), Some(domain.id()));
    assert_ne!(output.allocation_id(), input_id);
    output.with_host_read(assert_real_factor).unwrap();
}
