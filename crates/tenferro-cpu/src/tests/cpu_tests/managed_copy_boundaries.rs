//! Managed-domain materialization and the read-to-write copy boundary.

#[allow(unused_imports)]
use crate::test_support::RunErased;
use std::any::Any;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use num_complex::{Complex32, Complex64};
use tenferro_tensor::{
    AllocationDomainId, AllocationId, BackendStorage, DType, DynRank,
    ErasedHostTensor, Host, HostAccessError, HostReadGuard, HostWriteGuard, MemoryKind, Placement,
    SharedTensorAllocationDomain, StorageBuffer, Tensor, TensorRead, TensorScalar, TensorView,
    TensorWrite, TypedTensor, TypedTensorView,
};

use crate::{CpuEngine, Error};

struct FakeManagedBuffer<T> {
    values: Mutex<Vec<T>>,
    domain: AllocationDomainId,
    allocation: AllocationId,
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
        Ok(HostReadGuard::new(guard))
    }

    fn map_write(&mut self) -> Result<HostWriteGuard<'_, T>, HostAccessError> {
        let mut guard = self
            .values
            .lock()
            .map_err(|_| HostAccessError::BackendFailure {
                message: "fake write lock poisoned".to_string(),
            })?;
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
}

impl FakeDomain {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            id: AllocationDomainId::fresh(),
            next_allocation: AtomicU64::new(1),
        })
    }

    fn typed<T: TensorScalar + Copy + Send + Sync + 'static>(
        &self,
        shape: &[usize],
        values: Vec<T>,
    ) -> Tensor {
        self.backend_tensor(shape, values, MemoryKind::Managed)
    }

    /// A backend buffer in this domain with an explicit memory kind.
    fn backend_tensor<T: TensorScalar + Copy + Send + Sync + 'static>(
        &self,
        shape: &[usize],
        values: Vec<T>,
        memory_kind: MemoryKind,
    ) -> Tensor {
        let buffer = FakeManagedBuffer {
            values: Mutex::new(values),
            domain: self.id,
            allocation: AllocationId::from_backend_id(
                self.next_allocation.fetch_add(1, Ordering::Relaxed),
            ),
        };
        Tensor::from_typed(
            TypedTensor::<T>::from_buffer_col_major(
                shape.to_vec(),
                StorageBuffer::Backend(Box::new(buffer)),
                Placement {
                    memory_kind,
                    device: None,
                    cpu_affinity: None,
                },
            )
            .unwrap(),
        )
    }
}

impl SharedTensorAllocationDomain for FakeDomain {
    fn id(&self) -> AllocationDomainId {
        self.id
    }

    fn allocate(&self, dtype: DType, shape: &[usize]) -> tenferro_tensor::Result<Tensor> {
        let len = shape.iter().product();
        Ok(match dtype {
            DType::F32 => self.typed(shape, vec![0.0_f32; len]),
            DType::F64 => self.typed(shape, vec![0.0_f64; len]),
            DType::I32 => self.typed(shape, vec![0_i32; len]),
            DType::I64 => self.typed(shape, vec![0_i64; len]),
            DType::Bool => self.typed(shape, vec![false; len]),
            DType::C32 => self.typed(shape, vec![Complex32::default(); len]),
            DType::C64 => self.typed(shape, vec![Complex64::default(); len]),
            other => {
                return Err(Error::unsupported_dtype(
                    "FakeDomain::allocate",
                    other,
                    "fake domain allocates preset scalars only",
                ))
            }
        })
    }
}

/// An allocator in the input's domain that returns one specific wrong output.
#[derive(Debug)]
struct FaultyDomain {
    inner: Arc<FakeDomain>,
    fault: Fault,
}

#[derive(Clone, Copy, Debug)]
enum Fault {
    AllocationFails,
    WrongShape,
    ForeignOutputDomain,
    WrongDtype,
}

impl SharedTensorAllocationDomain for FaultyDomain {
    fn id(&self) -> AllocationDomainId {
        self.inner.id
    }

    fn allocate(&self, dtype: DType, shape: &[usize]) -> tenferro_tensor::Result<Tensor> {
        match self.fault {
            Fault::AllocationFails => Err(Error::runtime_state(
                "FaultyDomain::allocate",
                "allocation unavailable",
            )),
            Fault::WrongShape => self.inner.allocate(dtype, &[1]),
            Fault::ForeignOutputDomain => FakeDomain::new().allocate(dtype, shape),
            Fault::WrongDtype => self.inner.allocate(DType::F32, shape),
        }
    }
}

fn backend_in(domain: Arc<dyn SharedTensorAllocationDomain>) -> CpuEngine {
    crate::test_support::cpu_engine(1)
        .unwrap()
        .with_allocation_domain(domain)
}

fn to_contiguous(backend: &mut CpuEngine, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
    crate::test_support::on_engine(backend, |session| session.to_contiguous_read(TensorRead::from_tensor(input)))
        .unwrap()
}

/// The managed buffer's elements, its domain, and its allocation id.
fn managed_parts<T: TensorScalar + Copy + 'static>(
    tensor: &Tensor,
) -> (Vec<T>, Option<AllocationDomainId>, Option<AllocationId>) {
    let typed = tensor.as_typed::<T>().expect("managed tensor dtype");
    let values = typed.backend_buffer().unwrap().map_read().unwrap().to_vec();
    (values, typed.allocation_domain(), typed.allocation_id())
}

#[test]
fn managed_to_contiguous_copies_every_dtype_into_a_fresh_same_domain_allocation() {
    fn check<T: TensorScalar + Copy + PartialEq + fmt::Debug + Send + Sync + 'static>(
        values: Vec<T>,
    ) {
        let domain = FakeDomain::new();
        let input = domain.typed(&[2, 2], values.clone());
        let mut backend = backend_in(domain.clone());

        let output = to_contiguous(&mut backend, &input).unwrap();

        assert_eq!(output.dtype(), T::dtype());
        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(output.placement().memory_kind, MemoryKind::Managed);
        let (copied, output_domain, output_id) = managed_parts::<T>(&output);
        let (_, _, input_id) = managed_parts::<T>(&input);
        assert_eq!(copied, values);
        assert_eq!(output_domain, Some(domain.id));
        // A semantic snapshot owns independent storage, not an alias of the input.
        assert_ne!(output_id, input_id);
    }

    check(vec![1.0_f32, 2.0, 3.0, 4.0]);
    check(vec![1.0_f64, 2.0, 3.0, 4.0]);
    check(vec![1_i32, -2, 3, -4]);
    check(vec![1_i64, -2, 3, -4]);
    check(vec![true, false, false, true]);
    check((1..=4).map(|x| Complex32::new(x as f32, -1.0)).collect());
    check((1..=4).map(|x| Complex64::new(x as f64, -1.0)).collect());
}

#[test]
fn managed_to_contiguous_rejects_incompatible_allocator_output() {
    for (fault, expected_message) in [
        (Fault::AllocationFails, "allocation unavailable"),
        (
            Fault::WrongShape,
            "shared allocator returned incompatible output",
        ),
        (
            Fault::ForeignOutputDomain,
            "shared allocator returned incompatible output",
        ),
        (
            Fault::WrongDtype,
            "shared allocator returned the wrong dtype",
        ),
    ] {
        let domain = FakeDomain::new();
        let input = domain.typed(&[2], vec![1.0_f64, 2.0]);
        let mut backend = backend_in(Arc::new(FaultyDomain {
            inner: domain.clone(),
            fault,
        }));

        let error = to_contiguous(&mut backend, &input).unwrap_err();

        match error {
            Error::RuntimeState { message, .. } => {
                assert_eq!(message, expected_message, "{fault:?}")
            }
            other => panic!("{fault:?}: expected RuntimeState, got {other:?}"),
        }
        // The failed copy leaves the managed input readable and unchanged.
        assert_eq!(managed_parts::<f64>(&input).0, [1.0, 2.0]);
    }
}

#[test]
fn managed_to_contiguous_refuses_an_input_from_another_domain() {
    let own = FakeDomain::new();
    let foreign = FakeDomain::new();
    let input = foreign.typed(&[2], vec![1.0_f32, 2.0]);
    let mut backend = backend_in(own.clone());

    let error = to_contiguous(&mut backend, &input).unwrap_err();

    assert!(
        matches!(
            &error,
            Error::HostAccess {
                source: HostAccessError::ForeignDomain { expected, actual },
                ..
            } if *expected == own.id && *actual == foreign.id
        ),
        "{error:?}"
    );
}

#[test]
fn managed_to_contiguous_refuses_a_backend_input_outside_managed_memory() {
    let domain = FakeDomain::new();
    let mut backend = backend_in(domain.clone());
    // Same domain and a backend buffer, but device memory is not host-mappable.
    let input = domain.backend_tensor(&[2], vec![1.0_f64, 2.0], MemoryKind::Device);

    let error = to_contiguous(&mut backend, &input).unwrap_err();

    assert!(
        matches!(
            error,
            Error::HostAccess {
                source: HostAccessError::Unsupported { backend: "backend" },
                ..
            }
        ),
        "{error:?}"
    );
}

#[test]
fn host_input_under_a_managed_domain_is_cloned_as_host_storage() {
    let domain = FakeDomain::new();
    let mut backend = backend_in(domain.clone());
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();

    let output = to_contiguous(&mut backend, &input).unwrap();

    // Only backend-owned inputs enter the shared domain; a host tensor keeps
    // its host placement and gets an independent host allocation.
    assert_eq!(output.placement(), input.placement());
    assert_eq!(output.as_slice::<f64>().unwrap(), [1.0, 2.0]);
    assert_ne!(
        output.as_slice::<f64>().unwrap().as_ptr(),
        input.as_slice::<f64>().unwrap().as_ptr()
    );
}

#[test]
fn to_contiguous_copies_an_externally_defined_payload_into_owned_storage() {
    let payload = ErasedHostTensor::new(
        TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1.0_f64, 2.0])
            .unwrap(),
    );
    let element = payload.element_type_id();
    let input = Tensor::external(payload);
    let mut backend = crate::test_support::cpu_engine(1).unwrap();

    let output = to_contiguous(&mut backend, &input).unwrap();

    assert_eq!(output.dtype(), DType::External(element));
    assert_eq!(output.placement(), input.placement());
    let payload_values = |tensor: &Tensor| {
        tensor
            .external_payload()
            .and_then(|payload| payload.downcast_ref::<f64>())
            .expect("an f64 external payload")
            .as_slice()
            .as_ptr()
    };
    assert_eq!(
        output
            .external_payload()
            .and_then(|payload| payload.downcast_ref::<f64>())
            .unwrap()
            .as_slice(),
        [1.0, 2.0]
    );
    // The copy owns its storage rather than aliasing the caller's payload.
    assert_ne!(payload_values(&output), payload_values(&input));
}

fn copy_into(read: TensorRead<'_>, write: TensorWrite<'_>) -> tenferro_tensor::Result<()> {
    crate::test_support::cpu_engine(1)
        .unwrap()
        .run_erased(|session| session.copy_read_into(read, write))
        .unwrap()
}

#[test]
fn copy_read_into_copies_each_borrowed_view_dtype_into_an_owned_tensor() {
    macro_rules! check {
        ($variant:ident, $ty:ty, $values:expr) => {{
            let values: [$ty; 4] = $values;
            let mut destination = Tensor::from_typed(
                TypedTensor::<$ty>::from_vec_col_major(vec![2, 2], vec![<$ty>::default(); 4])
                    .unwrap(),
            );
            // A transposed source exercises the strided copy, not a flat memcpy.
            let source = TypedTensorView::from_col_major(&[2, 2], &values)
                .unwrap()
                .transpose_view([1, 0])
                .unwrap();
            copy_into(
                TensorRead::from_view(TensorView::$variant(source)),
                TensorWrite::from_tensor(&mut destination),
            )
            .unwrap();
            assert_eq!(
                destination.as_slice::<$ty>().unwrap(),
                [values[0], values[2], values[1], values[3]]
            );
        }};
    }

    check!(F32, f32, [1.0, 2.0, 3.0, 4.0]);
    check!(F64, f64, [1.0, 2.0, 3.0, 4.0]);
    check!(I32, i32, [1, 2, 3, 4]);
    check!(I64, i64, [1, 2, 3, 4]);
    check!(Bool, bool, [true, false, true, true]);
    check!(
        C32,
        Complex32,
        [1.0, 2.0, 3.0, 4.0].map(|x| Complex32::new(x, -x))
    );
    check!(
        C64,
        Complex64,
        [1.0, 2.0, 3.0, 4.0].map(|x| Complex64::new(x, -x))
    );
}

#[test]
fn copy_read_into_reports_a_view_dtype_mismatch_without_writing() {
    let source = [1.0_f32, 2.0];
    let mut destination =
        Tensor::from_typed(TypedTensor::<f64>::from_vec_col_major(vec![2], vec![-1.0; 2]).unwrap());

    let error = copy_into(
        TensorRead::from_view(TensorView::F32(
            TypedTensorView::from_col_major(&[2], &source).unwrap(),
        )),
        TensorWrite::from_tensor(&mut destination),
    )
    .unwrap_err();

    assert!(
        matches!(error, Error::Validation { .. }),
        "expected a dtype-mismatch validation error, got {error:?}"
    );
    assert!(error.to_string().contains("F32") && error.to_string().contains("F64"));
    assert_eq!(destination.as_slice::<f64>().unwrap(), [-1.0, -1.0]);
}

#[test]
fn copy_read_into_refuses_an_externally_defined_source_payload() {
    let payload = ErasedHostTensor::new(
        TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1.0_f64, 2.0])
            .unwrap(),
    );
    let element = payload.element_type_id();
    let source = Tensor::external(payload);
    let mut destination =
        Tensor::from_typed(TypedTensor::<f64>::from_vec_col_major(vec![2], vec![-1.0; 2]).unwrap());

    let error = copy_into(
        TensorRead::from_tensor(&source),
        TensorWrite::from_tensor(&mut destination),
    )
    .unwrap_err();

    match error {
        Error::UnsupportedDType { op, dtype, message } => {
            assert_eq!(op, "copy_tensor_read_into");
            assert_eq!(dtype, DType::External(element));
            assert_eq!(
                message,
                "an externally defined payload is not a runtime read"
            );
        }
        other => panic!("expected UnsupportedDType, got {other:?}"),
    }
    assert_eq!(destination.as_slice::<f64>().unwrap(), [-1.0, -1.0]);
}
