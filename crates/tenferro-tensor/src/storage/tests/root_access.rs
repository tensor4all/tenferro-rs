//! Host and backend root access through the private owner capabilities.

use std::any::Any;
use std::fmt::Debug;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use num_complex::{Complex32, Complex64};

use crate::{
    AllocationDomainId, AllocationId, BackendStorage, DType, DeviceAccessError,
    DeviceAccessRequest, PreparedDeviceAccess, StorageBuffer, TensorScalar,
};

use super::super::root::{
    import_backend_buffer, import_host_vec, HostAllocation, OwnedStorage, RootImportError,
    RootResourcePin,
};
use super::super::{
    AccessError, BackendAllocation, ByteRange, HostBufferRecycler, ProviderCapabilities,
    ProviderKind,
};

#[test]
fn provider_capabilities_distinguish_host_access() {
    assert!(!ProviderCapabilities::none().host_access());
    assert!(ProviderCapabilities::host().host_access());
    assert_eq!(
        ProviderCapabilities::default(),
        ProviderCapabilities::none()
    );
}

fn host_round_trip<T>(data: Vec<T>, replacement: T)
where
    T: TensorScalar + Copy + PartialEq + Debug,
{
    let dtype = T::dtype();
    let byte_len = data.len() * std::mem::size_of::<T>();
    let mut owner = import_host_vec(data.clone()).unwrap();
    let span = owner.root_span();

    assert_eq!(owner.provider_kind(), ProviderKind::Cpu);
    assert!(owner.backend_allocation().is_none());
    assert!(owner.backend_buffer::<T>().is_none());
    assert!(owner.backend_buffer_mut::<T>().is_none());
    assert_eq!(owner.root_identity().extent().byte_len(), byte_len);
    assert!(matches!(owner.host_buffer::<T>(), Some(StorageBuffer::Host(host)) if host == &data));

    {
        let read = owner.as_ref();
        assert_eq!(read.provider_kind(), ProviderKind::Cpu);
        assert_eq!(read.host_slice::<T>(span, dtype).unwrap(), data.as_slice());
        assert_eq!(read.map_read(span, dtype).unwrap().bytes().len(), byte_len);
    }
    {
        let mut write = owner.as_mut();
        assert!(write.backend_buffer_mut::<T>().is_none());
        assert_eq!(write.map_write(span, dtype).unwrap().len(), byte_len);
        write.host_slice_mut::<T>(span, dtype).unwrap()[0] = replacement;
    }

    let mut expected = data;
    expected[0] = replacement;
    assert_eq!(
        owner
            .into_host_vec::<T>()
            .map_err(|(_, error)| error)
            .unwrap(),
        expected
    );
}

#[test]
fn host_roots_round_trip_every_preset_scalar() {
    host_round_trip(vec![1.0_f32, 2.0], -1.0);
    host_round_trip(vec![1.0_f64, 2.0, 3.0], -1.0);
    host_round_trip(vec![1_i32, 2], -1);
    host_round_trip(vec![1_i64, 2], -1);
    host_round_trip(vec![true, false, true], false);
    host_round_trip(vec![Complex32::new(1.0, 2.0)], Complex32::new(0.0, -1.0));
    host_round_trip(
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
        Complex64::new(0.0, -1.0),
    );
}

#[test]
fn complex_host_roots_reinterpret_as_interleaved_real_parts() {
    let mut owner =
        import_host_vec(vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)]).unwrap();
    let span = owner.root_span();
    assert_eq!(
        owner.as_ref().host_slice::<f64>(span, DType::F64).unwrap(),
        &[1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        owner
            .as_ref()
            .map_read(span, DType::F64)
            .unwrap()
            .bytes()
            .len(),
        32
    );
    owner
        .as_mut()
        .host_slice_mut::<f64>(span, DType::F64)
        .unwrap()[3] = -4.0;
    assert_eq!(
        owner
            .into_host_vec::<Complex64>()
            .map_err(|(_, error)| error)
            .unwrap(),
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)]
    );

    let owner = import_host_vec(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let span = owner.root_span();
    assert_eq!(
        owner
            .as_ref()
            .host_slice::<Complex32>(span, DType::C32)
            .unwrap(),
        &[Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)]
    );
}

#[test]
fn host_roots_reject_unsupported_representations() {
    let mut owner = import_host_vec(vec![1_i32, 2]).unwrap();
    let span = owner.root_span();
    let mismatch = |error: AccessError| match error {
        AccessError::DTypeMismatch { expected, actual } => (expected, actual),
        other => panic!("expected dtype mismatch, got {other:?}"),
    };

    let read = owner.as_ref();
    assert_eq!(
        mismatch(read.map_read(span, DType::F32).err().unwrap()),
        (DType::I32, DType::F32)
    );
    assert_eq!(
        mismatch(read.host_slice::<f32>(span, DType::F32).unwrap_err()),
        (DType::I32, DType::F32)
    );
    // The requested dtype must also name the slice element type.
    assert_eq!(
        mismatch(read.host_slice::<f32>(span, DType::I32).unwrap_err()),
        (DType::I32, DType::I32)
    );

    let write = owner.as_mut();
    assert_eq!(
        mismatch(write.map_write(span, DType::I64).err().unwrap()),
        (DType::I32, DType::I64)
    );
    assert_eq!(
        mismatch(write.host_slice_mut::<i64>(span, DType::I64).unwrap_err()),
        (DType::I32, DType::I64)
    );
}

#[test]
fn host_sub_spans_map_only_element_aligned_windows() {
    let mut owner = import_host_vec(vec![1_i32, 2, 3, 4]).unwrap();
    let identity = owner.root_identity();

    let window = identity.bind_relative_range(ByteRange::new(4, 8)).unwrap();
    assert_eq!(
        owner
            .as_ref()
            .host_slice::<i32>(window, DType::I32)
            .unwrap(),
        &[2, 3]
    );
    let mapping = owner.as_ref().map_read(window, DType::I32).unwrap();
    assert_eq!(mapping.bytes(), &[2, 0, 0, 0, 3, 0, 0, 0]);
    drop(mapping);
    owner
        .as_mut()
        .host_slice_mut::<i32>(window, DType::I32)
        .unwrap()[1] = 30;

    let not_element_aligned = |error: AccessError| match error {
        AccessError::Provider { message } => {
            assert_eq!(
                message,
                "host mapping span is not an element-aligned subrange"
            );
        }
        other => panic!("expected element-alignment failure, got {other:?}"),
    };
    // Offset 2 is not a multiple of the i32 size; length 6 is not either.
    for relative in [ByteRange::new(2, 4), ByteRange::new(4, 6)] {
        let span = identity.bind_relative_range(relative).unwrap();
        not_element_aligned(owner.as_ref().map_read(span, DType::I32).err().unwrap());
        not_element_aligned(
            owner
                .as_ref()
                .host_slice::<i32>(span, DType::I32)
                .unwrap_err(),
        );
        not_element_aligned(owner.as_mut().map_write(span, DType::I32).err().unwrap());
        not_element_aligned(
            owner
                .as_mut()
                .host_slice_mut::<i32>(span, DType::I32)
                .unwrap_err(),
        );
    }

    assert_eq!(
        owner
            .into_host_vec::<i32>()
            .map_err(|(_, error)| error)
            .unwrap(),
        vec![1, 2, 30, 4]
    );
}

#[test]
fn device_requests_are_validated_against_the_root_identity() {
    let mut owner = import_host_vec(vec![1_i32, 2]).unwrap();
    let key = owner.root_identity().extent().key();
    let shape = [2_usize];
    let strides = [1_isize];
    let request = |domain, byte_len, element_size| {
        DeviceAccessRequest::new(
            domain,
            key.local(),
            byte_len,
            element_size,
            &shape,
            &strides,
            0,
        )
    };
    let invalid = |message: &str| DeviceAccessError::InvalidRequest {
        message: message.to_owned(),
    };

    let read = owner.as_ref();
    assert_eq!(
        read.prepare_device_access(request(AllocationDomainId::fresh(), 8, 4))
            .unwrap_err(),
        invalid("prepared request does not match the root allocation identity")
    );
    assert_eq!(
        read.prepare_device_access(request(key.domain(), 16, 4))
            .unwrap_err(),
        invalid("prepared request exceeds the root allocation extent")
    );
    assert_eq!(
        read.prepare_device_access(request(key.domain(), 8, 0))
            .unwrap_err(),
        invalid("prepared request has a zero element size")
    );
    // A valid request reaches the pin, and a host root has no device path.
    assert_eq!(
        read.prepare_device_access(request(key.domain(), 8, 4))
            .unwrap_err(),
        DeviceAccessError::Unsupported { backend: "host" }
    );

    let write = owner.as_mut();
    assert_eq!(
        write
            .prepare_device_access(request(key.domain(), 8, 0))
            .unwrap_err(),
        invalid("prepared request has a zero element size")
    );
    assert_eq!(
        write
            .prepare_device_access(request(key.domain(), 8, 4))
            .unwrap_err(),
        DeviceAccessError::Unsupported { backend: "host" }
    );
}

#[derive(Debug)]
struct PreparedMarker(usize);

impl PreparedDeviceAccess for PreparedMarker {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

#[derive(Debug)]
struct ProviderBuffer {
    family: &'static str,
    domain: Option<AllocationDomainId>,
    allocation: Option<AllocationId>,
    len: usize,
    prepares: Arc<AtomicUsize>,
}

impl ProviderBuffer {
    fn identified(family: &'static str, local: u64) -> Self {
        Self {
            family,
            domain: Some(AllocationDomainId::fresh()),
            allocation: Some(AllocationId::from_backend_id(local)),
            len: 3,
            prepares: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl BackendStorage<f64> for ProviderBuffer {
    fn backend_family(&self) -> &'static str {
        self.family
    }

    fn len(&self) -> usize {
        self.len
    }

    fn allocation_domain(&self) -> Option<AllocationDomainId> {
        self.domain
    }

    fn allocation_id(&self) -> Option<AllocationId> {
        self.allocation
    }

    fn prepare_device_access(
        &self,
        request: DeviceAccessRequest<'_>,
    ) -> Result<Box<dyn PreparedDeviceAccess>, DeviceAccessError> {
        self.prepares.fetch_add(1, Ordering::Relaxed);
        Ok(Box::new(PreparedMarker(request.byte_len())))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn import_provider(buffer: ProviderBuffer) -> OwnedStorage {
    import_backend_buffer::<f64>(StorageBuffer::Backend(Box::new(buffer))).unwrap()
}

/// The `(backend, field)` a rejected backend import reports as missing.
fn missing_identity(buffer: StorageBuffer<f64>) -> (&'static str, &'static str) {
    match import_backend_buffer::<f64>(buffer) {
        Ok(_) => panic!("backend import must be rejected"),
        Err(error) => match error.source() {
            RootImportError::MissingAllocationIdentity { backend, field } => (*backend, *field),
            other => panic!("expected missing identity, got {other:?}"),
        },
    }
}

#[test]
fn backend_import_requires_a_provider_buffer_with_full_identity() {
    assert_eq!(
        missing_identity(StorageBuffer::Host(vec![1.0])),
        ("host", "backend buffer")
    );

    let mut no_domain = ProviderBuffer::identified("test-provider", 1);
    no_domain.domain = None;
    assert_eq!(
        missing_identity(StorageBuffer::Backend(Box::new(no_domain))),
        ("test-provider", "allocation domain")
    );

    let mut no_id = ProviderBuffer::identified("test-provider", 2);
    no_id.allocation = None;
    assert_eq!(
        missing_identity(StorageBuffer::Backend(Box::new(no_id))),
        ("test-provider", "allocation id")
    );
}

#[test]
fn backend_import_rejects_a_provider_length_whose_byte_size_overflows() {
    let mut oversized = ProviderBuffer::identified("test-provider", 6);
    oversized.len = usize::MAX;
    match import_backend_buffer::<f64>(StorageBuffer::Backend(Box::new(oversized))) {
        Ok(_) => panic!("an overflowing provider length must be rejected"),
        Err(error) => assert!(matches!(
            error.source(),
            RootImportError::ByteLengthOverflow { element_size: 8 }
        )),
    }
}

#[test]
fn backend_roots_map_provider_family_to_kind() {
    for (family, kind) in [
        ("cuda", ProviderKind::Cuda),
        ("webgpu", ProviderKind::WebGpu),
        ("test-provider", ProviderKind::Other("test-provider")),
    ] {
        let owner = import_provider(ProviderBuffer::identified(family, 3));
        assert_eq!(owner.provider_kind(), kind);
        assert_eq!(owner.as_ref().provider_kind(), kind);
    }
}

#[test]
fn backend_roots_expose_the_provider_buffer_and_delegate_device_access() {
    let buffer = ProviderBuffer::identified("test-provider", 4);
    let prepares = Arc::clone(&buffer.prepares);
    let mut owner = import_provider(buffer);
    let span = owner.root_span();
    let extent = owner.root_identity().extent();
    assert_eq!(extent.byte_len(), 3 * std::mem::size_of::<f64>());
    assert_eq!(
        extent.guaranteed_alignment().get(),
        std::mem::align_of::<f64>()
    );

    let allocation = owner.backend_allocation().unwrap();
    assert_eq!(allocation.capabilities(), ProviderCapabilities::none());
    let debug = format!("{allocation:?}");
    assert!(debug.contains("BackendStorageAllocation"), "{debug}");
    assert!(debug.contains("test-provider"), "{debug}");

    assert!(owner.host_buffer::<f64>().is_none());
    assert!(owner.backend_buffer::<f32>().is_none());
    assert!(matches!(
        owner.backend_buffer::<f64>(),
        Some(StorageBuffer::Backend(provider)) if provider.len() == 3
    ));
    assert!(owner.backend_buffer_mut::<f64>().is_some());
    assert!(owner.as_mut().backend_buffer_mut::<f64>().is_some());

    let key = extent.key();
    let shape = [3_usize];
    let strides = [1_isize];
    let request = DeviceAccessRequest::new(
        key.domain(),
        key.local(),
        extent.byte_len(),
        std::mem::size_of::<f64>(),
        &shape,
        &strides,
        0,
    );
    let prepared = owner.as_ref().prepare_device_access(request).unwrap();
    assert_eq!(
        prepared
            .as_any()
            .downcast_ref::<PreparedMarker>()
            .unwrap()
            .0,
        24
    );
    owner.as_mut().prepare_device_access(request).unwrap();
    assert_eq!(prepares.load(Ordering::Relaxed), 2);

    // A provider root has no host representation or default byte mapping.
    assert!(matches!(
        owner.as_ref().host_slice::<f64>(span, DType::F64),
        Err(AccessError::Unsupported { backend: "backend" })
    ));
    assert!(matches!(
        owner.as_mut().host_slice_mut::<f64>(span, DType::F64),
        Err(AccessError::Unsupported { backend: "backend" })
    ));
    assert!(matches!(
        owner.as_ref().map_read(span, DType::F64),
        Err(AccessError::Unsupported {
            backend: "unimplemented"
        })
    ));
    assert!(matches!(
        owner.as_mut().map_write(span, DType::F64),
        Err(AccessError::Unsupported {
            backend: "unimplemented"
        })
    ));
}

#[test]
fn backend_roots_refuse_host_export_and_return_the_owner_unchanged() {
    let owner = import_provider(ProviderBuffer::identified("test-provider", 5));
    let (mut owner, error) = match owner.into_host_vec::<f64>() {
        Ok(_) => panic!("a provider root has no host vector"),
        Err(rejected) => rejected,
    };
    assert!(matches!(
        error,
        AccessError::Unsupported {
            backend: "non-host"
        }
    ));
    assert_eq!(owner.provider_kind(), ProviderKind::Other("test-provider"));

    let recycler: Arc<dyn HostBufferRecycler<f64>> = Arc::new(RecordingRecycler::default());
    assert!(matches!(
        owner.set_host_recycler(Arc::downgrade(&recycler)),
        Err(AccessError::Unsupported {
            backend: "non-host"
        })
    ));

    let pin = owner.into_root_pin();
    assert!(matches!(pin, RootResourcePin::Backend(_)));
    assert!(pin.backend_allocation().is_some());
}

#[test]
fn host_root_pin_reports_host_provider_metadata() {
    let owner = import_host_vec(vec![1.0_f64, 2.0, 3.0]).unwrap();
    let extent = owner.root_identity().extent();
    let pin = owner.into_root_pin();
    assert!(pin.backend_allocation().is_none());
    let RootResourcePin::HostF64(allocation) = &pin else {
        panic!("an f64 host vector must pin inline as HostF64");
    };
    assert_eq!(allocation.root_extent(), extent);
    assert_eq!(allocation.provider_kind(), ProviderKind::Cpu);
    assert_eq!(allocation.capabilities(), ProviderCapabilities::host());
    assert!(allocation.as_any().is::<HostAllocation<f64>>());
    let debug = format!("{allocation:?}");
    assert!(debug.contains("HostAllocation"), "{debug}");
    assert!(debug.contains("element_count: 3"), "{debug}");
}

#[derive(Debug, Default)]
struct RecordingRecycler(Mutex<Vec<Vec<f64>>>);

impl HostBufferRecycler<f64> for RecordingRecycler {
    fn recycle(&self, data: Vec<f64>) {
        self.0.lock().unwrap().push(data);
    }
}

#[test]
fn host_recycler_receives_the_vector_only_when_the_owner_is_destroyed() {
    let recycler = Arc::new(RecordingRecycler::default());
    let as_dyn: Arc<dyn HostBufferRecycler<f64>> = recycler.clone();

    let mut owner = import_host_vec(vec![1.0_f64, 2.0]).unwrap();
    assert!(matches!(
        owner.set_host_recycler::<f32>(Arc::downgrade(
            &(Arc::new(NoopRecycler) as Arc<dyn HostBufferRecycler<f32>>)
        )),
        Err(AccessError::Unsupported {
            backend: "non-host"
        })
    ));
    owner.set_host_recycler(Arc::downgrade(&as_dyn)).unwrap();
    assert!(recycler.0.lock().unwrap().is_empty());
    drop(owner);
    assert_eq!(*recycler.0.lock().unwrap(), vec![vec![1.0, 2.0]]);

    // Exporting the vector detaches the recycler, so nothing is recycled.
    let mut owner = import_host_vec(vec![3.0_f64]).unwrap();
    owner.set_host_recycler(Arc::downgrade(&as_dyn)).unwrap();
    assert_eq!(
        owner
            .into_host_vec::<f64>()
            .map_err(|(_, error)| error)
            .unwrap(),
        vec![3.0]
    );
    assert_eq!(recycler.0.lock().unwrap().len(), 1);

    // A recycler that is already gone is skipped on destruction.
    let short_lived: Arc<dyn HostBufferRecycler<f64>> = Arc::new(RecordingRecycler::default());
    let mut owner = import_host_vec(vec![4.0_f64]).unwrap();
    owner
        .set_host_recycler(Arc::downgrade(&short_lived))
        .unwrap();
    drop(short_lived);
    drop(owner);
    assert_eq!(recycler.0.lock().unwrap().len(), 1);
}

#[derive(Debug)]
struct NoopRecycler;

impl HostBufferRecycler<f32> for NoopRecycler {
    fn recycle(&self, _data: Vec<f32>) {}
}
