use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::error::Error as _;
use std::hash::{Hash, Hasher};

use tenferro_tensor::BoxError;

use crate::cubecl::identity::{CudaComputeCapability, CudaDeviceUuid};
use crate::cubecl::device::{
    discover_with, unavailable_device_error, CudaDeviceError, CudaDeviceId, CudaDeviceInfo,
    DiscoveryDriver,
};

fn test_uuid(ordinal: u32) -> CudaDeviceUuid {
    CudaDeviceUuid::from_bytes([ordinal as u8; 16])
}

fn test_info(id: CudaDeviceId, name: &str) -> CudaDeviceInfo {
    CudaDeviceInfo::new(
        id,
        name,
        test_uuid(id.ordinal()),
        CudaComputeCapability { major: 9, minor: 0 },
        40 * 1024 * 1024 * 1024,
    )
}

#[derive(Copy, Clone)]
enum FakeDriverScenario {
    Success,
    InitializeFailure,
    CountFailure,
    NameFailure(CudaDeviceId),
}

struct FakeDriver {
    names: Vec<String>,
    scenario: FakeDriverScenario,
    calls: RefCell<Vec<&'static str>>,
    attempted_ordinals: RefCell<Vec<CudaDeviceId>>,
}

impl FakeDriver {
    fn new(names: Vec<String>, scenario: FakeDriverScenario) -> Self {
        Self {
            names,
            scenario,
            calls: RefCell::new(Vec::new()),
            attempted_ordinals: RefCell::new(Vec::new()),
        }
    }

    fn success(names: Vec<String>) -> Self {
        Self::new(names, FakeDriverScenario::Success)
    }

    fn calls(&self) -> Vec<&'static str> {
        self.calls.borrow().clone()
    }

    fn attempted_ordinals(&self) -> Vec<CudaDeviceId> {
        self.attempted_ordinals.borrow().clone()
    }
}

impl DiscoveryDriver for FakeDriver {
    fn initialize(&self) -> Result<(), BoxError> {
        self.calls.borrow_mut().push("initialize");
        match self.scenario {
            FakeDriverScenario::InitializeFailure => {
                Err(Box::new(std::io::Error::other("fake initialize failure")))
            }
            FakeDriverScenario::Success
            | FakeDriverScenario::CountFailure
            | FakeDriverScenario::NameFailure(_) => Ok(()),
        }
    }

    fn device_count(&self) -> Result<u32, BoxError> {
        self.calls.borrow_mut().push("device_count");
        match self.scenario {
            FakeDriverScenario::CountFailure => {
                Err(Box::new(std::io::Error::other("fake device count failure")))
            }
            FakeDriverScenario::Success
            | FakeDriverScenario::InitializeFailure
            | FakeDriverScenario::NameFailure(_) => Ok(self.names.len() as u32),
        }
    }

    fn device_name(&self, device: CudaDeviceId) -> Result<String, BoxError> {
        self.calls.borrow_mut().push("device_name");
        self.attempted_ordinals.borrow_mut().push(device);
        if matches!(self.scenario, FakeDriverScenario::NameFailure(failed) if failed == device)
        {
            return Err(Box::new(std::io::Error::other("fake device name failure")));
        }
        Ok(self.names[device.ordinal() as usize].clone())
    }

    fn device_uuid(&self, device: CudaDeviceId) -> Result<CudaDeviceUuid, BoxError> {
        self.calls.borrow_mut().push("device_uuid");
        if matches!(self.scenario, FakeDriverScenario::NameFailure(failed) if failed == device)
        {
            return Err(Box::new(std::io::Error::other("fake device uuid failure")));
        }
        Ok(test_uuid(device.ordinal()))
    }

    fn compute_capability(
        &self,
        device: CudaDeviceId,
    ) -> Result<CudaComputeCapability, BoxError> {
        self.calls.borrow_mut().push("compute_capability");
        if matches!(self.scenario, FakeDriverScenario::NameFailure(failed) if failed == device)
        {
            return Err(Box::new(std::io::Error::other(
                "fake compute capability failure",
            )));
        }
        Ok(CudaComputeCapability { major: 9, minor: 0 })
    }

    fn total_memory_bytes(&self, device: CudaDeviceId) -> Result<u64, BoxError> {
        self.calls.borrow_mut().push("total_memory_bytes");
        if matches!(self.scenario, FakeDriverScenario::NameFailure(failed) if failed == device)
        {
            return Err(Box::new(std::io::Error::other("fake total memory failure")));
        }
        Ok(40 * 1024 * 1024 * 1024)
    }
}

fn assert_cuda_device_id_traits<T>()
where
    T: Copy + Clone + Eq + PartialEq + Ord + PartialOrd + Hash,
{
}

#[test]
fn cuda_device_id_has_value_semantics_and_deterministic_debug() {
    const ID: CudaDeviceId = CudaDeviceId::from_ordinal(7);

    assert_cuda_device_id_traits::<CudaDeviceId>();
    assert_eq!(ID.ordinal(), 7);
    assert!(ID < CudaDeviceId::from_ordinal(8));
    assert_eq!(format!("{ID:?}"), "CudaDeviceId(7)");

    let mut first_hasher = DefaultHasher::new();
    ID.hash(&mut first_hasher);
    let mut second_hasher = DefaultHasher::new();
    CudaDeviceId::from_ordinal(7).hash(&mut second_hasher);
    assert_eq!(first_hasher.finish(), second_hasher.finish());
}

#[test]
fn cuda_device_info_exposes_id_and_metadata() {
    let id = CudaDeviceId::from_ordinal(2);
    let info = test_info(id, "NVIDIA H100");

    assert_eq!(info.id(), id);
    assert_eq!(info.name(), "NVIDIA H100");
    assert_eq!(info.uuid(), test_uuid(2));
    assert_eq!(info.compute_capability().major, 9);
    assert_eq!(info.compute_capability().minor, 0);
    assert_eq!(info.total_memory_bytes(), 40 * 1024 * 1024 * 1024);
    assert_eq!(info, info.clone());
}

#[test]
fn unavailable_device_selection_preserves_requested_id_and_discovered_records() {
    let requested = CudaDeviceId::from_ordinal(2);
    let discovered = vec![
        test_info(CudaDeviceId::from_ordinal(0), "NVIDIA H100"),
        test_info(CudaDeviceId::from_ordinal(1), "NVIDIA A100"),
    ];

    let error = unavailable_device_error(requested, discovered.clone());

    assert!(matches!(
        error,
        CudaDeviceError::Unavailable {
            requested: actual_requested,
            discovered: actual_discovered,
        } if actual_requested == requested && actual_discovered.as_ref() == discovered
    ));
}

#[test]
fn discovery_of_zero_devices_returns_empty() {
    let driver = FakeDriver::success(Vec::new());

    assert!(discover_with(&driver).unwrap().is_empty());
}

#[test]
fn discovery_preserves_ordinal_order_and_is_deterministic() {
    let driver = FakeDriver::success(vec!["NVIDIA A100".into(), "NVIDIA H100".into()]);
    let expected = vec![
        test_info(CudaDeviceId::from_ordinal(0), "NVIDIA A100"),
        test_info(CudaDeviceId::from_ordinal(1), "NVIDIA H100"),
    ];

    let first = discover_with(&driver).unwrap();
    let second = discover_with(&driver).unwrap();

    assert_eq!(first, expected);
    assert_eq!(first, second);
    assert_eq!(format!("{first:?}"), format!("{second:?}"));
}

#[test]
fn discovery_initialize_failure_returns_provider_neutral_error() {
    let driver = FakeDriver::new(Vec::new(), FakeDriverScenario::InitializeFailure);

    let error = discover_with(&driver).expect_err("initialize failure should be returned");

    assert!(matches!(
        &error,
        CudaDeviceError::Discovery {
            operation: "initialize_driver",
            source,
        } if source.downcast_ref::<std::io::Error>().is_some()
            && source.to_string() == "fake initialize failure"
    ));
    assert_eq!(
        error.source().map(ToString::to_string).as_deref(),
        Some("fake initialize failure")
    );
    assert_eq!(driver.calls(), vec!["initialize"]);
}

#[test]
fn discovery_count_failure_returns_provider_neutral_error() {
    let driver = FakeDriver::new(vec!["NVIDIA A100".into()], FakeDriverScenario::CountFailure);

    let error = discover_with(&driver).expect_err("count failure should be returned");

    assert!(matches!(
        &error,
        CudaDeviceError::Discovery {
            operation: "enumerate_devices",
            source,
        } if source.downcast_ref::<std::io::Error>().is_some()
            && source.to_string() == "fake device count failure"
    ));
    assert_eq!(
        error.source().map(ToString::to_string).as_deref(),
        Some("fake device count failure")
    );
    assert_eq!(driver.calls(), vec!["initialize", "device_count"]);
}

#[test]
fn discovery_name_failure_returns_error_without_partial_devices() {
    let failed_device = CudaDeviceId::from_ordinal(1);
    let driver = FakeDriver::new(
        vec!["NVIDIA A100".into(), "NVIDIA H100".into()],
        FakeDriverScenario::NameFailure(failed_device),
    );

    let error = discover_with(&driver).expect_err("name failure should be returned");

    assert!(matches!(
        &error,
        CudaDeviceError::Discovery {
            operation: "get_device_name",
            source,
        } if source.downcast_ref::<std::io::Error>().is_some()
            && source.to_string() == "fake device name failure"
    ));
    assert_eq!(
        error.source().map(ToString::to_string).as_deref(),
        Some("fake device name failure")
    );
    assert_eq!(
        driver.attempted_ordinals(),
        vec![CudaDeviceId::from_ordinal(0), failed_device]
    );
    assert_eq!(
        driver.calls(),
        vec![
            "initialize",
            "device_count",
            "device_name",
            "device_uuid",
            "compute_capability",
            "total_memory_bytes",
            "device_name",
        ]
    );
}

#[test]
fn cuda_device_error_discovery_preserves_fields_and_source() {
    let error = CudaDeviceError::Discovery {
        operation: "enumerate_devices",
        source: Box::new(std::io::Error::other("driver query failed")),
    };

    assert!(matches!(
        &error,
        CudaDeviceError::Discovery {
            operation: "enumerate_devices",
            source,
        } if source.to_string() == "driver query failed"
    ));
    assert_eq!(
        error.to_string(),
        "CUDA device discovery failed during enumerate_devices: driver query failed"
    );
    assert_eq!(error.operation(), Some("enumerate_devices"));
    assert_eq!(error.requested(), None);
    assert_eq!(error.discovered(), None);
    assert_eq!(error.device(), None);
    assert_eq!(
        error.source().map(ToString::to_string).as_deref(),
        Some("driver query failed")
    );
}

#[test]
fn cuda_device_error_unavailable_preserves_fields_without_source() {
    let requested = CudaDeviceId::from_ordinal(2);
    let discovered = vec![
        test_info(CudaDeviceId::from_ordinal(0), "NVIDIA H100"),
        test_info(CudaDeviceId::from_ordinal(1), "NVIDIA A100"),
    ]
    .into_boxed_slice();
    let error = CudaDeviceError::Unavailable {
        requested,
        discovered,
    };

    assert!(matches!(
        &error,
        CudaDeviceError::Unavailable {
            requested: actual_requested,
            discovered: actual_discovered,
        } if *actual_requested == requested
            && actual_discovered.as_ref() == [
                test_info(CudaDeviceId::from_ordinal(0), "NVIDIA H100"),
                test_info(CudaDeviceId::from_ordinal(1), "NVIDIA A100"),
            ]
    ));
    assert!(error.to_string().contains(
        "requested CUDA device CudaDeviceId(2) is unavailable; discovered devices: ["
    ));
    assert!(error.to_string().contains(
        "CudaDeviceInfo { id: CudaDeviceId(0), name: \"NVIDIA H100\", uuid: CudaDeviceUuid([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), compute_capability: CudaComputeCapability { major: 9, minor: 0 }, total_memory_bytes: 42949672960 }"
    ));
    assert_eq!(error.operation(), None);
    assert_eq!(error.requested(), Some(requested));
    assert_eq!(
        error.discovered(),
        Some(
            [
                test_info(CudaDeviceId::from_ordinal(0), "NVIDIA H100"),
                test_info(CudaDeviceId::from_ordinal(1), "NVIDIA A100"),
            ]
            .as_slice()
        )
    );
    assert_eq!(error.device(), None);
    assert!(error.source().is_none());
}

#[test]
fn cuda_device_error_initialization_preserves_fields_and_source() {
    let device = CudaDeviceId::from_ordinal(1);
    let error = CudaDeviceError::Initialization {
        device,
        operation: "create_client",
        source: Box::new(std::io::Error::other("CUDA context failed")),
    };

    assert!(matches!(
        &error,
        CudaDeviceError::Initialization {
            device: actual_device,
            operation: "create_client",
            source,
        } if *actual_device == device && source.to_string() == "CUDA context failed"
    ));
    assert_eq!(
        error.to_string(),
        "CUDA device CudaDeviceId(1) initialization failed during create_client: CUDA context failed"
    );
    assert_eq!(error.operation(), Some("create_client"));
    assert_eq!(error.requested(), None);
    assert_eq!(error.discovered(), None);
    assert_eq!(error.device(), Some(device));
    assert_eq!(
        error.source().map(ToString::to_string).as_deref(),
        Some("CUDA context failed")
    );
}
