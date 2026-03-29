use super::support::{diag_f64, vector_f64};
use num_complex::Complex32;
use num_complex::Complex64;
use tenferro::{forward_ad, set_default_runtime, snapshot, RuntimeContext, ScalarValue, Tensor};
use tenferro_dynamic_compute as dynamic_compute;
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

fn tensor2(values: &[f64], d0: usize, d1: usize) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(values, &[d0, d1], MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn primal_snapshot_returns_public_snapshot_boundary() {
    let x = Tensor::from_tensor(tensor2(&[1.0, 2.0, 3.0, 4.0], 2, 2));
    let snapshot = x.primal_snapshot();

    match snapshot {
        snapshot::DynTensor::F64(value) => {
            assert_eq!(value.logical_dims(), &[2, 2]);
            assert_eq!(value.axis_classes(), &[0, 1]);
            assert!(value.is_dense());
        }
        other => panic!("expected f64 snapshot, got {other:?}"),
    }
}

#[test]
fn primal_snapshot_preserves_structured_layout_and_to_dense_materializes() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let diag = diag_f64(&[2.0, 3.0]);
    let snapshot = diag.primal_snapshot();

    assert_eq!(snapshot.scalar_type(), tenferro::ScalarType::F64);
    assert_eq!(snapshot.dims(), &[2, 2]);
    assert_eq!(snapshot.axis_classes(), &[0, 0]);
    assert!(snapshot.is_diag());
    assert!(!snapshot.is_dense());

    let dense = snapshot.to_dense().unwrap();
    assert!(dense.is_dense());
    assert_eq!(dense.axis_classes(), &[0, 1]);
    assert_eq!(dense.dims(), &[2, 2]);

    let payload = dense.payload_f64().unwrap().buffer().as_slice().unwrap();
    assert_eq!(payload, &[2.0, 0.0, 0.0, 3.0]);
}

#[test]
fn try_scalar_value_returns_dynamic_scalar_without_casting() {
    let x = Tensor::from_tensor(
        DenseTensor::<Complex64>::from_slice(
            &[Complex64::new(2.0, -3.0)],
            &[],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
    );

    let value = x.try_scalar_value().unwrap();
    assert_eq!(value, ScalarValue::C64(Complex64::new(2.0, -3.0)));
}

#[test]
fn try_scalar_value_rejects_non_rank0_tensors() {
    let x = Tensor::from_tensor(vector_f64(&[1.0, 2.0]));
    let err = x.try_scalar_value().unwrap_err();
    let message = match err {
        tenferro::Error::InvalidAdTensor { message } => message,
        other => panic!("expected InvalidAdTensor, got {other:?}"),
    };
    assert!(message.contains("rank-0"));
}

#[test]
fn detach_remains_compute_tensor_while_snapshot_is_export_boundary() {
    let mut x = Tensor::from_tensor(
        DenseTensor::<f64>::from_slice(&[5.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    );
    x.set_requires_grad(true).unwrap();

    let detached: dynamic_compute::Tensor = x.detach();
    assert_eq!(
        detached
            .as_f64()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[5.0]
    );

    let snapshot = x.primal_snapshot();
    assert_eq!(snapshot.scalar_type(), tenferro::ScalarType::F64);
    assert!(snapshot.dims().is_empty());
}

#[test]
fn try_scalar_value_and_primal_snapshot_cover_remaining_runtime_variants_and_modes() {
    let mut reverse_f32 = Tensor::from_tensor(
        DenseTensor::<f32>::from_slice(&[1.5_f32], &[], MemoryOrder::ColumnMajor).unwrap(),
    );
    reverse_f32.set_requires_grad(true).unwrap();
    let reverse_snapshot = reverse_f32.primal_snapshot();
    assert!(matches!(reverse_snapshot, snapshot::DynTensor::F32(_)));
    assert_eq!(
        reverse_f32.try_scalar_value().unwrap(),
        ScalarValue::F32(1.5_f32)
    );

    let (forward_scalar, tangent) = forward_ad::dual_level(|fw| {
        let primal = Tensor::from_tensor(
            DenseTensor::<Complex32>::from_slice(
                &[Complex32::new(2.0, -1.0)],
                &[],
                MemoryOrder::ColumnMajor,
            )
            .unwrap(),
        );
        let tangent = Tensor::from_tensor(
            DenseTensor::<Complex32>::from_slice(
                &[Complex32::new(0.5, 0.25)],
                &[],
                MemoryOrder::ColumnMajor,
            )
            .unwrap(),
        );
        let dual = fw.make_dual(&primal, &tangent)?;
        fw.unpack_dual(&dual)
    })
    .unwrap();

    assert!(matches!(
        forward_scalar.primal_snapshot(),
        snapshot::DynTensor::C32(_)
    ));
    assert_eq!(
        forward_scalar.try_scalar_value().unwrap(),
        ScalarValue::C32(Complex32::new(2.0, -1.0))
    );
    assert_eq!(
        tangent.unwrap().try_scalar_value().unwrap(),
        ScalarValue::C32(Complex32::new(0.5, 0.25))
    );
}

#[test]
fn primal_mode_scalar_snapshots_cover_all_runtime_variants() {
    let f32 = Tensor::from_tensor(
        DenseTensor::<f32>::from_slice(&[2.5_f32], &[], MemoryOrder::ColumnMajor).unwrap(),
    );
    assert!(matches!(f32.primal_snapshot(), snapshot::DynTensor::F32(_)));
    assert_eq!(f32.try_scalar_value().unwrap(), ScalarValue::F32(2.5_f32));

    let f64 = Tensor::from_tensor(
        DenseTensor::<f64>::from_slice(&[-4.0_f64], &[], MemoryOrder::ColumnMajor).unwrap(),
    );
    assert!(matches!(f64.primal_snapshot(), snapshot::DynTensor::F64(_)));
    assert_eq!(f64.try_scalar_value().unwrap(), ScalarValue::F64(-4.0_f64));

    let c32 = Tensor::from_tensor(
        DenseTensor::<Complex32>::from_slice(
            &[Complex32::new(-1.0, 0.5)],
            &[],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
    );
    assert!(matches!(c32.primal_snapshot(), snapshot::DynTensor::C32(_)));
    assert_eq!(
        c32.try_scalar_value().unwrap(),
        ScalarValue::C32(Complex32::new(-1.0, 0.5))
    );

    let c64 = Tensor::from_tensor(
        DenseTensor::<Complex64>::from_slice(
            &[Complex64::new(1.25, -0.75)],
            &[],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
    );
    assert!(matches!(c64.primal_snapshot(), snapshot::DynTensor::C64(_)));
    assert_eq!(
        c64.try_scalar_value().unwrap(),
        ScalarValue::C64(Complex64::new(1.25, -0.75))
    );
}
