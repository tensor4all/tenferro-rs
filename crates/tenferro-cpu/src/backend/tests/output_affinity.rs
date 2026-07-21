use super::*;

use tenferro_tensor::backend::{ElementwiseFusionInst, ElementwiseFusionOp};
use tenferro_tensor::{DType, MemoryKind, Placement};

fn remote_domain(selected: CpuDomainId) -> CpuDomainId {
    let candidate = CpuDomainId::new(selected.as_u64().wrapping_add(1));
    if candidate == selected {
        CpuDomainId::new(selected.as_u64().wrapping_sub(1))
    } else {
        candidate
    }
}

fn placed_f64(shape: Vec<usize>, data: Vec<f64>, domain: CpuDomainId) -> Tensor {
    let mut tensor = TypedTensor::<f64>::from_vec_col_major(shape, data).unwrap();
    tensor.set_placement(Placement {
        memory_kind: MemoryKind::UnpinnedHost,
        device: None,
        cpu_affinity: Some(domain),
    });
    Tensor::F64(tensor)
}

#[test]
fn direct_and_session_fresh_outputs_use_the_selected_domain() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let selected = backend.execution_info().domain_id();
    let remote = remote_domain(selected);
    let input = placed_f64(vec![2], vec![1.0, 2.0], remote);

    let direct = backend.neg(&input).unwrap();
    let session = backend
        .with_backend_session(|session| session.exp(&input))
        .unwrap();

    assert_eq!(input.placement().cpu_affinity, Some(remote));
    assert_eq!(direct.placement().cpu_affinity, Some(selected));
    assert_eq!(session.placement().cpu_affinity, Some(selected));
}

#[test]
fn dot_and_fusion_vec_outputs_use_the_selected_domain() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let selected = backend.execution_info().domain_id();
    let remote = remote_domain(selected);
    let lhs = placed_f64(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], remote);
    let rhs = placed_f64(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0], remote);
    let fusion_len = 16 * 1024;
    let fusion_lhs = placed_f64(vec![fusion_len], vec![1.0; fusion_len], remote);
    let fusion_rhs = placed_f64(vec![fusion_len], vec![2.0; fusion_len], remote);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let fusion = ElementwiseFusionPlan::new(
        DType::F64,
        2,
        vec![2, 3],
        vec![
            ElementwiseFusionInst::new(ElementwiseFusionOp::Add, vec![0, 1]),
            ElementwiseFusionInst::new(ElementwiseFusionOp::Multiply, vec![0, 1]),
        ],
    );

    let dot = backend.dot_general(&lhs, &rhs, &config).unwrap();
    let outputs = backend
        .execute_elementwise_fusion(&[&fusion_lhs, &fusion_rhs], &fusion)
        .unwrap()
        .unwrap();

    assert_eq!(dot.placement().cpu_affinity, Some(selected));
    assert_eq!(outputs.len(), 2);
    assert!(outputs
        .iter()
        .all(|output| output.placement().cpu_affinity == Some(selected)));
}

#[test]
fn metadata_only_reshape_and_caller_owned_output_are_not_retagged() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let selected = backend.execution_info().domain_id();
    let remote = remote_domain(selected);
    let input = placed_f64(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], remote);
    let mut output = placed_f64(vec![2, 2], vec![0.0; 4], remote);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    let reshaped = backend.reshape(&input, &[4]).unwrap();
    backend
        .dot_general_read_into(
            TensorRead::from_tensor(&input),
            TensorRead::from_tensor(&input),
            &config,
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap();

    assert_eq!(input.placement().cpu_affinity, Some(remote));
    assert_eq!(reshaped.placement().cpu_affinity, Some(remote));
    assert_eq!(output.placement().cpu_affinity, Some(remote));
}

#[test]
fn direct_tensor_read_reshape_preserves_remote_storage_affinity() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let selected = backend.execution_info().domain_id();
    let remote = remote_domain(selected);
    let input = placed_f64(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], remote);

    let output = backend
        .reshape_read(TensorRead::from_tensor(&input), &[4])
        .unwrap();

    assert_eq!(input.placement().cpu_affinity, Some(remote));
    assert_eq!(output.placement().cpu_affinity, Some(remote));
}

#[test]
fn session_tensor_read_reshape_preserves_remote_storage_affinity() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let selected = backend.execution_info().domain_id();
    let remote = remote_domain(selected);
    let input = placed_f64(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], remote);

    let output = backend
        .with_backend_session(|session| session.reshape_read(TensorRead::from_tensor(&input), &[4]))
        .unwrap();

    assert_eq!(input.placement().cpu_affinity, Some(remote));
    assert_eq!(output.placement().cpu_affinity, Some(remote));
}

#[test]
fn direct_and_session_cpu_noop_transfers_preserve_remote_storage_affinity() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let selected = backend.execution_info().domain_id();
    let remote = remote_domain(selected);
    let input = placed_f64(vec![2], vec![1.0, 2.0], remote);

    let direct_upload = backend.upload_host_tensor(&input).unwrap();
    let direct_download = backend.download_to_host(&input).unwrap();
    let (session_upload, session_download) = backend
        .with_backend_session(|session| -> crate::Result<_> {
            Ok((
                session.upload_host_tensor(&input)?,
                session.download_to_host(&input)?,
            ))
        })
        .unwrap();

    assert_eq!(input.placement().cpu_affinity, Some(remote));
    for output in [
        direct_upload,
        direct_download,
        session_upload,
        session_download,
    ] {
        assert_eq!(output.placement().cpu_affinity, Some(remote));
    }
}

#[test]
fn reshaping_a_borrowed_view_tags_only_the_materialized_output() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let selected = backend.execution_info().domain_id();
    let remote = remote_domain(selected);
    let input = placed_f64(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], remote);
    let Tensor::F64(input_tensor) = &input else {
        unreachable!()
    };
    let view = input_tensor.as_view().transpose_view([1, 0]).unwrap();

    let output = backend
        .reshape_read(
            TensorRead::from_view(tenferro_tensor::TensorView::F64(view)),
            &[4],
        )
        .unwrap();

    assert_eq!(input.placement().cpu_affinity, Some(remote));
    assert_eq!(output.placement().cpu_affinity, Some(selected));
}

#[test]
fn validation_failure_does_not_mutate_or_retag_caller_owned_output() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let selected = backend.execution_info().domain_id();
    let remote = remote_domain(selected);
    let input = placed_f64(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], remote);
    let mut output = placed_f64(vec![2, 2], vec![9.0, 8.0, 7.0, 6.0], remote);
    let invalid = DotGeneralConfig {
        lhs_contracting_dims: vec![2],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    let error = backend
        .dot_general_read_into(
            TensorRead::from_tensor(&input),
            TensorRead::from_tensor(&input),
            &invalid,
            TensorWrite::from_tensor(&mut output),
        )
        .unwrap_err();

    assert!(matches!(error, tenferro_tensor::Error::Validation { .. }));
    assert_eq!(output.placement().cpu_affinity, Some(remote));
    assert_eq!(output.as_slice::<f64>().unwrap(), &[9.0, 8.0, 7.0, 6.0]);
}

#[test]
fn lazy_tensor_value_tags_its_fresh_base() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let selected = backend.execution_info().domain_id();
    let remote = remote_domain(selected);
    let lhs = placed_f64(vec![3, 2], vec![1.0; 6], remote);
    let rhs = placed_f64(vec![4], vec![2.0; 4], remote);
    let Tensor::F64(lhs_tensor) = &lhs else {
        unreachable!()
    };
    let lhs_view = lhs_tensor.as_view().transpose_view([1, 0]).unwrap();

    let value = backend
        .execute_broadcast_multiply_value(
            TensorRead::from_view(tenferro_tensor::TensorView::F64(lhs_view)),
            &[2, 3, 4],
            &[0, 1],
            TensorRead::from_tensor(&rhs),
            &[2, 3, 4],
            &[2],
        )
        .unwrap()
        .unwrap();

    assert!(matches!(value, TensorValue::View(_)));
    let affinity = match value.tensor_read() {
        TensorRead::Tensor(tensor) => tensor.placement().cpu_affinity,
        TensorRead::View(view) => match view {
            tenferro_tensor::TensorView::F32(view) => view.placement().cpu_affinity,
            tenferro_tensor::TensorView::F64(view) => view.placement().cpu_affinity,
            tenferro_tensor::TensorView::I32(view) => view.placement().cpu_affinity,
            tenferro_tensor::TensorView::I64(view) => view.placement().cpu_affinity,
            tenferro_tensor::TensorView::Bool(view) => view.placement().cpu_affinity,
            tenferro_tensor::TensorView::C32(view) => view.placement().cpu_affinity,
            tenferro_tensor::TensorView::C64(view) => view.placement().cpu_affinity,
        },
    };
    assert_eq!(affinity, Some(selected));
    assert_eq!(lhs.placement().cpu_affinity, Some(remote));
    assert_eq!(rhs.placement().cpu_affinity, Some(remote));
}

#[test]
fn fresh_tagging_preserves_device_and_memory_kind_fields() {
    let mut tensor = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    let device = tenferro_tensor::DeviceId {
        kind: tenferro_tensor::DeviceKind::Other("fixture".to_owned()),
        ordinal: 3,
    };
    tensor.set_placement(Placement {
        memory_kind: MemoryKind::Other("fixture-memory".to_owned()),
        device: Some(device.clone()),
        cpu_affinity: None,
    });
    let mut tensor = Tensor::F64(tensor);

    tag_fresh_output(&mut tensor, CpuDomainId::new(11));

    assert_eq!(
        tensor.placement().memory_kind,
        MemoryKind::Other("fixture-memory".to_owned())
    );
    assert_eq!(tensor.placement().device, Some(device));
    assert_eq!(tensor.placement().cpu_affinity, Some(CpuDomainId::new(11)));
}
