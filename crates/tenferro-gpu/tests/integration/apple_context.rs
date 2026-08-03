#![cfg(all(feature = "webgpu", target_os = "macos"))]

use tenferro_gpu::{AppleContext, AppleTransferStats};
use tenferro_tensor::{HostAccessError, StorageBuffer, Tensor, TensorDot, TypedTensor};

fn apple_context() -> Option<AppleContext> {
    match AppleContext::new() {
        Ok(context) => Some(context),
        Err(error) => {
            eprintln!("skipping Apple context test: {error}");
            None
        }
    }
}

#[test]
fn independent_contexts_reject_foreign_managed_allocations() {
    let (Some(first), Some(second)) = (apple_context(), apple_context()) else {
        return;
    };
    assert_ne!(first.domain_id(), second.domain_id());

    let host = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
    let managed = first.upload_tensor(&host).unwrap();
    let error = second.download_tensor(&managed).unwrap_err();
    assert!(matches!(
        error,
        tenferro_tensor::Error::HostAccess {
            source: HostAccessError::ForeignDomain { .. },
            ..
        }
    ));
}

#[test]
fn managed_upload_maps_without_post_creation_transfers_and_keeps_identity() {
    let Some(context) = apple_context() else {
        return;
    };
    let host = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
    let managed = context.upload_tensor(&host).unwrap();
    assert_eq!(
        context.transfer_stats(),
        AppleTransferStats {
            uploaded_bytes: 8,
            downloaded_bytes: 0,
        }
    );

    let Tensor::F32(typed) = &managed else {
        panic!("expected f32 tensor")
    };
    assert_eq!(typed.allocation_domain(), Some(context.domain_id()));
    let allocation = typed.allocation_id().unwrap();
    let StorageBuffer::Backend(buffer) = typed.buffer() else {
        panic!("expected managed backend buffer")
    };
    assert_eq!(&*buffer.map_read().unwrap(), &[1.0, 2.0]);
    {
        let mut write = buffer.map_write().unwrap();
        write.copy_from_slice(&[3.0, 2.0]).unwrap();
        assert!(matches!(
            buffer.map_read(),
            Err(HostAccessError::OverlappingHostMapping)
        ));
    }
    assert_eq!(&*buffer.map_read().unwrap(), &[3.0, 2.0]);
    assert_eq!(typed.allocation_id(), Some(allocation));
    assert_eq!(context.transfer_stats().downloaded_bytes, 0);

    let downloaded = context.download_tensor(&managed).unwrap();
    assert_eq!(downloaded.as_slice::<f32>().unwrap(), &[3.0, 2.0]);
    assert_eq!(context.transfer_stats().downloaded_bytes, 8);
}

#[test]
fn cpu_domain_allocator_produces_write_only_managed_outputs_without_transfers() {
    let Some(context) = apple_context() else {
        return;
    };
    let domain = context.cpu_backend().shared_allocation_domain().unwrap();
    let output = domain.allocate(tenferro_tensor::DType::F64, &[2]).unwrap();
    let Tensor::F64(output) = output else {
        panic!("expected f64 output")
    };
    assert_eq!(output.allocation_domain(), Some(context.domain_id()));
    let StorageBuffer::Backend(buffer) = output.buffer() else {
        panic!("expected managed output")
    };
    buffer
        .map_write()
        .unwrap()
        .copy_from_slice(&[5.0, 8.0])
        .unwrap();
    assert_eq!(&*buffer.map_read().unwrap(), &[5.0, 8.0]);
    assert_eq!(context.transfer_stats(), AppleTransferStats::default());
}

#[test]
fn metal_output_stays_in_the_context_domain_without_host_transfers() {
    let Some(context) = apple_context() else {
        return;
    };
    let lhs = Tensor::F32(TypedTensor::from_vec_col_major(vec![1, 1], vec![2.0_f32]).unwrap());
    let rhs = Tensor::F32(TypedTensor::from_vec_col_major(vec![1, 1], vec![3.0_f32]).unwrap());
    let lhs = context.upload_tensor(&lhs).unwrap();
    let rhs = context.upload_tensor(&rhs).unwrap();
    let Tensor::F32(lhs_typed) = &lhs else {
        panic!("expected f32 lhs")
    };
    let lhs_allocation = lhs_typed.allocation_id().unwrap();
    let StorageBuffer::Backend(lhs_buffer) = lhs_typed.buffer() else {
        panic!("expected managed lhs")
    };
    assert_eq!(&*lhs_buffer.map_read().unwrap(), &[2.0]);
    let before = context.transfer_stats();
    let mut metal = context.metal_backend().clone();
    let output = metal
        .dot_general(
            &lhs,
            &rhs,
            &tenferro_tensor::DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        )
        .unwrap();
    metal.synchronize().unwrap();
    assert_eq!(&*lhs_buffer.map_read().unwrap(), &[2.0]);
    assert_eq!(lhs_typed.allocation_id(), Some(lhs_allocation));

    let Tensor::F32(output) = output else {
        panic!("expected f32 output")
    };
    assert_eq!(output.allocation_domain(), Some(context.domain_id()));
    assert_eq!(context.transfer_stats(), before);
    let StorageBuffer::Backend(buffer) = output.buffer() else {
        panic!("expected managed backend output")
    };
    assert_eq!(&*buffer.map_read().unwrap(), &[6.0]);
}
