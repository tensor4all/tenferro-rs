// Provider-facing ownership evidence for the combined CUDA/WebGPU crate.
//
// These tests are deliberately hardware-gated: the provider contract is about
// the real runtime allocation domain and explicit device duplication, not a
// host-only mock of a backend handle.

#[cfg(feature = "cuda")]
mod cuda {
    use tenferro_gpu::{cuda::gpu_available, cuda::upload_tensor, cuda::CudaBackend, cuda::CudaDeviceId};
    use tenferro_tensor::{
        AllocationDomainId, AllocationId, DType, Tensor, TensorRead, TensorStructural,
    };

    fn identity(tensor: &Tensor) -> (Option<AllocationDomainId>, Option<AllocationId>) {
        let Tensor::F32(tensor) = tensor else {
            panic!("provider test uses f32 tensors")
        };
        (tensor.allocation_domain(), tensor.allocation_id())
    }

    #[test]
    fn explicit_duplicate_has_new_identity_in_the_same_cuda_domain() {
        if !gpu_available() {
            return;
        }
        let mut backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
        let host = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
        let input = upload_tensor(backend.runtime(), &host).unwrap();
        let (domain, allocation) = identity(&input);

        let duplicate = backend.cast(&input, DType::F32).unwrap();

        let (duplicate_domain, duplicate_allocation) = identity(&duplicate);
        assert_eq!(duplicate_domain, domain);
        assert_ne!(duplicate_allocation, allocation);
    }

    #[test]
    fn cuda_runtime_rejects_a_foreign_allocation_domain_before_binding() {
        if !gpu_available() {
            return;
        }
        let first = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
        let mut second = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
        let host = Tensor::from_vec_col_major(vec![1], vec![1.0_f32]).unwrap();
        let input = upload_tensor(first.runtime(), &host).unwrap();

        let error = second
            .to_contiguous_read(TensorRead::from_tensor(&input))
            .unwrap_err();
        assert!(matches!(error, tenferro_tensor::Error::RuntimeState { .. }));
    }
}

#[cfg(feature = "webgpu")]
mod webgpu {
    use tenferro_gpu::{webgpu::upload_webgpu_tensor, webgpu::WebGpuBackend};
    use tenferro_tensor::{AllocationDomainId, AllocationId, Tensor, TensorRead, TensorStructural};

    fn identity(tensor: &Tensor) -> (Option<AllocationDomainId>, Option<AllocationId>) {
        let Tensor::F32(tensor) = tensor else {
            panic!("provider test uses f32 tensors")
        };
        (tensor.allocation_domain(), tensor.allocation_id())
    }

    #[test]
    fn explicit_materialization_has_new_identity_in_the_same_webgpu_domain() {
        let Ok(mut backend) = WebGpuBackend::new_default() else {
            return;
        };
        let host = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
        let input = upload_webgpu_tensor(backend.runtime(), &host).unwrap();
        let (domain, allocation) = identity(&input);

        let duplicate = backend
            .to_contiguous_read(TensorRead::from_tensor(&input))
            .unwrap();

        let (duplicate_domain, duplicate_allocation) = identity(&duplicate);
        assert_eq!(duplicate_domain, domain);
        assert_ne!(duplicate_allocation, allocation);
    }

    #[test]
    fn webgpu_runtime_rejects_a_foreign_allocation_domain_before_binding() {
        let Ok(first) = WebGpuBackend::new_default() else {
            return;
        };
        let Ok(mut second) = WebGpuBackend::new_default() else {
            return;
        };
        let host = Tensor::from_vec_col_major(vec![1], vec![1.0_f32]).unwrap();
        let input = upload_webgpu_tensor(first.runtime(), &host).unwrap();

        let error = second
            .to_contiguous_read(TensorRead::from_tensor(&input))
            .unwrap_err();
        assert!(matches!(error, tenferro_tensor::Error::HostAccess { .. }));
    }
}
