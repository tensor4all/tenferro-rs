// Provider-facing ownership evidence for the combined CUDA/WebGPU crate.
//
// These tests are deliberately hardware-gated: the provider contract is about
// the real runtime allocation domain and explicit device duplication, not a
// host-only mock of a backend handle.

#[cfg(feature = "cuda")]
mod cuda {
    use super::engine_run::EngineRun;
    use tenferro_gpu::{
        cuda::download_tensor, cuda::gpu_available, cuda::upload_tensor, cuda::CudaEngine,
        cuda::CudaDeviceId,
    };
    use tenferro_tensor::{
        AllocationDomainId, AllocationId, DType, Tensor, TensorRead, TensorStructural, TensorView,
    };

    fn identity(tensor: &Tensor) -> (Option<AllocationDomainId>, Option<AllocationId>) {
        let Some(tensor) = tensor.as_typed::<f32>() else {
            panic!("provider test uses f32 tensors")
        };
        (tensor.allocation_domain(), tensor.allocation_id())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn explicit_duplicate_has_new_identity_in_the_same_cuda_domain() {
        assert!(gpu_available(), "CUDA test requires an available device");
        let mut backend = CudaEngine::open(CudaDeviceId::from_ordinal(0)).unwrap();
        let host = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
        let input = upload_tensor(backend.runtime(), &host).unwrap();
        let (domain, allocation) = identity(&input);

        let duplicate = backend
            .run(|session| session.cast(&input, DType::F32)).unwrap()
            .unwrap();

        let (duplicate_domain, duplicate_allocation) = identity(&duplicate);
        assert_eq!(duplicate_domain, domain);
        assert_ne!(duplicate_allocation, allocation);
        // A fresh identity alone would also pass for a zero-filled tensor: the copy has to
        // carry the values and the label.
        assert_eq!(duplicate.dtype(), DType::F32);
        assert_eq!(duplicate.shape(), &[2]);
        let back = download_tensor(backend.runtime(), &duplicate).unwrap();
        assert_eq!(back.as_slice::<f32>().unwrap(), &[1.0, 2.0]);
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn cast_between_dtypes_converts_values_and_has_new_identity() {
        assert!(gpu_available(), "CUDA test requires an available device");
        let mut backend = CudaEngine::open(CudaDeviceId::from_ordinal(0)).unwrap();
        let host = Tensor::from_vec_col_major(vec![3], vec![1.5_f32, -2.25, 3.0]).unwrap();
        let input = upload_tensor(backend.runtime(), &host).unwrap();
        let (domain, allocation) = identity(&input);

        let converted = backend
            .run(|session| session.cast(&input, DType::F64))
            .unwrap()
            .unwrap();

        assert_eq!(converted.dtype(), DType::F64);
        let back = download_tensor(backend.runtime(), &converted).unwrap();
        assert_eq!(back.as_slice::<f64>().unwrap(), &[1.5, -2.25, 3.0]);
        let storage = converted.as_typed::<f64>().expect("f64 payload");
        assert_eq!(storage.allocation_domain(), domain);
        assert_ne!(storage.allocation_id(), allocation);
    }

    /// The non-compact case: a metadata-only transpose view over a device allocation is read into
    /// a compact tensor, and that tensor is then duplicated through `cast`. `cast` itself takes an
    /// owned dense `Tensor`, so a strided layout can only reach it after materialization — the
    /// sequence below is the whole route, and the identity assertion is relative to the
    /// materialized input.
    #[test]
    #[ignore = "requires CUDA"]
    fn a_transposed_view_is_materialized_then_duplicated_with_a_fresh_identity() {
        assert!(gpu_available(), "CUDA test requires an available device");
        let mut backend = CudaEngine::open(CudaDeviceId::from_ordinal(0)).unwrap();
        let data: Vec<f32> = (0..12).map(|index| index as f32).collect();
        let source = upload_tensor(
            backend.runtime(),
            &Tensor::from_vec_col_major(vec![3, 4], data.clone()).unwrap(),
        )
        .unwrap();
        let view = source
            .as_typed::<f32>()
            .expect("f32 payload")
            .backend_region_view(vec![4, 3], vec![3, 1], 0)
            .expect("transposed view");

        let materialized = backend
            .to_contiguous_read(TensorRead::from_view(TensorView::F32(view)))
            .expect("materialize the strided view");
        assert_eq!(materialized.shape(), &[4, 3]);
        let back = download_tensor(backend.runtime(), &materialized).unwrap();
        // Column-major `[4, 3]`: element (row, column) is the source's (column, row).
        let mut expected = Vec::with_capacity(12);
        for column in 0..3 {
            for row in 0..4 {
                expected.push(data[row * 3 + column]);
            }
        }
        assert_eq!(back.as_slice::<f32>().unwrap(), &expected[..]);

        let (source_domain, source_allocation) = identity(&materialized);
        let duplicate = backend
            .run(|session| session.cast(&materialized, DType::F32))
            .unwrap()
            .unwrap();
        let (duplicate_domain, duplicate_allocation) = identity(&duplicate);
        assert_eq!(duplicate_domain, source_domain);
        assert_ne!(duplicate_allocation, source_allocation);
        let copy = download_tensor(backend.runtime(), &duplicate).unwrap();
        assert_eq!(copy.as_slice::<f32>().unwrap(), &expected[..]);
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn cuda_runtime_rejects_a_foreign_allocation_domain_before_binding() {
        assert!(gpu_available(), "CUDA test requires an available device");
        let first = CudaEngine::open(CudaDeviceId::from_ordinal(0)).unwrap();
        let mut second = CudaEngine::open(CudaDeviceId::from_ordinal(0)).unwrap();
        let host = Tensor::from_vec_col_major(vec![1], vec![1.0_f32]).unwrap();
        let input = upload_tensor(first.runtime(), &host).unwrap();

        let error = second
            .run(|session| {
                session.to_contiguous_read(TensorRead::from_tensor(&input))
            }).unwrap()
            .unwrap_err();
        assert!(matches!(error, tenferro_tensor::Error::RuntimeState { .. }));
    }
}

