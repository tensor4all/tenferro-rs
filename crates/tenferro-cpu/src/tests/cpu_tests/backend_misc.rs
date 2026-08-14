use super::*;
use tenferro_tensor::{
    BackendStorageHandle, MemoryKind, Placement, StorageBuffer, TensorViewCanonicalization,
    TypedTensorView, TypedTensorViewMut,
};

fn opaque_backend_placement() -> Placement {
    Placement {
        memory_kind: MemoryKind::Device,
        device: None,
        cpu_affinity: None,
    }
}

#[test]
fn cpu_runtime_materialization_dispatches_all_dtypes_with_backend_session_parity() {
    macro_rules! assert_materialized {
        ($variant:ident, $ty:ty, $values:expr) => {{
            let values: [$ty; 2] = $values;
            let view = TensorView::$variant(
                TypedTensorView::from_slice(vec![2], vec![-1], 1, &values).unwrap(),
            );
            let read = TensorRead::from_view(view);
            let mut backend = CpuBackend::with_threads(2).unwrap();

            let direct = backend.to_contiguous_read(read.clone()).unwrap();
            let session = backend
                .with_backend_session(|exec| exec.to_contiguous_read(read))
                .unwrap();

            let expected = [values[1], values[0]];
            assert_eq!(direct.as_slice::<$ty>().unwrap(), &expected);
            assert_eq!(session.as_slice::<$ty>().unwrap(), &expected);
        }};
    }

    assert_materialized!(F32, f32, [1.25, -2.5]);
    assert_materialized!(F64, f64, [3.5, -4.75]);
    assert_materialized!(I32, i32, [5, -6]);
    assert_materialized!(I64, i64, [7, -8]);
    assert_materialized!(Bool, bool, [true, false]);
    assert_materialized!(
        C32,
        Complex32,
        [Complex32::new(1.0, -2.0), Complex32::new(-3.0, 4.0)]
    );
    assert_materialized!(
        C64,
        Complex64,
        [Complex64::new(5.0, -6.0), Complex64::new(-7.0, 8.0)]
    );
}

#[test]
fn cpu_runtime_materialization_rejects_owned_host_buffer_with_device_placement() {
    let mut input = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    input.set_placement(opaque_backend_placement());
    let input = Tensor::F64(input);
    let mut backend = CpuBackend::new();

    let err = backend
        .to_contiguous_read(TensorRead::from_tensor(&input))
        .unwrap_err();

    assert!(matches!(
        err,
        Error::RuntimeState {
            op: "CpuBackend::to_contiguous_read",
            ref message,
        } if message.contains("source host placement") && message.contains("Device")
    ));
}

#[test]
fn cpu_runtime_materialization_rejects_host_view_with_device_placement() {
    let mut input = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    input.set_placement(opaque_backend_placement());
    let mut backend = CpuBackend::new();

    let err = backend
        .to_contiguous_read(TensorRead::from_view(TensorView::F64(input.as_view())))
        .unwrap_err();

    assert!(matches!(
        err,
        Error::RuntimeState {
            op: "CpuBackend::to_contiguous_read",
            ref message,
        } if message.contains("source host placement") && message.contains("Device")
    ));
}

#[test]
fn cpu_runtime_copy_dispatches_all_dtypes_with_backend_session_parity() {
    macro_rules! assert_copied {
        ($variant:ident, $ty:ty, $values:expr, $zeros:expr) => {{
            let values: Vec<$ty> = $values;
            let src =
                Tensor::$variant(TypedTensor::from_vec_col_major(vec![2], values.clone()).unwrap());
            let mut direct_dst =
                Tensor::$variant(TypedTensor::from_vec_col_major(vec![2], $zeros).unwrap());
            let mut session_dst =
                Tensor::$variant(TypedTensor::from_vec_col_major(vec![2], $zeros).unwrap());
            let mut backend = CpuBackend::with_threads(2).unwrap();

            backend
                .copy_read_into(
                    TensorRead::from_tensor(&src),
                    TensorWrite::from_tensor(&mut direct_dst),
                )
                .unwrap();
            backend
                .with_backend_session(|exec| {
                    exec.copy_read_into(
                        TensorRead::from_tensor(&src),
                        TensorWrite::from_tensor(&mut session_dst),
                    )
                })
                .unwrap();

            assert_eq!(direct_dst.as_slice::<$ty>().unwrap(), values.as_slice());
            assert_eq!(session_dst.as_slice::<$ty>().unwrap(), values.as_slice());
        }};
    }

    assert_copied!(F32, f32, vec![1.25, -2.5], vec![0.0; 2]);
    assert_copied!(F64, f64, vec![3.5, -4.75], vec![0.0; 2]);
    assert_copied!(I32, i32, vec![5, -6], vec![0; 2]);
    assert_copied!(I64, i64, vec![7, -8], vec![0; 2]);
    assert_copied!(Bool, bool, vec![true, false], vec![false; 2]);
    assert_copied!(
        C32,
        Complex32,
        vec![Complex32::new(1.0, -2.0), Complex32::new(-3.0, 4.0)],
        vec![Complex32::new(0.0, 0.0); 2]
    );
    assert_copied!(
        C64,
        Complex64,
        vec![Complex64::new(5.0, -6.0), Complex64::new(-7.0, 8.0)],
        vec![Complex64::new(0.0, 0.0); 2]
    );
}

#[test]
fn cpu_runtime_copy_handles_strided_source_and_destination_without_allocation() {
    let mut backend = CpuBackend::with_threads(2).unwrap();
    backend.reclaim_buffer(Tensor::I32(
        TypedTensor::from_vec_col_major(vec![4], vec![0_i32; 4]).unwrap(),
    ));
    let retained_before = backend.buffer_pool_len().unwrap();
    let src_data = [0_i32, 1, 2, 3, 4, 5, 6, 7];
    let src = TypedTensorView::from_slice(vec![2, 2], vec![2, 4], 1, &src_data).unwrap();
    let mut dst_data = [-1_i32; 8];
    let dst = TypedTensorViewMut::from_slice(vec![2, 2], vec![3, 1], 1, &mut dst_data).unwrap();

    backend
        .copy_read_into(
            TensorRead::from_view(TensorView::I32(src)),
            TensorWrite::from_view(TensorViewMut::I32(dst)),
        )
        .unwrap();

    assert_eq!(dst_data, [-1, 1, 5, -1, 3, 7, -1, -1]);
    assert_eq!(backend.buffer_pool_len().unwrap(), retained_before);
}

#[test]
fn cpu_runtime_copy_reports_dtype_shape_and_placement_errors() {
    let mut backend = CpuBackend::new();

    let src = Tensor::from_vec_col_major(vec![2], vec![1_i32, 2]).unwrap();
    let mut wrong_dtype = Tensor::from_vec_col_major(vec![2], vec![0_i64, 0]).unwrap();
    assert!(matches!(
        backend.copy_read_into(
            TensorRead::from_tensor(&src),
            TensorWrite::from_tensor(&mut wrong_dtype),
        ),
        Err(Error::Validation {
            op: "CpuBackend::copy_read_into",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        })
    ));

    let mut wrong_shape = Tensor::from_vec_col_major(vec![3], vec![0_i32; 3]).unwrap();
    assert!(matches!(
        backend.copy_read_into(
            TensorRead::from_tensor(&src),
            TensorWrite::from_tensor(&mut wrong_shape),
        ),
        Err(Error::Validation {
            op: "CpuBackend::copy_read_into",
            source: tenferro_tensor::ValidationError::ShapeMismatch(_),
        })
    ));

    let mut misplaced = Tensor::from_vec_col_major(vec![2], vec![0_i32; 2]).unwrap();
    match &mut misplaced {
        Tensor::I32(tensor) => tensor.set_placement(opaque_backend_placement()),
        _ => unreachable!(),
    }
    assert!(matches!(
        backend.copy_read_into(
            TensorRead::from_tensor(&src),
            TensorWrite::from_tensor(&mut misplaced),
        ),
        Err(Error::RuntimeState {
            op: "CpuBackend::copy_read_into",
            ref message,
        }) if message.contains("destination") && message.contains("host placement")
    ));
}

#[test]
fn cpu_copy_into_copies_exactly_between_strided_host_views() {
    let mut backend = CpuBackend::with_threads(2).unwrap();
    let src_data = [0_i32, 1, 2, 3, 4, 5, 6, 7];
    let src = TypedTensorView::from_slice(vec![2, 2], vec![2, 4], 1, &src_data).unwrap();
    let mut dst_data = [-1_i32; 8];
    let mut dst = TypedTensorViewMut::from_slice(vec![2, 2], vec![3, 1], 1, &mut dst_data).unwrap();

    backend.copy_into(&src, &mut dst).unwrap();

    assert_eq!(dst_data, [-1, 1, 5, -1, 3, 7, -1, -1]);
}

#[test]
fn cpu_copy_into_reports_shape_mismatch_with_canonical_op_name() {
    let mut backend = CpuBackend::new();
    let src = TypedTensor::<i32>::from_vec_col_major(vec![2], vec![1, 2]).unwrap();
    let mut dst = TypedTensor::<i32>::from_vec_col_major(vec![3], vec![0, 0, 0]).unwrap();

    let err = backend
        .copy_into(&src.as_view(), &mut dst.as_view_mut())
        .unwrap_err();

    assert!(matches!(
        err,
        Error::Validation {
            op: "CpuBackend::copy_into",
            source: tenferro_tensor::ValidationError::ShapeMismatch(_),
        }
    ));
}

#[test]
fn cpu_copy_into_rejects_backend_source_without_download() {
    let mut backend = CpuBackend::new();
    let src = TypedTensor::<f64>::from_buffer_col_major(
        vec![2],
        StorageBuffer::Backend(Box::new(BackendStorageHandle::<f64>::new_with_len(9, 2))),
        opaque_backend_placement(),
    )
    .unwrap();
    let mut dst = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, 0.0]).unwrap();

    let err = backend
        .copy_into(&src.as_view(), &mut dst.as_view_mut())
        .unwrap_err();

    assert!(matches!(
        err,
        Error::RuntimeState {
            op: "CpuBackend::copy_into",
            ref message,
        } if message.contains("download")
    ));
}

#[test]
fn cpu_copy_into_rejects_backend_destination_without_download() {
    let mut backend = CpuBackend::new();
    let src = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    let mut dst = TypedTensor::<f64>::from_buffer_col_major(
        vec![2],
        StorageBuffer::Backend(Box::new(BackendStorageHandle::<f64>::new_with_len(8, 2))),
        opaque_backend_placement(),
    )
    .unwrap();

    let err = backend
        .copy_into(&src.as_view(), &mut dst.as_view_mut())
        .unwrap_err();

    assert!(matches!(
        err,
        Error::RuntimeState {
            op: "CpuBackend::copy_into",
            ref message,
        } if message.contains("download")
    ));
}

#[test]
fn cpu_copy_into_rejects_host_source_with_device_placement() {
    let mut backend = CpuBackend::new();
    let mut src = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    src.set_placement(opaque_backend_placement());
    let mut dst = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, 0.0]).unwrap();

    let err = backend
        .copy_into(&src.as_view(), &mut dst.as_view_mut())
        .unwrap_err();

    assert!(matches!(
        err,
        Error::RuntimeState {
            op: "CpuBackend::copy_into",
            ref message,
        } if message.contains("source") && message.contains("host placement")
    ));
}

#[test]
fn cpu_copy_into_rejects_host_destination_with_device_placement() {
    let mut backend = CpuBackend::new();
    let src = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    let mut dst = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, 0.0]).unwrap();
    dst.set_placement(opaque_backend_placement());

    let err = backend
        .copy_into(&src.as_view(), &mut dst.as_view_mut())
        .unwrap_err();

    assert!(matches!(
        err,
        Error::RuntimeState {
            op: "CpuBackend::copy_into",
            ref message,
        } if message.contains("destination") && message.contains("host placement")
    ));
}

#[test]
fn test_reclaim_buffer_returns_host_buffer_to_pool() {
    let mut backend = CpuBackend::new();
    assert_eq!(backend.buffer_pool_len().unwrap(), 0);
    let t = TensorElementwise::add(
        &mut backend,
        &Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap()),
        &Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![3.0, 4.0]).unwrap()),
    )
    .unwrap();
    backend.reclaim_buffer(t);
    assert!(backend.buffer_pool_len().unwrap() > 0);
}

#[test]
fn test_elementwise_add_acquires_output_from_pool() {
    let mut backend = CpuBackend::new();
    backend.reclaim_buffer(Tensor::F64(
        TypedTensor::from_vec_col_major(vec![4], vec![0.0; 4]).unwrap(),
    ));
    assert_eq!(backend.buffer_pool_len().unwrap(), 1);

    let lhs =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    let rhs =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![4], vec![4.0, 3.0, 2.0, 1.0]).unwrap());
    let out = backend.add(&lhs, &rhs).unwrap();

    assert_eq!(backend.buffer_pool_len().unwrap(), 0);
    assert_eq!(get_f64(&out, &[0]), 5.0);
    assert_eq!(get_f64(&out, &[3]), 5.0);
    backend.reclaim_buffer(out);
    assert_eq!(backend.buffer_pool_len().unwrap(), 1);
}

#[test]
fn test_broadcast_multiply_fusion_computes_outer_product_without_materialized_inputs() {
    let mut backend = CpuBackend::new();
    let lhs = Tensor::F64(TypedTensor::from_vec_col_major(vec![3], vec![2.0, 3.0, 5.0]).unwrap());
    let rhs = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![7.0, 11.0]).unwrap());

    let out = backend
        .execute_broadcast_multiply(
            TensorRead::from_tensor(&lhs),
            &[3, 2],
            &[0],
            TensorRead::from_tensor(&rhs),
            &[3, 2],
            &[1],
        )
        .unwrap()
        .expect("CPU backend should execute broadcast multiply directly");

    assert_eq!(out.shape(), &[3, 2]);
    assert_eq!(
        out.as_slice::<f64>().unwrap(),
        &[14.0, 21.0, 35.0, 22.0, 33.0, 55.0]
    );
}

#[test]
fn test_cpu_elementwise_fusion_executes_add_mul_plan() {
    let mut backend = CpuBackend::new();
    let n = 65_536usize;
    let lhs_data = (0..n).map(|i| i as f64 + 1.0).collect::<Vec<_>>();
    let rhs_data = (0..n).map(|i| (i as f64 + 1.0) * 10.0).collect::<Vec<_>>();
    let lhs = Tensor::F64(TypedTensor::from_vec_col_major(vec![n], lhs_data).unwrap());
    let rhs = Tensor::F64(TypedTensor::from_vec_col_major(vec![n], rhs_data).unwrap());
    let fusion_plan = tenferro_tensor::backend::ElementwiseFusionPlan::new(
        DType::F64,
        2,
        vec![3],
        vec![
            tenferro_tensor::backend::ElementwiseFusionInst::new(
                tenferro_tensor::backend::ElementwiseFusionOp::Add,
                vec![0, 1],
            ),
            tenferro_tensor::backend::ElementwiseFusionInst::new(
                tenferro_tensor::backend::ElementwiseFusionOp::Multiply,
                vec![2, 0],
            ),
        ],
    );

    let outputs = backend
        .execute_elementwise_fusion(&[&lhs, &rhs], &fusion_plan)
        .unwrap()
        .expect("CPU backend should execute supported elementwise fusion plans");

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape(), &[n]);
    let actual = outputs[0].as_slice::<f64>().unwrap();
    assert_eq!(actual[0], 11.0);
    assert_eq!(actual[1], 44.0);
    assert_eq!(actual[n - 1], 11.0 * (n as f64).powi(2));
}

#[test]
fn test_cpu_elementwise_fusion_executes_broadcast_chain_plan() {
    let mut backend = CpuBackend::new();
    let n = 8_192usize;
    let lhs_data = (0..n).map(|i| i as f64 + 1.0).collect::<Vec<_>>();
    let lhs = Tensor::F64(TypedTensor::from_vec_col_major(vec![n], lhs_data).unwrap());
    let rhs = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![7.0, 11.0]).unwrap());
    let output_shape = vec![n, 2];
    let fusion_plan = tenferro_tensor::backend::ElementwiseFusionPlan::with_input_views(
        DType::F64,
        vec![
            tenferro_tensor::backend::ElementwiseFusionInputView::broadcast_in_dim(
                output_shape.clone(),
                vec![0],
            ),
            tenferro_tensor::backend::ElementwiseFusionInputView::broadcast_in_dim(
                output_shape,
                vec![1],
            ),
        ],
        vec![3],
        vec![
            tenferro_tensor::backend::ElementwiseFusionInst::new(
                tenferro_tensor::backend::ElementwiseFusionOp::Multiply,
                vec![0, 1],
            ),
            tenferro_tensor::backend::ElementwiseFusionInst::new(
                tenferro_tensor::backend::ElementwiseFusionOp::Add,
                vec![2, 0],
            ),
        ],
    );

    let outputs = backend
        .execute_elementwise_fusion(&[&lhs, &rhs], &fusion_plan)
        .unwrap()
        .expect("CPU backend should execute broadcast input views in fusion plans");

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape(), &[n, 2]);
    let actual = outputs[0].as_slice::<f64>().unwrap();
    assert_eq!(actual[0], 8.0);
    assert_eq!(actual[n - 1], n as f64 * 8.0);
    assert_eq!(actual[n], 12.0);
    assert_eq!(actual[2 * n - 1], n as f64 * 12.0);
}

#[test]
fn test_cpu_elementwise_fusion_broadcasts_mapped_unit_axes() {
    let mut backend = CpuBackend::new();
    let n = 16_384usize;
    let lhs = Tensor::F64(TypedTensor::from_vec_col_major(vec![1], vec![3.0]).unwrap());
    let rhs_data = (0..n).map(|i| i as f64 + 1.0).collect::<Vec<_>>();
    let rhs = Tensor::F64(TypedTensor::from_vec_col_major(vec![n], rhs_data).unwrap());
    let fusion_plan = tenferro_tensor::backend::ElementwiseFusionPlan::with_input_views(
        DType::F64,
        vec![
            tenferro_tensor::backend::ElementwiseFusionInputView::broadcast_in_dim(
                vec![n],
                vec![0],
            ),
            tenferro_tensor::backend::ElementwiseFusionInputView::Identity,
        ],
        vec![3],
        vec![
            tenferro_tensor::backend::ElementwiseFusionInst::new(
                tenferro_tensor::backend::ElementwiseFusionOp::Multiply,
                vec![0, 1],
            ),
            tenferro_tensor::backend::ElementwiseFusionInst::new(
                tenferro_tensor::backend::ElementwiseFusionOp::Add,
                vec![2, 0],
            ),
        ],
    );

    let outputs = backend
        .execute_elementwise_fusion(&[&lhs, &rhs], &fusion_plan)
        .unwrap()
        .expect("CPU backend should broadcast mapped unit axes in fusion plans");

    assert_eq!(outputs[0].shape(), &[n]);
    let actual = outputs[0].as_slice::<f64>().unwrap();
    assert_eq!(actual[0], 6.0);
    assert_eq!(actual[n - 1], 3.0 * n as f64 + 3.0);
}

#[test]
fn test_materialize_tensor_read_covers_host_tensor_and_view_variants() {
    let mut buffers = crate::buffer_pool::BufferPool::new();
    let tensors = [
        Tensor::F32(TypedTensor::from_vec_col_major(vec![1], vec![1.0_f32]).unwrap()),
        Tensor::F64(TypedTensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap()),
        Tensor::I32(TypedTensor::from_vec_col_major(vec![1], vec![1_i32]).unwrap()),
        Tensor::I64(TypedTensor::from_vec_col_major(vec![1], vec![1_i64]).unwrap()),
        Tensor::Bool(TypedTensor::from_vec_col_major(vec![1], vec![true]).unwrap()),
        Tensor::C32(
            TypedTensor::from_vec_col_major(vec![1], vec![Complex32::new(1.0, 0.0)]).unwrap(),
        ),
        Tensor::C64(
            TypedTensor::from_vec_col_major(vec![1], vec![Complex64::new(1.0, 0.0)]).unwrap(),
        ),
    ];

    for tensor in &tensors {
        let materialized = crate::materialize_tensor_read(
            &mut buffers,
            "dot_general",
            TensorRead::from_tensor(tensor),
        )
        .unwrap();
        assert_eq!(materialized.dtype(), tensor.dtype());
        assert_eq!(materialized.shape(), tensor.shape());
    }

    macro_rules! assert_materialized_view {
        ($view:expr, $ty:ty, $expected:expr) => {{
            let materialized = crate::materialize_tensor_read(
                &mut buffers,
                "dot_general",
                TensorRead::from_view($view),
            )
            .unwrap();
            assert_eq!(materialized.shape(), &[2]);
            assert_eq!(materialized.as_slice::<$ty>().unwrap(), $expected);
        }};
    }

    let shape = [2usize];
    let f32s = [1.25_f32, -2.5];
    let f64s = [3.5_f64, -4.75];
    let i32s = [5_i32, -6];
    let i64s = [7_i64, -8];
    let bools = [true, false];
    let c32s = [Complex32::new(1.0, -2.0), Complex32::new(-3.0, 4.0)];
    let c64s = [Complex64::new(5.0, -6.0), Complex64::new(-7.0, 8.0)];
    assert_materialized_view!(TensorView::f32(&shape, &f32s).unwrap(), f32, &f32s);
    assert_materialized_view!(TensorView::f64(&shape, &f64s).unwrap(), f64, &f64s);
    assert_materialized_view!(TensorView::i32(&shape, &i32s).unwrap(), i32, &i32s);
    assert_materialized_view!(TensorView::i64(&shape, &i64s).unwrap(), i64, &i64s);
    assert_materialized_view!(TensorView::bool(&shape, &bools).unwrap(), bool, &bools);
    assert_materialized_view!(TensorView::c32(&shape, &c32s).unwrap(), Complex32, &c32s);
    assert_materialized_view!(TensorView::c64(&shape, &c64s).unwrap(), Complex64, &c64s);
}

#[test]
fn cpu_view_materialization_uses_pool_aware_strided_copy() {
    let cpu_lib = include_str!("../../lib.rs");
    let dispatcher = cpu_lib
        .split_once("fn materialize_tensor_view")
        .unwrap()
        .1
        .split_once("#[cfg(test)]")
        .unwrap()
        .0;
    let structural = include_str!("../../structural.rs");
    let helper = structural
        .split_once("pub(crate) fn typed_materialize_view_with_pool")
        .unwrap()
        .1
        .split_once("pub(crate) fn typed_copy_view_into")
        .unwrap()
        .0;

    assert!(
        dispatcher.contains("structural::typed_materialize_view_with_pool"),
        "CPU TensorView materialization must dispatch through the pool-aware strided helper"
    );
    assert!(
        !dispatcher.contains("to_contiguous"),
        "CPU TensorView materialization must not bypass the CPU pool with TypedTensorView::to_contiguous"
    );
    assert!(helper.contains("StridedView::new"));
    assert!(helper.contains("map_into("));
    assert!(helper.contains("PooledUninitOutput"));
    assert!(helper.contains("// SAFETY:"));
    assert!(
        helper.contains("let mut out = PooledUninitOutput"),
        "full-overwrite pooled allocation requires the repository invariant marker"
    );
    for forbidden in [
        "for ",
        "flat_to_multi",
        "to_contiguous",
        "materialize_view_buffer_col_major",
        "zeroed_tensor_from_pool",
        "filled_tensor_from_pool",
        "TypedTensor::zeros",
        "vec![",
    ] {
        assert!(
            !helper.contains(forbidden),
            "typed materialization helper must not contain `{forbidden}`"
        );
    }
}

#[test]
fn cpu_view_materialization_preserves_transposed_and_scattered_values() {
    let mut backend = CpuBackend::new();

    let transposed_storage = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let transposed = tenferro_tensor::TypedTensorView::from_col_major(&[2, 3], &transposed_storage)
        .unwrap()
        .transpose_view([1, 0])
        .unwrap();
    let transposed = backend
        .reshape_read(TensorRead::from_view(TensorView::F64(transposed)), &[3, 2])
        .unwrap();
    assert_eq!(transposed.shape(), &[3, 2]);
    assert_eq!(
        transposed.as_slice::<f64>().unwrap(),
        &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    );

    let scattered_storage = (0..20).map(|value| value as f64 * 10.0).collect::<Vec<_>>();
    let mut scattered_shape = vec![1; 24];
    scattered_shape[0] = 2;
    scattered_shape[11] = 2;
    let mut scattered_strides = vec![19; 24];
    scattered_strides[0] = 3;
    scattered_strides[11] = 11;
    let scattered = tenferro_tensor::TypedTensorView::from_slice(
        scattered_shape,
        scattered_strides,
        2,
        &scattered_storage,
    )
    .unwrap();
    let scattered = backend
        .reshape_read(TensorRead::from_view(TensorView::F64(scattered)), &[4])
        .unwrap();
    assert_eq!(scattered.shape(), &[4]);
    assert_eq!(
        scattered.as_slice::<f64>().unwrap(),
        &[20.0, 50.0, 130.0, 160.0]
    );
}

#[test]
fn cpu_structural_read_transpose_explicit_stride_exact_output() {
    let mut backend = CpuBackend::new();
    let storage = (0..16).map(|value| value as f64 * 10.0).collect::<Vec<_>>();
    let view = tenferro_tensor::TypedTensorView::from_slice([2, 3], [2, 5], 1, &storage).unwrap();

    let output = backend
        .transpose_read(TensorRead::from_view(TensorView::F64(view)), &[1, 0])
        .unwrap();

    assert_eq!(output.shape(), &[3, 2]);
    assert_eq!(
        output.as_slice::<f64>().unwrap(),
        &[10.0, 60.0, 110.0, 30.0, 80.0, 130.0]
    );
}

#[test]
fn cpu_structural_read_reshape_explicit_stride_exact_output() {
    let mut backend = CpuBackend::new();
    let storage = (0..10).collect::<Vec<i32>>();
    let view = tenferro_tensor::TypedTensorView::from_slice([2, 2], [3, -1], 5, &storage).unwrap();

    let output = backend
        .reshape_read(TensorRead::from_view(TensorView::I32(view)), &[4])
        .unwrap();

    assert_eq!(output.shape(), &[4]);
    assert_eq!(output.as_slice::<i32>().unwrap(), &[5, 8, 4, 7]);
}

#[test]
fn cpu_structural_read_broadcast_in_dim_explicit_stride_exact_output() {
    let mut backend = CpuBackend::new();
    let storage = [0_i64, 10, 20, 30, 40];
    let view = tenferro_tensor::TypedTensorView::from_slice([2, 1], [-2, 7], 4, &storage).unwrap();

    let output = backend
        .broadcast_in_dim_read(
            TensorRead::from_view(TensorView::I64(view)),
            &[2, 3],
            &[0, 1],
        )
        .unwrap();

    assert_eq!(output.shape(), &[2, 3]);
    assert_eq!(output.as_slice::<i64>().unwrap(), &[40, 20, 40, 20, 40, 20]);
}

#[test]
fn cpu_structural_read_direct_helpers_preserve_view_placement() {
    let mut buffers = crate::buffer_pool::BufferPool::new();
    let mut input = TypedTensor::<i64>::from_vec_col_major(vec![2], vec![9, 13]).unwrap();
    input.set_placement(tenferro_tensor::Placement {
        memory_kind: tenferro_tensor::MemoryKind::PinnedHost,
        device: None,
        cpu_affinity: None,
    });

    let transpose = crate::structural::transpose_read_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::I64(input.as_view())),
        &[0],
    )
    .unwrap();
    let reshape = crate::structural::reshape_read_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::I64(input.as_view())),
        &[1, 2],
    )
    .unwrap();
    let broadcast = crate::structural::broadcast_in_dim_read_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::I64(input.as_view())),
        &[2, 2],
        &[0],
    )
    .unwrap();

    for output in [transpose, reshape, broadcast] {
        assert_eq!(
            output.placement().memory_kind,
            tenferro_tensor::MemoryKind::PinnedHost
        );
        assert_eq!(output.placement().device, None);
    }
}

#[test]
fn cpu_structural_read_direct_helpers_cover_all_dtypes() {
    let mut buffers = crate::buffer_pool::BufferPool::new();

    macro_rules! assert_dtype_dispatch {
        ($variant:ident, $dtype:expr, $storage:expr) => {{
            let storage = $storage;
            let view = tenferro_tensor::TypedTensorView::from_slice([2], [1], 0, &storage).unwrap();
            let transpose = crate::structural::transpose_read_with_pool(
                &mut buffers,
                TensorRead::from_view(TensorView::$variant(view)),
                &[0],
            )
            .unwrap();

            let view = tenferro_tensor::TypedTensorView::from_slice([2], [1], 0, &storage).unwrap();
            let reshape = crate::structural::reshape_read_with_pool(
                &mut buffers,
                TensorRead::from_view(TensorView::$variant(view)),
                &[1, 2],
            )
            .unwrap();

            let view = tenferro_tensor::TypedTensorView::from_slice([2], [1], 0, &storage).unwrap();
            let broadcast = crate::structural::broadcast_in_dim_read_with_pool(
                &mut buffers,
                TensorRead::from_view(TensorView::$variant(view)),
                &[2, 2],
                &[0],
            )
            .unwrap();

            assert_eq!(transpose.dtype(), $dtype);
            assert_eq!(reshape.dtype(), $dtype);
            assert_eq!(broadcast.dtype(), $dtype);
        }};
    }

    assert_dtype_dispatch!(F32, DType::F32, [1.0_f32, 2.0]);
    assert_dtype_dispatch!(F64, DType::F64, [1.0_f64, 2.0]);
    assert_dtype_dispatch!(I32, DType::I32, [1_i32, 2]);
    assert_dtype_dispatch!(I64, DType::I64, [1_i64, 2]);
    assert_dtype_dispatch!(Bool, DType::Bool, [true, false]);
    assert_dtype_dispatch!(
        C32,
        DType::C32,
        [Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)]
    );
    assert_dtype_dispatch!(
        C64,
        DType::C64,
        [Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)]
    );
}

#[test]
fn cpu_structural_read_direct_helpers_cover_zero_stride_empty_and_rank_zero() {
    let mut buffers = crate::buffer_pool::BufferPool::new();

    let repeated_storage = [7_i32, 11];
    let repeated =
        tenferro_tensor::TypedTensorView::from_slice([2, 3], [1, 0], 0, &repeated_storage).unwrap();
    let repeated = crate::structural::transpose_read_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::I32(repeated)),
        &[1, 0],
    )
    .unwrap();
    assert_eq!(repeated.shape(), &[3, 2]);
    assert_eq!(repeated.as_slice::<i32>().unwrap(), &[7, 7, 7, 11, 11, 11]);

    let empty_storage: [f64; 0] = [];
    let empty =
        tenferro_tensor::TypedTensorView::from_slice([0, 3], [5, -2], 0, &empty_storage).unwrap();
    let empty = crate::structural::broadcast_in_dim_read_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::F64(empty)),
        &[0, 3, 2],
        &[0, 1],
    )
    .unwrap();
    assert_eq!(empty.shape(), &[0, 3, 2]);
    assert_eq!(empty.as_slice::<f64>().unwrap(), &[] as &[f64]);

    let scalar_storage = [42.5_f64];
    let scalar = tenferro_tensor::TypedTensorView::from_slice([], [], 0, &scalar_storage).unwrap();
    let scalar = crate::structural::reshape_read_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::F64(scalar)),
        &[],
    )
    .unwrap();
    assert_eq!(scalar.shape(), &[] as &[usize]);
    assert_eq!(scalar.as_slice::<f64>().unwrap(), &[42.5]);
}

#[test]
fn cpu_structural_read_direct_helpers_match_owned_validation_errors() {
    let mut buffers = crate::buffer_pool::BufferPool::new();
    let input = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap());

    let owned_transpose =
        crate::structural::transpose_with_pool(&mut buffers, &input, &[1]).unwrap_err();
    let view_transpose = crate::structural::transpose_read_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::F64(match &input {
            Tensor::F64(input) => input.as_view(),
            _ => unreachable!(),
        })),
        &[1],
    )
    .unwrap_err();
    assert_eq!(view_transpose.kind(), owned_transpose.kind());
    assert_eq!(view_transpose.to_string(), owned_transpose.to_string());

    let owned_reshape = crate::structural::reshape(&input, &[3]).unwrap_err();
    let view_reshape = crate::structural::reshape_read_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::F64(match &input {
            Tensor::F64(input) => input.as_view(),
            _ => unreachable!(),
        })),
        &[3],
    )
    .unwrap_err();
    assert_eq!(view_reshape.kind(), owned_reshape.kind());
    assert_eq!(view_reshape.to_string(), owned_reshape.to_string());

    let owned_broadcast =
        crate::structural::broadcast_in_dim_with_pool(&mut buffers, &input, &[3], &[0])
            .unwrap_err();
    let view_broadcast = crate::structural::broadcast_in_dim_read_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::F64(match &input {
            Tensor::F64(input) => input.as_view(),
            _ => unreachable!(),
        })),
        &[3],
        &[0],
    )
    .unwrap_err();
    assert_eq!(view_broadcast.kind(), owned_broadcast.kind());
    assert_eq!(view_broadcast.to_string(), owned_broadcast.to_string());
}

#[test]
fn cpu_structural_read_empty_pathological_layout_returns_typed_errors_without_panicking() {
    let empty_storage: [f64; 0] = [];

    let transpose = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let mut buffers = crate::buffer_pool::BufferPool::new();
        let view = tenferro_tensor::TypedTensorView::from_slice(
            [0, usize::MAX],
            [0, 0],
            0,
            &empty_storage,
        )
        .unwrap();
        crate::structural::transpose_read_with_pool(
            &mut buffers,
            TensorRead::from_view(TensorView::F64(view)),
            &[0, 1],
        )
    }));
    let transpose = transpose
        .expect("transpose_read must not panic")
        .unwrap_err();
    assert!(matches!(transpose, crate::Error::Validation { .. }));

    let broadcast = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let mut buffers = crate::buffer_pool::BufferPool::new();
        let view =
            tenferro_tensor::TypedTensorView::from_slice([0], [0], 0, &empty_storage).unwrap();
        crate::structural::broadcast_in_dim_read_with_pool(
            &mut buffers,
            TensorRead::from_view(TensorView::F64(view)),
            &[0, usize::MAX],
            &[0],
        )
    }));
    let broadcast = broadcast
        .expect("broadcast_in_dim_read must not panic")
        .unwrap_err();
    assert!(matches!(broadcast, crate::Error::Validation { .. }));
}

#[test]
fn cpu_structural_read_backend_and_exec_session_outputs_match() {
    let storage = [0.0_f64, 10.0, 20.0, 30.0, 40.0, 50.0];
    let mut backend = CpuBackend::new();

    let backend_outputs = {
        let transpose_view =
            tenferro_tensor::TypedTensorView::from_slice([2, 2], [2, -1], 3, &storage).unwrap();
        let reshape_view =
            tenferro_tensor::TypedTensorView::from_slice([2, 2], [2, -1], 3, &storage).unwrap();
        let broadcast_view =
            tenferro_tensor::TypedTensorView::from_slice([2, 1], [2, 0], 1, &storage).unwrap();
        [
            backend
                .transpose_read(
                    TensorRead::from_view(TensorView::F64(transpose_view)),
                    &[1, 0],
                )
                .unwrap(),
            backend
                .reshape_read(TensorRead::from_view(TensorView::F64(reshape_view)), &[4])
                .unwrap(),
            backend
                .broadcast_in_dim_read(
                    TensorRead::from_view(TensorView::F64(broadcast_view)),
                    &[2, 3],
                    &[0, 1],
                )
                .unwrap(),
        ]
    };

    let session_outputs = backend.with_backend_session(|session| {
        let transpose_view =
            tenferro_tensor::TypedTensorView::from_slice([2, 2], [2, -1], 3, &storage).unwrap();
        let reshape_view =
            tenferro_tensor::TypedTensorView::from_slice([2, 2], [2, -1], 3, &storage).unwrap();
        let broadcast_view =
            tenferro_tensor::TypedTensorView::from_slice([2, 1], [2, 0], 1, &storage).unwrap();
        [
            session
                .transpose_read(
                    TensorRead::from_view(TensorView::F64(transpose_view)),
                    &[1, 0],
                )
                .unwrap(),
            session
                .reshape_read(TensorRead::from_view(TensorView::F64(reshape_view)), &[4])
                .unwrap(),
            session
                .broadcast_in_dim_read(
                    TensorRead::from_view(TensorView::F64(broadcast_view)),
                    &[2, 3],
                    &[0, 1],
                )
                .unwrap(),
        ]
    });

    for (backend_output, session_output) in backend_outputs.iter().zip(&session_outputs) {
        assert_eq!(backend_output.shape(), session_output.shape());
        assert_eq!(
            backend_output.as_slice::<f64>().unwrap(),
            session_output.as_slice::<f64>().unwrap()
        );
    }
}

#[test]
fn cpu_view_materialization_handles_negative_and_zero_strides() {
    let mut buffers = crate::buffer_pool::BufferPool::new();

    let reversed_storage = [0_i32, 10, 20, 30, 40, 50];
    let reversed =
        tenferro_tensor::TypedTensorView::from_slice([3], [-2], 5, &reversed_storage).unwrap();
    let reversed = crate::structural::typed_materialize_view_with_pool(
        &mut buffers,
        &reversed,
        "negative_stride_materialize",
    )
    .unwrap();
    assert_eq!(reversed.as_slice().unwrap(), &[50, 30, 10]);

    let broadcast_storage = [7_i32, 11];
    let broadcast =
        tenferro_tensor::TypedTensorView::from_slice([2, 3], [1, 0], 0, &broadcast_storage)
            .unwrap();
    let broadcast = crate::structural::typed_materialize_view_with_pool(
        &mut buffers,
        &broadcast,
        "zero_stride_materialize",
    )
    .unwrap();
    assert_eq!(broadcast.shape(), &[2, 3]);
    assert_eq!(broadcast.as_slice().unwrap(), &[7, 11, 7, 11, 7, 11]);
}

#[test]
fn cpu_view_materialization_handles_empty_and_rank_zero_views() {
    let mut buffers = crate::buffer_pool::BufferPool::new();

    let empty_storage: [f64; 0] = [];
    let empty = tenferro_tensor::TypedTensorView::from_col_major(&[0, 3], &empty_storage).unwrap();
    let empty = crate::structural::typed_materialize_view_with_pool(
        &mut buffers,
        &empty,
        "empty_materialize",
    )
    .unwrap();
    assert_eq!(empty.shape(), &[0, 3]);
    assert_eq!(empty.as_slice().unwrap(), &[] as &[f64]);

    let scalar_storage = [42.5_f64];
    let scalar = tenferro_tensor::TypedTensorView::from_col_major(&[], &scalar_storage).unwrap();
    let scalar = crate::structural::typed_materialize_view_with_pool(
        &mut buffers,
        &scalar,
        "rank_zero_materialize",
    )
    .unwrap();
    assert_eq!(scalar.shape(), &[] as &[usize]);
    assert_eq!(scalar.as_slice().unwrap(), &[42.5]);
}

#[test]
fn cpu_view_materialization_preserves_static_rank_and_placement() {
    let mut buffers = crate::buffer_pool::BufferPool::new();
    let storage = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let ranked =
        tenferro_tensor::TypedTensorView::<_, tenferro_tensor::Rank<2>>::from_slice_ranked(
            [3, 2],
            [2, 1],
            0,
            &storage,
        )
        .unwrap();
    let ranked: TypedTensor<f64, tenferro_tensor::Rank<2>> =
        crate::structural::typed_materialize_view_with_pool(
            &mut buffers,
            &ranked,
            "static_rank_materialize",
        )
        .unwrap();
    assert_eq!(ranked.shape(), &[3, 2]);
    assert_eq!(ranked.as_slice().unwrap(), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);

    let mut placed = TypedTensor::<i64>::from_vec_col_major(vec![2], vec![9_i64, 13]).unwrap();
    placed.set_placement(tenferro_tensor::Placement {
        memory_kind: tenferro_tensor::MemoryKind::PinnedHost,
        device: None,
        cpu_affinity: None,
    });
    let placed = crate::structural::typed_materialize_view_with_pool(
        &mut buffers,
        &placed.as_view(),
        "placement_materialize",
    )
    .unwrap();
    assert_eq!(
        placed.placement().memory_kind,
        tenferro_tensor::MemoryKind::PinnedHost
    );
    assert_eq!(placed.as_slice().unwrap(), &[9, 13]);
}

#[test]
fn cpu_view_materialization_rejects_backend_buffer_with_caller_operation_name() {
    let backend_tensor = TypedTensor::<f64>::from_buffer_col_major(
        vec![2],
        tenferro_tensor::StorageBuffer::Backend(Box::new(tenferro_tensor::BackendStorageHandle::<
            f64,
        >::new_with_len(17, 2))),
        tenferro_tensor::Placement {
            memory_kind: tenferro_tensor::MemoryKind::Device,
            device: Some(tenferro_tensor::DeviceId {
                kind: tenferro_tensor::DeviceKind::Gpu(tenferro_tensor::GpuBackendKind::Cuda),
                ordinal: 0,
            }),
            cpu_affinity: None,
        },
    )
    .unwrap();
    let view = backend_tensor
        .backend_region_view(vec![2], vec![1], 0)
        .unwrap();
    let mut buffers = crate::buffer_pool::BufferPool::new();
    let error = crate::structural::typed_materialize_view_with_pool(
        &mut buffers,
        &view,
        "review_materialize_op",
    )
    .unwrap_err();

    assert!(matches!(
        error,
        crate::Error::RuntimeState {
            op: "review_materialize_op",
            ref message,
        } if message.contains("download to host")
    ));
}

#[test]
fn test_structural_transpose_acquires_output_from_pool() {
    let mut backend = CpuBackend::new();
    backend.reclaim_buffer(Tensor::F64(
        TypedTensor::from_vec_col_major(vec![4], vec![0.0; 4]).unwrap(),
    ));
    assert_eq!(backend.buffer_pool_len().unwrap(), 1);

    let input =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    let out = backend.transpose(&input, &[1, 0]).unwrap();

    assert_eq!(backend.buffer_pool_len().unwrap(), 0);
    assert_eq!(get_f64(&out, &[0, 0]), 1.0);
    assert_eq!(get_f64(&out, &[1, 0]), 3.0);
    assert_eq!(get_f64(&out, &[0, 1]), 2.0);
    assert_eq!(get_f64(&out, &[1, 1]), 4.0);
    backend.reclaim_buffer(out);
    assert_eq!(backend.buffer_pool_len().unwrap(), 1);
}

#[test]
fn test_cast_acquires_output_from_dtype_pool() {
    let mut backend = CpuBackend::new();
    backend.reclaim_buffer(Tensor::F32(
        TypedTensor::from_vec_col_major(vec![4], vec![0.0; 4]).unwrap(),
    ));
    assert_eq!(backend.buffer_pool_len().unwrap(), 1);

    let input =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![4], vec![1.25, 2.5, 3.75, 4.0]).unwrap());
    let out = backend.cast(&input, DType::F32).unwrap();

    assert_eq!(backend.buffer_pool_len().unwrap(), 0);
    assert_eq!(get_f32(&out, &[0]), 1.25);
    assert_eq!(get_f32(&out, &[3]), 4.0);
    backend.reclaim_buffer(out);
    assert_eq!(backend.buffer_pool_len().unwrap(), 1);
}

#[test]
fn test_slice_acquires_output_from_pool() {
    let mut backend = CpuBackend::new();
    backend.reclaim_buffer(Tensor::F64(
        TypedTensor::from_vec_col_major(vec![2], vec![0.0; 2]).unwrap(),
    ));
    assert_eq!(backend.buffer_pool_len().unwrap(), 1);

    let input =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    let config = SliceConfig {
        starts: vec![1],
        limits: vec![3],
        strides: vec![1],
    };
    let out = backend.slice(&input, &config).unwrap();

    assert_eq!(backend.buffer_pool_len().unwrap(), 0);
    assert_eq!(get_f64(&out, &[0]), 2.0);
    assert_eq!(get_f64(&out, &[1]), 3.0);
    backend.reclaim_buffer(out);
    assert_eq!(backend.buffer_pool_len().unwrap(), 1);
}

#[test]
fn test_pad_acquires_and_zeroes_output_from_pool() {
    let mut backend = CpuBackend::new();
    backend.reclaim_buffer(Tensor::F64(
        TypedTensor::from_vec_col_major(vec![4], vec![9.0; 4]).unwrap(),
    ));
    assert_eq!(backend.buffer_pool_len().unwrap(), 1);

    let input = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap());
    let config = PadConfig {
        edge_padding_low: vec![1],
        edge_padding_high: vec![1],
        interior_padding: vec![0],
    };
    let out = backend.pad(&input, &config).unwrap();

    assert_eq!(backend.buffer_pool_len().unwrap(), 0);
    assert_eq!(get_f64(&out, &[0]), 0.0);
    assert_eq!(get_f64(&out, &[1]), 1.0);
    assert_eq!(get_f64(&out, &[2]), 2.0);
    assert_eq!(get_f64(&out, &[3]), 0.0);
    backend.reclaim_buffer(out);
    assert_eq!(backend.buffer_pool_len().unwrap(), 1);
}

#[test]
fn test_dynamic_update_slice_acquires_clone_from_pool() {
    let mut backend = CpuBackend::new();
    backend.reclaim_buffer(Tensor::F64(
        TypedTensor::from_vec_col_major(vec![4], vec![9.0; 4]).unwrap(),
    ));
    assert_eq!(backend.buffer_pool_len().unwrap(), 1);

    let operand =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![4], vec![0.0, 1.0, 2.0, 3.0]).unwrap());
    let update = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![7.0, 8.0]).unwrap());
    let starts = Tensor::I64(TypedTensor::from_vec_col_major(vec![1], vec![1]).unwrap());
    let out = backend
        .dynamic_update_slice(&operand, &update, &starts)
        .unwrap();

    assert_eq!(backend.buffer_pool_len().unwrap(), 0);
    assert_eq!(get_f64(&out, &[0]), 0.0);
    assert_eq!(get_f64(&out, &[1]), 7.0);
    assert_eq!(get_f64(&out, &[2]), 8.0);
    assert_eq!(get_f64(&out, &[3]), 3.0);
    backend.reclaim_buffer(out);
    assert_eq!(backend.buffer_pool_len().unwrap(), 1);
}

#[test]
fn test_reclaim_buffer_covers_all_dtypes() {
    let mut backend = CpuBackend::new();
    let f32_t = Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0f32, 2.0]).unwrap());
    backend.reclaim_buffer(f32_t);
    let c32_t = Tensor::C32(
        TypedTensor::from_vec_col_major(vec![1], vec![Complex32::new(1.0, 0.0)]).unwrap(),
    );
    backend.reclaim_buffer(c32_t);
    let c64_t = Tensor::C64(
        TypedTensor::from_vec_col_major(vec![1], vec![Complex64::new(1.0, 0.0)]).unwrap(),
    );
    backend.reclaim_buffer(c64_t);
    assert!(backend.buffer_pool_len().unwrap() >= 3);
}

#[test]
fn test_install_with_pool_preserves_buffers() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let t = TensorElementwise::add(
        &mut backend,
        &Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap()),
        &Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![3.0, 4.0]).unwrap()),
    )
    .unwrap();
    assert_eq!(get_f64(&t, &[0]), 4.0);
    assert_eq!(get_f64(&t, &[1]), 6.0);
    assert_eq!(backend.buffer_pool_len().unwrap(), 0);
}

#[test]
fn test_with_linalg_pool_reports_poison_after_panic() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    backend.reclaim_buffer(Tensor::F64(
        TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap(),
    ));
    assert_eq!(backend.buffer_pool_len().unwrap(), 1);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = backend.with_linalg_pool::<()>(|_, _| panic!("forced linalg panic"));
    }));

    assert!(result.is_err());
    assert_eq!(
        backend.buffer_pool_len().unwrap_err().kind(),
        tenferro_tensor::ErrorKind::RuntimeState
    );
}

#[test]
fn test_backend_session_reports_poison_after_panic() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    backend.reclaim_buffer(Tensor::F64(
        TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap(),
    ));
    assert_eq!(backend.buffer_pool_len().unwrap(), 1);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        backend.with_backend_session::<()>(|_| panic!("forced session panic"));
    }));

    assert!(result.is_err());
    assert_eq!(
        backend.buffer_pool_len().unwrap_err().kind(),
        tenferro_tensor::ErrorKind::RuntimeState
    );
}

#[test]
fn test_exec_session_read_reductions_and_reclaim_cover_typed_paths() {
    let mut backend = CpuBackend::new();
    backend.with_backend_session(|exec| {
        let lhs = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
        let rhs = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
        let added = exec
            .add_read(TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs))
            .unwrap();
        assert_eq!(added.as_slice::<f64>().unwrap(), &[4.0, 6.0]);

        let view_data = [2.0_f64, 3.0];
        let view_shape = [2usize];
        assert_eq!(
            exec.reduce_sum_read(
                TensorRead::from_view(TensorView::f64(&view_shape, &view_data).unwrap()),
                &[0],
            )
            .unwrap()
            .as_slice::<f64>()
            .unwrap(),
            &[5.0]
        );
        assert_eq!(
            exec.reduce_prod_read(TensorRead::from_tensor(&lhs), &[0])
                .unwrap()
                .as_slice::<f64>()
                .unwrap(),
            &[2.0]
        );
        assert_eq!(
            exec.reduce_max_read(TensorRead::from_tensor(&rhs), &[0])
                .unwrap()
                .as_slice::<f64>()
                .unwrap(),
            &[4.0]
        );
        assert_eq!(
            exec.reduce_min_read(TensorRead::from_tensor(&rhs), &[0])
                .unwrap()
                .as_slice::<f64>()
                .unwrap(),
            &[3.0]
        );

        exec.reclaim_buffer(Tensor::F32(
            TypedTensor::from_vec_col_major(vec![1], vec![0.0_f32]).unwrap(),
        ));
        exec.reclaim_buffer(Tensor::F64(
            TypedTensor::from_vec_col_major(vec![1], vec![0.0_f64]).unwrap(),
        ));
        exec.reclaim_buffer(Tensor::I32(
            TypedTensor::from_vec_col_major(vec![1], vec![0_i32]).unwrap(),
        ));
        exec.reclaim_buffer(Tensor::I64(
            TypedTensor::from_vec_col_major(vec![1], vec![0_i64]).unwrap(),
        ));
        exec.reclaim_buffer(Tensor::Bool(
            TypedTensor::from_vec_col_major(vec![1], vec![false]).unwrap(),
        ));
        exec.reclaim_buffer(Tensor::C32(
            TypedTensor::from_vec_col_major(vec![1], vec![Complex32::new(0.0, 0.0)]).unwrap(),
        ));
        exec.reclaim_buffer(Tensor::C64(
            TypedTensor::from_vec_col_major(vec![1], vec![Complex64::new(0.0, 0.0)]).unwrap(),
        ));
    });

    assert!(backend.buffer_pool_len().unwrap() >= 7);
}

#[test]
fn test_default_backend_session_methods_cover_cache_fallbacks() {
    #[doc(hidden)]
    struct DefaultOnlyBackendSessionMarker;
    struct DefaultOnlyBackend;

    macro_rules! panic_backend_methods {
        ($($name:ident($($arg:ident : $argty:ty),*) -> $ret:ty;)+) => {
            $(
                fn $name(&mut self, $($arg: $argty),*) -> $ret {
                    $(let _ = &$arg;)*
                    panic!(concat!(stringify!($name), " should not be called by this test"))
                }
            )+
        };
    }

    impl BackendRuntimeCache for DefaultOnlyBackend {
        type RuntimeCache = ();
    }

    impl TensorElementwise for DefaultOnlyBackend {
        panic_backend_methods! {
        sub(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        mul(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        neg(input: &Tensor) -> crate::Result<Tensor>;
        div(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        abs(input: &Tensor) -> crate::Result<Tensor>;
        sign(input: &Tensor) -> crate::Result<Tensor>;
        maximum(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        minimum(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> crate::Result<Tensor>;
        select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> crate::Result<Tensor>;
        clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) -> crate::Result<Tensor>;
        }

        fn add(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
            CpuBackend::new().add(lhs, rhs)
        }

        fn conj(&mut self, input: &Tensor) -> crate::Result<Tensor> {
            CpuBackend::new().conj(input)
        }
    }

    impl TensorAnalytic for DefaultOnlyBackend {
        panic_backend_methods! {
        exp(input: &Tensor) -> crate::Result<Tensor>;
        log(input: &Tensor) -> crate::Result<Tensor>;
        sin(input: &Tensor) -> crate::Result<Tensor>;
        cos(input: &Tensor) -> crate::Result<Tensor>;
        tanh(input: &Tensor) -> crate::Result<Tensor>;
        sqrt(input: &Tensor) -> crate::Result<Tensor>;
        rsqrt(input: &Tensor) -> crate::Result<Tensor>;
        pow(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        expm1(input: &Tensor) -> crate::Result<Tensor>;
        log1p(input: &Tensor) -> crate::Result<Tensor>;
        }
    }

    impl TensorStructural for DefaultOnlyBackend {
        panic_backend_methods! {
        transpose(input: &Tensor, perm: &[usize]) -> crate::Result<Tensor>;
        reshape(input: &Tensor, shape: &[usize]) -> crate::Result<Tensor>;
        broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) -> crate::Result<Tensor>;
        cast(input: &Tensor, to: DType) -> crate::Result<Tensor>;
        extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> crate::Result<Tensor>;
        embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> crate::Result<Tensor>;
        tril(input: &Tensor, k: i64) -> crate::Result<Tensor>;
        triu(input: &Tensor, k: i64) -> crate::Result<Tensor>;
        }
    }

    impl TensorReduction for DefaultOnlyBackend {
        fn reduce_sum(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
            CpuBackend::new().reduce_sum(input, axes)
        }

        fn reduce_prod(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
            CpuBackend::new().reduce_prod(input, axes)
        }

        fn reduce_max(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
            CpuBackend::new().reduce_max(input, axes)
        }

        fn reduce_min(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
            CpuBackend::new().reduce_min(input, axes)
        }
    }

    impl TensorIndexing for DefaultOnlyBackend {
        panic_backend_methods! {
        gather(operand: &Tensor, start_indices: &Tensor, config: &GatherConfig) -> crate::Result<Tensor>;
        scatter(operand: &Tensor, scatter_indices: &Tensor, updates: &Tensor, config: &ScatterConfig) -> crate::Result<Tensor>;
        slice(input: &Tensor, config: &SliceConfig) -> crate::Result<Tensor>;
        dynamic_slice(input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) -> crate::Result<Tensor>;
        dynamic_update_slice(operand: &Tensor, update: &Tensor, starts: &Tensor) -> crate::Result<Tensor>;
        pad(input: &Tensor, config: &PadConfig) -> crate::Result<Tensor>;
        concatenate(inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor>;
        reverse(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
        }
    }

    impl TensorDot for DefaultOnlyBackend {
        fn dot_general(
            &mut self,
            lhs: &Tensor,
            rhs: &Tensor,
            config: &DotGeneralConfig,
        ) -> crate::Result<Tensor> {
            CpuBackend::new().dot_general(lhs, rhs, config)
        }
    }

    impl BackendCachedDot for DefaultOnlyBackend {}

    impl BackendSession for DefaultOnlyBackend {
        fn session_type_id(&self) -> std::any::TypeId {
            std::any::TypeId::of::<DefaultOnlyBackendSessionMarker>()
        }

        unsafe fn session_data_mut(&mut self) -> *mut () {
            self as *mut Self as *mut ()
        }
    }

    impl BackendSessionHost for DefaultOnlyBackend {}

    impl TensorDeviceTransfer for DefaultOnlyBackend {
        fn download_to_host(&mut self, _tensor: TensorRead<'_>) -> crate::Result<Tensor> {
            Err(crate::Error::unsupported(
                "DefaultOnlyBackend::download_to_host",
                "test backend does not transfer tensors",
            ))
        }

        fn upload_host_tensor(&mut self, _tensor: TensorRead<'_>) -> crate::Result<Tensor> {
            Err(crate::Error::unsupported(
                "DefaultOnlyBackend::upload_host_tensor",
                "test backend does not transfer tensors",
            ))
        }
    }

    impl TensorBuffer for DefaultOnlyBackend {}

    impl TensorFusion for DefaultOnlyBackend {}

    impl TensorBackend for DefaultOnlyBackend {}

    struct DefaultOnlyExec;

    impl TensorElementwise for DefaultOnlyExec {
        panic_backend_methods! {
        sub(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        mul(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        neg(input: &Tensor) -> crate::Result<Tensor>;
        div(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        abs(input: &Tensor) -> crate::Result<Tensor>;
        sign(input: &Tensor) -> crate::Result<Tensor>;
        maximum(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        minimum(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> crate::Result<Tensor>;
        select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> crate::Result<Tensor>;
        clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) -> crate::Result<Tensor>;
        }

        fn add(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
            CpuBackend::new().add(lhs, rhs)
        }

        fn conj(&mut self, input: &Tensor) -> crate::Result<Tensor> {
            CpuBackend::new().conj(input)
        }
    }

    impl TensorAnalytic for DefaultOnlyExec {
        panic_backend_methods! {
        exp(input: &Tensor) -> crate::Result<Tensor>;
        log(input: &Tensor) -> crate::Result<Tensor>;
        sin(input: &Tensor) -> crate::Result<Tensor>;
        cos(input: &Tensor) -> crate::Result<Tensor>;
        tanh(input: &Tensor) -> crate::Result<Tensor>;
        sqrt(input: &Tensor) -> crate::Result<Tensor>;
        rsqrt(input: &Tensor) -> crate::Result<Tensor>;
        pow(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        expm1(input: &Tensor) -> crate::Result<Tensor>;
        log1p(input: &Tensor) -> crate::Result<Tensor>;
        }
    }

    impl TensorStructural for DefaultOnlyExec {
        panic_backend_methods! {
        transpose(input: &Tensor, perm: &[usize]) -> crate::Result<Tensor>;
        reshape(input: &Tensor, shape: &[usize]) -> crate::Result<Tensor>;
        broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) -> crate::Result<Tensor>;
        cast(input: &Tensor, to: DType) -> crate::Result<Tensor>;
        extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> crate::Result<Tensor>;
        embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> crate::Result<Tensor>;
        tril(input: &Tensor, k: i64) -> crate::Result<Tensor>;
        triu(input: &Tensor, k: i64) -> crate::Result<Tensor>;
        }
    }

    impl TensorReduction for DefaultOnlyExec {
        fn reduce_sum(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
            CpuBackend::new().reduce_sum(input, axes)
        }

        fn reduce_prod(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
            CpuBackend::new().reduce_prod(input, axes)
        }

        fn reduce_max(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
            CpuBackend::new().reduce_max(input, axes)
        }

        fn reduce_min(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
            CpuBackend::new().reduce_min(input, axes)
        }
    }

    impl TensorIndexing for DefaultOnlyExec {
        panic_backend_methods! {
        gather(operand: &Tensor, start_indices: &Tensor, config: &GatherConfig) -> crate::Result<Tensor>;
        scatter(operand: &Tensor, scatter_indices: &Tensor, updates: &Tensor, config: &ScatterConfig) -> crate::Result<Tensor>;
        slice(input: &Tensor, config: &SliceConfig) -> crate::Result<Tensor>;
        dynamic_slice(input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) -> crate::Result<Tensor>;
        dynamic_update_slice(operand: &Tensor, update: &Tensor, starts: &Tensor) -> crate::Result<Tensor>;
        pad(input: &Tensor, config: &PadConfig) -> crate::Result<Tensor>;
        concatenate(inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor>;
        reverse(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
        }
    }

    impl TensorDot for DefaultOnlyExec {
        fn dot_general(
            &mut self,
            lhs: &Tensor,
            rhs: &Tensor,
            config: &DotGeneralConfig,
        ) -> crate::Result<Tensor> {
            CpuBackend::new().dot_general(lhs, rhs, config)
        }
    }

    impl SessionCachedDot for DefaultOnlyExec {}

    impl TensorBuffer for DefaultOnlyExec {
        fn reclaim_buffer(&mut self, _tensor: Tensor) {}
    }

    impl TensorFusion for DefaultOnlyExec {}

    let lhs = Tensor::from_vec_col_major(vec![1, 1], vec![2.0_f64]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1, 1], vec![3.0_f64]).unwrap();
    let one_shape = [1usize, 1];
    let lhs_data = [2.0_f64];
    let rhs_data = [3.0_f64];
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let mut backend = DefaultOnlyBackend;
    let mut cache = ();

    let add_read_tensor = TensorElementwise::add_read(
        &mut backend,
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
    )
    .unwrap();
    assert_eq!(add_read_tensor.as_slice::<f64>().unwrap(), &[5.0]);
    let add_view_err = TensorElementwise::add_read(
        &mut backend,
        TensorRead::from_view(TensorView::f64(&one_shape, &lhs_data).unwrap()),
        TensorRead::from_tensor(&rhs),
    )
    .unwrap_err();
    assert!(matches!(
        &add_view_err,
        crate::Error::Unsupported { op: "add", .. }
    ));
    assert_eq!(add_view_err.kind(), tenferro_tensor::ErrorKind::Unsupported);

    let reduce_input = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap();
    let reduce_view_shape = [2usize];
    let reduce_view_data = [2.0_f64, 3.0];
    assert_eq!(
        TensorReduction::reduce_sum_read(
            &mut backend,
            TensorRead::from_tensor(&reduce_input),
            &[0],
        )
        .unwrap()
        .as_slice::<f64>()
        .unwrap(),
        &[5.0]
    );
    assert_eq!(
        TensorReduction::reduce_prod_read(
            &mut backend,
            TensorRead::from_tensor(&reduce_input),
            &[0],
        )
        .unwrap()
        .as_slice::<f64>()
        .unwrap(),
        &[6.0]
    );
    assert_eq!(
        TensorReduction::reduce_max_read(
            &mut backend,
            TensorRead::from_tensor(&reduce_input),
            &[0],
        )
        .unwrap()
        .as_slice::<f64>()
        .unwrap(),
        &[3.0]
    );
    assert_eq!(
        TensorReduction::reduce_min_read(
            &mut backend,
            TensorRead::from_tensor(&reduce_input),
            &[0],
        )
        .unwrap()
        .as_slice::<f64>()
        .unwrap(),
        &[2.0]
    );
    for (op, err) in [
        (
            "reduce_sum",
            TensorReduction::reduce_sum_read(
                &mut backend,
                TensorRead::from_view(
                    TensorView::f64(&reduce_view_shape, &reduce_view_data).unwrap(),
                ),
                &[0],
            )
            .unwrap_err(),
        ),
        (
            "reduce_prod",
            TensorReduction::reduce_prod_read(
                &mut backend,
                TensorRead::from_view(
                    TensorView::f64(&reduce_view_shape, &reduce_view_data).unwrap(),
                ),
                &[0],
            )
            .unwrap_err(),
        ),
        (
            "reduce_max",
            TensorReduction::reduce_max_read(
                &mut backend,
                TensorRead::from_view(
                    TensorView::f64(&reduce_view_shape, &reduce_view_data).unwrap(),
                ),
                &[0],
            )
            .unwrap_err(),
        ),
        (
            "reduce_min",
            TensorReduction::reduce_min_read(
                &mut backend,
                TensorRead::from_view(
                    TensorView::f64(&reduce_view_shape, &reduce_view_data).unwrap(),
                ),
                &[0],
            )
            .unwrap_err(),
        ),
    ] {
        assert!(matches!(
            &err,
            crate::Error::Unsupported { op: actual_op, .. } if *actual_op == op
        ));
    }

    let direct = BackendCachedDot::dot_general_cached(
        &mut backend,
        &mut cache,
        Some(0),
        &lhs,
        &rhs,
        &config,
    )
    .unwrap();
    assert_eq!(direct.as_slice::<f64>().unwrap(), &[6.0]);

    let lhs_folded =
        TensorDot::dot_general_with_conj(&mut backend, &lhs, &rhs, &config, true, false).unwrap();
    assert_eq!(lhs_folded.as_slice::<f64>().unwrap(), &[6.0]);

    let both_folded =
        TensorDot::dot_general_with_conj(&mut backend, &lhs, &rhs, &config, true, true).unwrap();
    assert_eq!(both_folded.as_slice::<f64>().unwrap(), &[6.0]);

    let read_views = TensorDot::dot_general_read(
        &mut backend,
        TensorRead::from_view(TensorView::f64(&one_shape, &lhs_data).unwrap()),
        TensorRead::from_view(TensorView::f64(&one_shape, &rhs_data).unwrap()),
        &config,
    )
    .unwrap();
    assert_eq!(read_views.as_slice::<f64>().unwrap(), &[6.0]);

    let rhs_folded = BackendCachedDot::dot_general_with_conj_cached(
        &mut backend,
        &mut cache,
        Some(1),
        &lhs,
        &rhs,
        &config,
        false,
        true,
    )
    .unwrap();
    assert_eq!(rhs_folded.as_slice::<f64>().unwrap(), &[6.0]);

    let upload_error = backend
        .upload_host_tensor(TensorRead::from_tensor(&lhs))
        .unwrap_err();
    assert!(matches!(
        upload_error,
        crate::Error::Unsupported {
            op: "DefaultOnlyBackend::upload_host_tensor",
            ..
        }
    ));
    let download_error = backend
        .download_to_host(TensorRead::from_tensor(&lhs))
        .unwrap_err();
    assert!(matches!(
        download_error,
        crate::Error::Unsupported {
            op: "DefaultOnlyBackend::download_to_host",
            ..
        }
    ));

    let fusion_plan =
        tenferro_tensor::backend::ElementwiseFusionPlan::new(DType::F64, 0, vec![], vec![]);
    assert!(backend
        .execute_elementwise_fusion(&[], &fusion_plan)
        .unwrap()
        .is_none());

    let session_value =
        BackendSessionHost::with_backend_session_cached(&mut backend, &mut cache, |exec| {
            let cached = exec
                .dot_general_cached(Some(2), &lhs, &rhs, &config)
                .unwrap();
            let folded = exec
                .dot_general_with_conj_cached(Some(3), &lhs, &rhs, &config, true, false)
                .unwrap();
            cached.as_slice::<f64>().unwrap()[0] + folded.as_slice::<f64>().unwrap()[0]
        });
    assert_eq!(session_value, 12.0);

    let mut exec = DefaultOnlyExec;
    let exec_read_tensor = TensorDot::dot_general_read(
        &mut exec,
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
        &config,
    )
    .unwrap();
    assert_eq!(exec_read_tensor.as_slice::<f64>().unwrap(), &[6.0]);
    let exec_read_views = TensorDot::dot_general_read(
        &mut exec,
        TensorRead::from_view(TensorView::f64(&one_shape, &lhs_data).unwrap()),
        TensorRead::from_view(TensorView::f64(&one_shape, &rhs_data).unwrap()),
        &config,
    )
    .unwrap();
    assert_eq!(exec_read_views.as_slice::<f64>().unwrap(), &[6.0]);
    let exec_no_conj =
        TensorDot::dot_general_with_conj(&mut exec, &lhs, &rhs, &config, false, false).unwrap();
    assert_eq!(exec_no_conj.as_slice::<f64>().unwrap(), &[6.0]);
    let exec_lhs_conj =
        TensorDot::dot_general_with_conj(&mut exec, &lhs, &rhs, &config, true, false).unwrap();
    assert_eq!(exec_lhs_conj.as_slice::<f64>().unwrap(), &[6.0]);
    let exec_rhs_conj =
        TensorDot::dot_general_with_conj(&mut exec, &lhs, &rhs, &config, false, true).unwrap();
    assert_eq!(exec_rhs_conj.as_slice::<f64>().unwrap(), &[6.0]);
    let exec_both_conj =
        TensorDot::dot_general_with_conj(&mut exec, &lhs, &rhs, &config, true, true).unwrap();
    assert_eq!(exec_both_conj.as_slice::<f64>().unwrap(), &[6.0]);
}

#[test]
fn test_pool_backed_elementwise_public_paths_cover_dtypes_and_scalars() {
    let f32_scalar = Tensor::F32(TypedTensor::from_vec_col_major(vec![], vec![2.0]).unwrap());
    let c32_vec = Tensor::C32(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex32::new(1.0, 1.0), Complex32::new(-3.0, 0.5)],
        )
        .unwrap(),
    );
    assert_eq!(
        add(&f32_scalar, &c32_vec)
            .unwrap()
            .as_slice::<Complex32>()
            .unwrap()[0],
        Complex32::new(3.0, 1.0)
    );
    assert_eq!(
        add(&c32_vec, &f32_scalar)
            .unwrap()
            .as_slice::<Complex32>()
            .unwrap()[1],
        Complex32::new(-1.0, 0.5)
    );
    assert_eq!(
        div(&f32_scalar, &c32_vec)
            .unwrap()
            .as_slice::<Complex32>()
            .unwrap()[0],
        Complex32::new(1.0, -1.0)
    );
    assert_eq!(
        mul(&c32_vec, &f32_scalar)
            .unwrap()
            .as_slice::<Complex32>()
            .unwrap()[0],
        Complex32::new(2.0, 2.0)
    );

    let f64_scalar = Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![4.0]).unwrap());
    let c64_vec = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.0, -1.0), Complex64::new(0.0, 2.0)],
        )
        .unwrap(),
    );
    assert_c64_close(
        div(&c64_vec, &f64_scalar)
            .unwrap()
            .as_slice::<Complex64>()
            .unwrap()[1],
        Complex64::new(0.0, 0.5),
    );

    assert_eq!(
        neg(&Tensor::from_vec_col_major(vec![1], vec![1_i64]).unwrap())
            .unwrap()
            .as_slice::<i64>()
            .unwrap(),
        &[-1]
    );
    assert!(conj(&Tensor::from_vec_col_major(vec![1], vec![1_i64]).unwrap()).is_err());
    assert_eq!(
        abs(&Tensor::from_vec_col_major(vec![1], vec![-1_i64]).unwrap())
            .unwrap()
            .as_slice::<i64>()
            .unwrap(),
        &[1]
    );
    assert_eq!(
        sign(&Tensor::from_vec_col_major(vec![1], vec![-1_i64]).unwrap())
            .unwrap()
            .as_slice::<i64>()
            .unwrap(),
        &[-1]
    );

    let a = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(3.0, 4.0), Complex64::new(1.0, 0.0)],
        )
        .unwrap(),
    );
    let b = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(0.0, 2.0), Complex64::new(5.0, 0.0)],
        )
        .unwrap(),
    );
    assert!(matches!(
        maximum(&a, &b),
        Err(crate::Error::Unsupported {
            op: "maximum",
            message,
        }) if message.contains("total order")
    ));
    assert!(matches!(
        minimum(&a, &b),
        Err(crate::Error::Unsupported {
            op: "minimum",
            message,
        }) if message.contains("total order")
    ));
    assert!(matches!(
        compare(&a, &b, &CompareDir::Ge),
        Err(crate::Error::Unsupported {
            op: "compare",
            message,
        }) if message.contains("total order")
    ));
    let pred = Tensor::Bool(TypedTensor::from_vec_col_major(vec![2], vec![true, true]).unwrap());
    assert_c64_close(
        get_c64(&select(&pred, &a, &b).unwrap(), &[1]),
        Complex64::new(1.0, 0.0),
    );
    assert!(matches!(
        clamp(&a, &b, &a),
        Err(crate::Error::Unsupported {
            op: "clamp",
            message,
        }) if message.contains("total order")
    ));
}

#[test]
fn test_pool_backed_analytic_public_paths_cover_supported_dtypes() {
    let real = Tensor::from_vec_col_major(vec![2], vec![0.0_f64, 4.0]).unwrap();
    assert_f64_close(
        crate::analytic::exp(&real)
            .unwrap()
            .as_slice::<f64>()
            .unwrap()[0],
        1.0,
    );
    assert_f64_close(
        crate::analytic::sqrt(&real)
            .unwrap()
            .as_slice::<f64>()
            .unwrap()[1],
        2.0,
    );
    assert_f64_close(
        crate::analytic::rsqrt(&real)
            .unwrap()
            .as_slice::<f64>()
            .unwrap()[1],
        0.5,
    );
    assert_f64_close(
        crate::analytic::log1p(&real)
            .unwrap()
            .as_slice::<f64>()
            .unwrap()[0],
        0.0,
    );
    assert_f64_close(
        crate::analytic::expm1(&real)
            .unwrap()
            .as_slice::<f64>()
            .unwrap()[0],
        0.0,
    );

    let complex = Tensor::C64(
        TypedTensor::from_vec_col_major(vec![1], vec![Complex64::new(1.0, 0.0)]).unwrap(),
    );
    assert_c64_close(
        crate::analytic::log(&complex)
            .unwrap()
            .as_slice::<Complex64>()
            .unwrap()[0],
        Complex64::new(0.0, 0.0),
    );
    assert!(crate::analytic::sin(&complex).is_ok());
    assert!(crate::analytic::cos(&complex).is_ok());
    assert!(crate::analytic::tanh(&complex).is_ok());

    let base = Tensor::from_vec_col_major(vec![2], vec![2.0_f32, 3.0]).unwrap();
    let exponent = Tensor::from_vec_col_major(vec![2], vec![3.0_f32, 2.0]).unwrap();
    assert_eq!(
        crate::analytic::pow(&base, &exponent)
            .unwrap()
            .as_slice::<f32>()
            .unwrap(),
        &[8.0, 9.0]
    );
    let int_tensor = Tensor::from_vec_col_major(vec![1], vec![1_i64]).unwrap();
    assert!(matches!(
        crate::analytic::exp(&int_tensor),
        Err(crate::Error::UnsupportedDType {
            op: "exp",
            dtype: DType::I64,
            message,
        }) if message == "CPU backend does not support this operation for I64; supported dtypes: F32/F64/C32/C64; convert to F64 before this operation"
    ));
    assert!(matches!(
        crate::analytic::exp_read_with_pool(
            &mut crate::buffer_pool::BufferPool::new(),
            TensorRead::from_tensor(&int_tensor),
        ),
        Err(crate::Error::UnsupportedDType {
            op: "exp",
            dtype: DType::I64,
            message,
        }) if message == "CPU backend does not support this operation for I64; supported dtypes: F32/F64/C32/C64; convert to F64 before this operation"
    ));
    assert!(crate::analytic::pow(&real, &base).is_err());
}

#[test]
fn contraction_unsupported_dtype_message_lists_recovery() {
    assert_eq!(
        crate::cpu_contraction_unsupported_dtype_message(DType::I64),
        "CPU contraction providers support F32/F64/C32/C64; convert I64 to F64 before contraction"
    );
    assert_eq!(
        crate::cpu_contraction_unsupported_dtype_message(DType::Bool),
        "CPU contraction providers support F32/F64/C32/C64"
    );
}

#[test]
fn test_pool_backed_structural_public_paths_cover_dispatch_and_helpers() {
    let matrix = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let transposed = transpose(&matrix, &[1, 0]).unwrap();
    assert_eq!(transposed.shape(), &[2, 2]);
    assert_eq!(transposed.as_slice::<f64>().unwrap(), &[1.0, 3.0, 2.0, 4.0]);

    let typed = TypedTensor::from_vec_col_major(vec![2, 2], vec![1_i64, 2, 3, 4]).unwrap();
    let typed_t = crate::structural::typed_transpose(&typed, &[1, 0]).unwrap();
    assert_eq!(typed_t.host_data().unwrap(), &[1, 3, 2, 4]);

    let row = TypedTensor::from_vec_col_major(vec![1, 2], vec![5.0_f32, 6.0]).unwrap();
    let typed_b = crate::structural::typed_broadcast_in_dim(&row, &[2, 2], &[0, 1]).unwrap();
    assert_eq!(typed_b.host_data().unwrap(), &[5.0, 5.0, 6.0, 6.0]);

    let scalar = Tensor::from_vec_col_major(vec![], vec![7.0_f64]).unwrap();
    let broadcasted = broadcast_in_dim(&scalar, &[2, 2], &[]).unwrap();
    assert_eq!(
        broadcasted.as_slice::<f64>().unwrap(),
        &[7.0, 7.0, 7.0, 7.0]
    );

    let mut backend = CpuBackend::new();
    let i64_matrix = Tensor::from_vec_col_major(vec![2, 2], vec![1_i64, 2, 3, 4]).unwrap();
    let as_c64 = backend.cast(&i64_matrix, DType::C64).unwrap();
    assert_eq!(as_c64.dtype(), DType::C64);
    let as_f32 = backend.cast(&as_c64, DType::F32).unwrap();
    assert_eq!(as_f32.dtype(), DType::F32);
    let as_c32 = backend.cast(&matrix, DType::C32).unwrap();
    assert_eq!(as_c32.dtype(), DType::C32);
    let as_i64 = backend.cast(&as_c32, DType::I64).unwrap();
    assert_eq!(as_i64.as_slice::<i64>().unwrap(), &[1, 2, 3, 4]);

    let diag = extract_diagonal(&matrix, 0, 1).unwrap();
    assert_eq!(diag.as_slice::<f64>().unwrap(), &[1.0, 4.0]);
    let embedded = embed_diagonal(&diag, 0, 1).unwrap();
    assert_eq!(embedded.shape(), &[2, 2]);
    assert_eq!(embedded.as_slice::<f64>().unwrap(), &[1.0, 0.0, 0.0, 4.0]);

    let typed_diag = crate::structural::typed_extract_diagonal(
        &TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
        0,
        1,
    )
    .unwrap();
    assert_eq!(typed_diag.host_data().unwrap(), &[1.0, 4.0]);
    let typed_embedded = crate::structural::typed_embed_diagonal(&typed_diag, 0, 1).unwrap();
    assert_eq!(typed_embedded.host_data().unwrap(), &[1.0, 0.0, 0.0, 4.0]);

    let lower = tril(&matrix, 0).unwrap();
    assert_eq!(lower.as_slice::<f64>().unwrap(), &[1.0, 2.0, 0.0, 4.0]);
    let upper = triu(&matrix, 0).unwrap();
    assert_eq!(upper.as_slice::<f64>().unwrap(), &[1.0, 0.0, 3.0, 4.0]);
    let typed_lower = crate::structural::typed_tril(
        &TypedTensor::from_vec_col_major(vec![2, 2], vec![1_i64, 2, 3, 4]).unwrap(),
        0,
    )
    .unwrap();
    assert_eq!(typed_lower.host_data().unwrap(), &[1, 2, 0, 4]);
    let typed_upper = crate::structural::typed_triu(
        &TypedTensor::from_vec_col_major(vec![2, 2], vec![1_i64, 2, 3, 4]).unwrap(),
        0,
    )
    .unwrap();
    assert_eq!(typed_upper.host_data().unwrap(), &[1, 0, 3, 4]);
    assert!(crate::structural::typed_tril(
        &TypedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        0
    )
    .is_err());

    let c32_matrix = Tensor::C32(
        TypedTensor::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex32::new(1.0, 0.0),
                Complex32::new(2.0, 0.0),
                Complex32::new(3.0, 0.0),
                Complex32::new(4.0, 0.0),
            ],
        )
        .unwrap(),
    );
    assert_eq!(transpose(&c32_matrix, &[1, 0]).unwrap().dtype(), DType::C32);
    assert_eq!(tril(&c32_matrix, 0).unwrap().dtype(), DType::C32);
}

fn assert_f64_slice_eq(tensor: &Tensor, expected: &[f64]) {
    assert_eq!(tensor.as_slice::<f64>().unwrap(), expected);
}

fn exercise_read_delegate_ops<B>(backend: &mut B)
where
    B: TensorElementwise + TensorAnalytic + TensorStructural + TensorDot + ?Sized,
{
    let lhs = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 4.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 8.0]).unwrap();
    let pred = Tensor::Bool(TypedTensor::from_vec_col_major(vec![2], vec![true, false]).unwrap());
    let lower = Tensor::from_vec_col_major(vec![2], vec![1.5_f64, 3.0]).unwrap();
    let upper = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 6.0]).unwrap();
    let view = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 4.0]).unwrap();
    let matrix =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    assert_f64_slice_eq(
        &backend
            .mul_read(
                TensorRead::from_view(TensorView::F64(view.as_view())),
                TensorRead::from_tensor(&rhs),
            )
            .unwrap(),
        &[2.0, 32.0],
    );
    assert_f64_slice_eq(
        &backend
            .neg_read(TensorRead::from_view(TensorView::F64(view.as_view())))
            .unwrap(),
        &[-1.0, -4.0],
    );
    assert_f64_slice_eq(
        &backend
            .div_read(TensorRead::from_tensor(&rhs), TensorRead::from_tensor(&lhs))
            .unwrap(),
        &[2.0, 2.0],
    );
    assert_f64_slice_eq(
        &backend
            .abs_read(TensorRead::from_view(TensorView::F64(view.as_view())))
            .unwrap(),
        &[1.0, 4.0],
    );
    assert_f64_slice_eq(
        &backend.sign_read(TensorRead::from_tensor(&lhs)).unwrap(),
        &[1.0, 1.0],
    );
    assert_f64_slice_eq(
        &backend
            .maximum_read(TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs))
            .unwrap(),
        &[2.0, 8.0],
    );
    assert_f64_slice_eq(
        &backend
            .minimum_read(TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs))
            .unwrap(),
        &[1.0, 4.0],
    );
    assert_eq!(
        backend
            .compare_read(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &CompareDir::Lt,
            )
            .unwrap()
            .as_slice::<bool>()
            .unwrap(),
        &[true, true]
    );
    assert_f64_slice_eq(
        &backend
            .select_read(
                TensorRead::from_tensor(&pred),
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
            )
            .unwrap(),
        &[1.0, 8.0],
    );
    assert_f64_slice_eq(
        &backend
            .clamp_read(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&lower),
                TensorRead::from_tensor(&upper),
            )
            .unwrap(),
        &[1.5, 4.0],
    );

    assert_f64_close(
        backend
            .exp_read(TensorRead::from_tensor(&lhs))
            .unwrap()
            .as_slice::<f64>()
            .unwrap()[0],
        1.0_f64.exp(),
    );
    assert_f64_slice_eq(
        &backend.log_read(TensorRead::from_tensor(&lhs)).unwrap(),
        &[0.0, 4.0_f64.ln()],
    );
    assert_eq!(
        backend
            .sin_read(TensorRead::from_tensor(&lhs))
            .unwrap()
            .shape(),
        &[2]
    );
    assert_eq!(
        backend
            .cos_read(TensorRead::from_tensor(&lhs))
            .unwrap()
            .shape(),
        &[2]
    );
    assert_eq!(
        backend
            .tanh_read(TensorRead::from_tensor(&lhs))
            .unwrap()
            .shape(),
        &[2]
    );
    assert_f64_slice_eq(
        &backend.sqrt_read(TensorRead::from_tensor(&lhs)).unwrap(),
        &[1.0, 2.0],
    );
    assert_f64_slice_eq(
        &backend.rsqrt_read(TensorRead::from_tensor(&lhs)).unwrap(),
        &[1.0, 0.5],
    );
    assert_f64_slice_eq(
        &backend
            .pow_read(TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs))
            .unwrap(),
        &[1.0, 65536.0],
    );
    assert_eq!(
        backend
            .expm1_read(TensorRead::from_tensor(&lhs))
            .unwrap()
            .shape(),
        &[2]
    );
    assert_eq!(
        backend
            .log1p_read(TensorRead::from_tensor(&lhs))
            .unwrap()
            .shape(),
        &[2]
    );

    assert_f64_slice_eq(
        &backend
            .reshape_read(
                TensorRead::from_view(TensorView::F64(matrix.as_view())),
                &[4],
            )
            .unwrap(),
        &[1.0, 2.0, 3.0, 4.0],
    );
    assert_f64_slice_eq(
        &backend
            .broadcast_in_dim_read(
                TensorRead::from_view(TensorView::F64(view.as_view())),
                &[2, 2],
                &[0],
            )
            .unwrap(),
        &[1.0, 4.0, 1.0, 4.0],
    );

    let cfg = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let rhs_matrix = Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 6.0, 7.0, 8.0]).unwrap();
    assert_eq!(
        backend
            .dot_general_read(
                TensorRead::from_view(TensorView::F64(matrix.as_view())),
                TensorRead::from_tensor(&rhs_matrix),
                &cfg,
            )
            .unwrap()
            .shape(),
        &[2, 2]
    );
}

#[test]
fn test_backend_and_session_read_delegates_cover_non_add_ops() {
    let mut backend = CpuBackend::new();
    exercise_read_delegate_ops(&mut backend);
    backend.with_backend_session(|exec| exercise_read_delegate_ops(exec));
}
