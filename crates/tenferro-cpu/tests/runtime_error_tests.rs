use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use tenferro_cpu::CpuBackend;
use tenferro_tensor::{
    Buffer, BufferHandle, DeviceId, DeviceKind, DotGeneralConfig, Error, GpuBackendKind,
    MemoryKind, PadConfig, Placement, ScatterConfig, SliceConfig, Tensor, TensorAnalytic,
    TensorDeviceTransfer, TensorDot, TensorElementwise, TensorIndexing, TensorStructural,
    TypedTensor,
};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn f32_tensor(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
    Tensor::F32(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn i64_tensor(shape: Vec<usize>, data: Vec<i64>) -> Tensor {
    Tensor::I64(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn source_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start = source.find(start).expect("section start should exist");
    let tail = &source[start..];
    let end = tail.find(end).expect("section end should exist");
    &tail[..end]
}

fn backend_f64_tensor(shape: Vec<usize>) -> Tensor {
    let len = shape.iter().product();
    Tensor::F64(
        TypedTensor::from_buffer_col_major(
            shape,
            Buffer::Backend(Arc::new(BufferHandle::<f64>::new_with_len(7, len))),
            Placement {
                memory_kind: MemoryKind::Device,
                device: Some(DeviceId {
                    kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                    ordinal: 0,
                }),
            },
        )
        .unwrap(),
    )
}

#[test]
fn backend_failure_helper_preserves_op_and_message() {
    let err = Error::backend_failure("custom_op", "device rejected launch");

    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "custom_op",
            ref message,
        } if message == "device rejected launch"
    ));
}

#[test]
fn cpu_device_transfer_rejects_backend_buffers_at_boundary() {
    let mut backend = CpuBackend::new();
    let tensor = backend_f64_tensor(vec![2]);

    for (actual, expected_op, expected_hint) in [
        (
            backend.download_to_host(&tensor),
            "CpuBackend::download_to_host",
            "download",
        ),
        (
            backend.upload_host_tensor(&tensor),
            "CpuBackend::upload_host_tensor",
            "download",
        ),
    ] {
        let err = actual.unwrap_err();
        assert!(matches!(
            err,
            Error::BackendFailure {
                op,
                ref message,
            } if op == expected_op && message.contains(expected_hint)
        ));
    }
}

#[test]
fn cpu_linalg_dispatch_does_not_use_panic_catching_as_error_handling() {
    let backend_dispatch = include_str!("../src/backend.rs");
    let exec_session_dispatch = include_str!("../src/exec_session.rs");

    assert!(
        !backend_dispatch.contains("catch_backend_panic"),
        "CpuBackend should return typed errors from linalg helpers instead of catching panics"
    );
    assert!(
        !exec_session_dispatch.contains("catch_backend_panic"),
        "CpuExecSession should return typed errors from linalg helpers instead of catching panics"
    );
    assert!(
        !backend_dispatch.contains("catch_unwind"),
        "CPU backend error handling should not depend on panic unwinding"
    );
}

#[test]
fn cpu_pooled_output_allocation_uses_checked_shape_product() {
    let indexing_alloc = include_str!("../src/indexing_alloc.rs");
    assert!(
        indexing_alloc.contains("checked_shape_product(\"cpu_pooled_output\", &shape)?"),
        "CPU pooled output allocation must reject shape-product overflow"
    );
    assert!(
        !indexing_alloc.contains("let len = shape.iter().product();"),
        "CPU pooled output allocation must not use unchecked shape.iter().product()"
    );
}

#[test]
fn cpu_zero_fill_pooled_outputs_use_checked_shape_product() {
    let structural = include_str!("../src/structural.rs");
    let filled_section = source_section(
        structural,
        "fn filled_tensor_from_pool",
        "fn clone_host_tensor_from_pool",
    );

    assert!(
        filled_section.contains("checked_shape_product(op, \"output shape\", &shape)?"),
        "CPU zero/fill pooled output allocation must reject shape-product overflow"
    );
    assert!(
        !filled_section.contains("let len = shape.iter().product();"),
        "CPU zero/fill pooled output allocation must not use unchecked shape.iter().product()"
    );
}

#[test]
fn cpu_uninit_pooled_output_allocation_uses_checked_shape_product() {
    let cpu_lib = include_str!("../src/lib.rs");
    let section = source_section(
        cpu_lib,
        "pub(crate) unsafe fn typed_array_uninit_from_pool",
        "pub(crate) fn tensor_from_array",
    );

    assert!(
        section.contains("checked_shape_product("),
        "CPU uninitialized pooled output allocation must reject shape-product overflow"
    );
    assert!(
        !section.contains("shape.iter().product"),
        "CPU uninitialized pooled output allocation must not use unchecked shape.iter().product()"
    );
}

#[test]
fn affinity_unsafe_blocks_document_safety_invariants() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/affinity.rs");
    let source = std::fs::read_to_string(&path).expect("affinity.rs should be readable");
    let lines: Vec<_> = source.lines().collect();

    for (idx, line) in lines.iter().enumerate() {
        if !line.contains("unsafe {") {
            continue;
        }

        let has_safety_comment = lines
            .iter()
            .take(idx)
            .rev()
            .take(4)
            .any(|line| line.trim_start().starts_with("// SAFETY:"));
        assert!(
            has_safety_comment,
            "src/affinity.rs:{} missing SAFETY comment before unsafe block: {line}",
            idx + 1
        );
    }
}

#[test]
fn cpu_reshape_concatenate_scatter_use_checked_boundary_arithmetic_contract() {
    let structural = include_str!("../src/structural.rs");
    let reshape_section = source_section(
        structural,
        "pub fn typed_reshape",
        "pub(crate) fn typed_broadcast_in_dim",
    );
    assert!(
        reshape_section.contains("checked_shape_product(\"reshape\", \"input shape\","),
        "CPU reshape must check input shape products"
    );
    assert!(
        reshape_section.contains("checked_shape_product(\"reshape\", \"output shape\","),
        "CPU reshape must check output shape products"
    );
    assert!(
        !reshape_section.contains("shape.iter().product"),
        "CPU reshape must not use unchecked shape.iter().product()"
    );

    let indexing = include_str!("../src/indexing.rs");
    let concat_section = source_section(indexing, "fn typed_concatenate", "fn typed_reverse");
    assert!(
        concat_section.contains("axis_extent = axis_extent.checked_add"),
        "CPU concatenate must check output-axis extent accumulation"
    );
    assert!(
        concat_section.contains("segment_end")
            && concat_section.contains(".checked_add(input.shape()[axis])"),
        "CPU concatenate must check segment prefix offsets"
    );

    let scatter_section = source_section(indexing, "fn typed_scatter", "fn typed_dynamic_slice");
    assert!(
        !source_section(indexing, "fn typed_gather", "fn typed_scatter")
            .contains("let _ = component"),
        "CPU gather validation should not need placeholder assignments for enumerated components"
    );
    assert!(
        !scatter_section.contains("window_fits"),
        "CPU scatter must not keep dead window-fit branches after clamp_window_start validation"
    );
    assert!(
        !scatter_section
            .contains("checked_product(\"scatter\", \"batch shape\", &batch_shape)?.max(1)"),
        "CPU scatter must not force a phantom iteration over zero-size batch domains"
    );
    assert!(
        !scatter_section.contains(
            "checked_product(\"scatter\", \"window update shape\", &window_shape_updates)?.max(1)"
        ),
        "CPU scatter must not force a phantom iteration over zero-size update windows"
    );
}

#[test]
fn cpu_reductions_use_common_empty_axes_validation_helpers() {
    let reduction = include_str!("../src/reduction.rs");

    for (start, end, empty_check) in [
        (
            "fn reduction_empty_axes_noop",
            "fn reduction_read_empty_axes_noop",
            "axes.is_empty()",
        ),
        (
            "fn reduction_read_empty_axes_noop",
            "fn nan_propagating_max",
            "if !axes.is_empty()",
        ),
    ] {
        let section = source_section(reduction, start, end);
        let validate_pos = section
            .find("validate_axes(op, axes, input.shape().len())?;")
            .unwrap_or_else(|| panic!("{start} should validate axes through the common helper"));
        let empty_pos = section
            .find(empty_check)
            .unwrap_or_else(|| panic!("{start} should have an empty-axis fast path"));
        assert!(
            validate_pos < empty_pos,
            "{start} must validate axes before empty-axis fast path"
        );
    }

    for (start, end, helper, op) in [
        (
            "pub fn reduce_sum",
            "pub(crate) fn reduce_sum_read",
            "reduction_empty_axes_noop",
            "reduce_sum",
        ),
        (
            "pub(crate) fn reduce_sum_read",
            "pub fn reduce_prod",
            "reduction_read_empty_axes_noop",
            "reduce_sum",
        ),
        (
            "pub fn reduce_prod",
            "pub(crate) fn reduce_prod_read",
            "reduction_empty_axes_noop",
            "reduce_prod",
        ),
        (
            "pub(crate) fn reduce_prod_read",
            "pub fn reduce_max",
            "reduction_read_empty_axes_noop",
            "reduce_prod",
        ),
        (
            "pub fn reduce_max",
            "pub(crate) fn reduce_max_read",
            "reduction_empty_axes_noop",
            "reduce_max",
        ),
        (
            "pub(crate) fn reduce_max_read",
            "pub fn reduce_min",
            "reduction_read_empty_axes_noop",
            "reduce_max",
        ),
        (
            "pub fn reduce_min",
            "pub(crate) fn reduce_min_read",
            "reduction_empty_axes_noop",
            "reduce_min",
        ),
        (
            "pub(crate) fn reduce_min_read",
            "fn typed_reduce<",
            "reduction_read_empty_axes_noop",
            "reduce_min",
        ),
    ] {
        let section = source_section(reduction, start, end);
        let call = if helper == "reduction_read_empty_axes_noop" {
            format!("{helper}(buffers, \"{op}\",")
        } else {
            format!("{helper}(\"{op}\",")
        };
        assert!(
            section.contains(&call),
            "{start} should route empty-axis handling through {helper}"
        );
    }
}

#[test]
fn cpu_backend_with_threads_rejects_zero_without_panicking() {
    let err = match CpuBackend::with_threads(0) {
        Ok(_) => panic!("zero threads should be rejected"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        tenferro_cpu::CpuBackendError::Tensor(Error::InvalidConfig {
            op: "CpuBackend::with_threads",
            ..
        })
    ));
}

#[test]
fn dot_general_rejects_out_of_bounds_contracting_dim() {
    let lhs = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let rhs = f64_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let mut backend = CpuBackend::new();

    let err = backend
        .dot_general(
            &lhs,
            &rhs,
            &DotGeneralConfig {
                lhs_contracting_dims: vec![2],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        )
        .unwrap_err();

    assert!(matches!(
        err,
        Error::AxisOutOfBounds {
            op: "dot_general",
            axis: 2,
            rank: 2,
        }
    ));
}

#[test]
fn add_rejects_shape_mismatch() {
    let lhs = f64_tensor(vec![2], vec![1.0, 2.0]);
    let rhs = f64_tensor(vec![3], vec![3.0, 4.0, 5.0]);
    let mut backend = CpuBackend::new();

    let err = <CpuBackend as TensorElementwise>::add(&mut backend, &lhs, &rhs).unwrap_err();

    assert!(matches!(
        err,
        Error::ShapeMismatch {
            op: "add",
            lhs,
            rhs,
        } if lhs == vec![2] && rhs == vec![3]
    ));
}

#[test]
fn cpu_backend_rejects_backend_buffers_without_panicking() {
    let lhs = backend_f64_tensor(vec![2]);
    let rhs = f64_tensor(vec![2], vec![3.0, 4.0]);
    let mut backend = CpuBackend::new();

    let result = catch_unwind(AssertUnwindSafe(|| {
        <CpuBackend as TensorElementwise>::add(&mut backend, &lhs, &rhs)
    }));

    assert!(result.is_ok(), "CPU backend should return Err, not panic");
    let err = result.unwrap().unwrap_err();
    assert!(matches!(err, Error::BackendFailure { op: "add", .. }));
    assert!(err.to_string().contains("download to host"));
}

#[test]
fn transpose_returns_error_instead_of_panicking() {
    let input = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let mut backend = CpuBackend::new();

    let result = catch_unwind(AssertUnwindSafe(|| backend.transpose(&input, &[0])));

    assert!(result.is_ok(), "transpose should return Err, not panic");
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "transpose",
            ..
        } | Error::InvalidConfig {
            op: "transpose",
            ..
        } | Error::RankMismatch {
            op: "transpose",
            ..
        } | Error::AxisOutOfBounds {
            op: "transpose",
            ..
        } | Error::DuplicateAxis {
            op: "transpose",
            ..
        }
    ));
}

#[test]
fn reshape_returns_error_instead_of_panicking() {
    let input = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let mut backend = CpuBackend::new();

    let result = catch_unwind(AssertUnwindSafe(|| backend.reshape(&input, &[3])));

    assert!(result.is_ok(), "reshape should return Err, not panic");
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::BackendFailure { op: "reshape", .. }
            | Error::InvalidConfig { op: "reshape", .. }
            | Error::ShapeMismatch { op: "reshape", .. }
    ));
}

#[test]
fn pow_returns_error_on_shape_mismatch_instead_of_panicking() {
    let lhs = f64_tensor(vec![2], vec![1.0, 2.0]);
    let rhs = f64_tensor(vec![1], vec![3.0]);
    let mut backend = CpuBackend::new();

    let result = catch_unwind(AssertUnwindSafe(|| backend.pow(&lhs, &rhs)));

    assert!(result.is_ok(), "pow should return Err, not panic");
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::ShapeMismatch { op: "pow", .. } | Error::BackendFailure { op: "pow", .. }
    ));
}

#[test]
fn slice_returns_error_instead_of_panicking() {
    let input = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let mut backend = CpuBackend::new();
    let config = SliceConfig {
        starts: vec![0],
        limits: vec![2],
        strides: vec![1],
    };

    let result = catch_unwind(AssertUnwindSafe(|| backend.slice(&input, &config)));

    assert!(result.is_ok(), "slice should return Err, not panic");
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::BackendFailure { op: "slice", .. }
            | Error::InvalidConfig { op: "slice", .. }
            | Error::RankMismatch { op: "slice", .. }
            | Error::AxisOutOfBounds { op: "slice", .. }
    ));
}

#[test]
fn pad_returns_error_instead_of_panicking() {
    let input = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let mut backend = CpuBackend::new();
    let config = PadConfig {
        edge_padding_low: vec![0, 0],
        edge_padding_high: vec![0],
        interior_padding: vec![0, 0],
    };

    let result = catch_unwind(AssertUnwindSafe(|| backend.pad(&input, &config)));

    assert!(result.is_ok(), "pad should return Err, not panic");
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::BackendFailure { op: "pad", .. }
            | Error::InvalidConfig { op: "pad", .. }
            | Error::RankMismatch { op: "pad", .. }
    ));
}

#[test]
fn concatenate_returns_error_on_empty_inputs() {
    let mut backend = CpuBackend::new();
    let inputs: Vec<&Tensor> = vec![];

    let result = catch_unwind(AssertUnwindSafe(|| backend.concatenate(&inputs, 0)));

    assert!(result.is_ok(), "concatenate should return Err, not panic");
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidConfig {
            op: "concatenate",
            ..
        }
    ));
}

#[test]
fn concatenate_returns_error_on_dtype_mismatch() {
    let mut backend = CpuBackend::new();
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = f32_tensor(vec![2, 2], vec![5.0f32, 6.0, 7.0, 8.0]);

    let result = catch_unwind(AssertUnwindSafe(|| backend.concatenate(&[&a, &b], 0)));

    assert!(
        result.is_ok(),
        "concatenate should return Err on dtype mismatch, not panic"
    );
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::DTypeMismatch {
            op: "concatenate",
            ..
        }
    ));
}

#[test]
fn concatenate_returns_error_on_rank_mismatch() {
    let mut backend = CpuBackend::new();
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = f64_tensor(vec![2], vec![5.0, 6.0]);

    let result = catch_unwind(AssertUnwindSafe(|| backend.concatenate(&[&a, &b], 0)));

    assert!(
        result.is_ok(),
        "concatenate should return Err on rank mismatch, not panic"
    );
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::RankMismatch {
            op: "concatenate",
            ..
        }
    ));
}

#[test]
fn concatenate_returns_error_on_axis_out_of_bounds() {
    let mut backend = CpuBackend::new();
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = f64_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);

    let result = catch_unwind(AssertUnwindSafe(|| backend.concatenate(&[&a, &b], 5)));

    assert!(
        result.is_ok(),
        "concatenate should return Err on axis out of bounds, not panic"
    );
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::AxisOutOfBounds {
            op: "concatenate",
            ..
        }
    ));
}

#[test]
fn concatenate_returns_error_on_shape_mismatch() {
    let mut backend = CpuBackend::new();
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(
        vec![2, 4],
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
    );

    let result = catch_unwind(AssertUnwindSafe(|| backend.concatenate(&[&a, &b], 0)));

    assert!(
        result.is_ok(),
        "concatenate should return Err on shape mismatch, not panic"
    );
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::ShapeMismatch {
            op: "concatenate",
            ..
        }
    ));
}

#[test]
fn concatenate_accepts_valid_inputs() {
    let mut backend = CpuBackend::new();
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = f64_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);

    let result = backend.concatenate(&[&a, &b], 0);

    assert!(result.is_ok());
    let out = result.unwrap();
    assert_eq!(out.shape(), &[4, 2]);
}

#[test]
fn scatter_negative_start_indices_clamp_like_dynamic_slice() {
    let mut backend = CpuBackend::new();
    let operand = f64_tensor(vec![3], vec![0.0, 0.0, 0.0]);
    let scatter_indices = i64_tensor(vec![1], vec![-1]);
    let updates = f64_tensor(vec![1], vec![5.0]);
    let config = ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };

    let out = backend
        .scatter(&operand, &scatter_indices, &updates, &config)
        .unwrap();

    match out {
        Tensor::F64(inner) => assert_eq!(inner.host_data().unwrap(), &[5.0, 0.0, 0.0]),
        other => panic!("unexpected output dtype: {:?}", other.dtype()),
    }
}
