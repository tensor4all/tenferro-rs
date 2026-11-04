use tenferro_tensor::DType;

use super::ones_tensor;

#[test]
fn ones_tensor_fills_every_core_dtype_and_rejects_external_scalars() {
    use tenferro_tensor::DType;

    assert_eq!(
        ones_tensor(DType::F32, vec![2])
            .unwrap()
            .as_slice::<f32>()
            .unwrap(),
        &[1.0, 1.0]
    );
    assert_eq!(
        ones_tensor(DType::F64, vec![2])
            .unwrap()
            .as_slice::<f64>()
            .unwrap(),
        &[1.0, 1.0]
    );
    assert_eq!(
        ones_tensor(DType::I32, vec![2])
            .unwrap()
            .as_slice::<i32>()
            .unwrap(),
        &[1, 1]
    );
    assert_eq!(
        ones_tensor(DType::I64, vec![2])
            .unwrap()
            .as_slice::<i64>()
            .unwrap(),
        &[1, 1]
    );
    assert_eq!(
        ones_tensor(DType::Bool, vec![2])
            .unwrap()
            .as_slice::<bool>()
            .unwrap(),
        &[true, true]
    );
    assert_eq!(
        ones_tensor(DType::C32, vec![1])
            .unwrap()
            .as_slice::<tenferro_tensor::Complex32>()
            .unwrap(),
        &[tenferro_tensor::Complex32::new(1.0, 0.0)]
    );
    assert_eq!(
        ones_tensor(DType::C64, vec![1])
            .unwrap()
            .as_slice::<tenferro_tensor::Complex64>()
            .unwrap(),
        &[tenferro_tensor::Complex64::new(1.0, 0.0)]
    );

    // An externally defined scalar is caller-owned; the runtime refuses to
    // fabricate a core identity tensor for it instead of guessing one.
    let external = DType::External(std::any::TypeId::of::<f64>());
    let error = ones_tensor(external, vec![1]).expect_err("external dtype must be rejected");
    assert!(
        matches!(error, crate::Error::Validation { .. }),
        "expected a typed validation error, got {error:?}"
    );
}

#[test]
fn bool_ones_tensor_rejects_shape_product_overflow_without_panicking() {
    let result = std::panic::catch_unwind(|| ones_tensor(DType::Bool, vec![usize::MAX, 2]));

    assert!(
        result.is_ok(),
        "ones_tensor must return a typed error, not panic"
    );
    let err = result.unwrap().unwrap_err();

    assert!(matches!(
        err,
        crate::Error::TensorRuntime(tenferro_tensor::Error::Validation {
            op: "ones_tensor",
            ..
        })
    ));
}

#[test]
fn a_retained_pooled_value_reads_back_and_renders_its_container() {
    // A pooled value is retained through its allocation group, read back as a borrowed
    // descriptor view, and its handle renders the container it holds rather than a slot.
    let tensor =
        tenferro_tensor::Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).expect("shape");
    let retained = super::RetainedValue::from_tensor(tensor);

    assert_eq!(retained.dtype(), DType::F64);
    assert_eq!(retained.shape(), &[2]);
    let rendered = format!("{retained:?}");
    assert!(rendered.contains("RetainedValue"), "{rendered}");
    assert!(rendered.contains("Pooled"), "{rendered}");

    match retained.tensor_read().expect("a pooled read") {
        tenferro_tensor::TensorRead::View(view) => {
            assert_eq!(view.as_slice::<f64>().expect("slice"), &[1.0, 2.0]);
        }
        tenferro_tensor::TensorRead::Tensor(_) => {
            panic!("a pooled value is read as a borrowed descriptor view");
        }
    }
}

#[test]
fn input_key_rejects_computed_values() {
    // A traced leaf is an addressable program input; a computed value is not, and
    // the traced layer reports that by returning no key instead of a fabricated one.
    use crate::{TraceContext, TracedTensor};

    let trace = TraceContext::new();
    let x = trace
        .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
        .unwrap();
    let computed = x.neg().unwrap();

    assert!(computed.input_key().is_none());
    let leaf_key = x.input_key().expect("a leaf exposes its input key");

    let derived: TracedTensor = trace.derive_tensor(x.trace_value()).unwrap();
    assert_eq!(derived.input_key(), Some(leaf_key));
}

#[test]
fn traced_tensor_debug_summarizes_without_graph_payload() {
    // Debug output is a summary: rank/dtype/shape and whether data is attached,
    // never the traced graph or the tensor payload.
    use crate::TraceContext;

    let trace = TraceContext::new();
    let x = trace.from_vec_col_major(vec![1], vec![7.0_f64]).unwrap();

    let debug = format!("{x:?}");

    assert!(debug.contains("TracedTensor"), "{debug}");
    assert!(debug.contains("rank: 1"), "{debug}");
    assert!(debug.contains("has_data: true"), "{debug}");
    assert!(
        !debug.contains("7.0"),
        "payload stays out of Debug: {debug}"
    );
}

#[test]
fn derived_tensor_preserves_summary_fields() {
    // The AD transform hands the traced layer a value of its own context, and the
    // rebuilt tensor carries that value's authoritative metadata.
    use crate::{TraceContext, TracedTensor};

    let trace = TraceContext::new();
    let x = trace.from_vec_col_major(vec![1], vec![3.0_f64]).unwrap();

    let derived: TracedTensor = trace.derive_tensor(x.trace_value()).unwrap();

    assert_eq!(derived.rank, 1);
    assert_eq!(derived.dtype, DType::F64);
    assert_eq!(
        derived.sym_shape().map(<[crate::SymDim]>::to_vec),
        Some(vec![crate::SymDim::from(1)])
    );
    assert!(derived.context().same_context(&trace));
}
