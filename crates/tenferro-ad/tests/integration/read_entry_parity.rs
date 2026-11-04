//! #2223: eager and traced callers hand a borrowed operand to one `*_read`
//! entry, so both produce the same values for a compact and a strided input.
//!
//! The eager input here is a real view: an untracked eager metadata op keeps a
//! borrowed layout, and the eager dispatch passes `EagerTensor::tensor_read()`
//! straight to the backend entry. The traced side binds the same logical tensor
//! twice, once as a compact borrowed read and once as the strided view itself.

use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_runtime::{
    DType, ExecutionEngine, GraphCompiler, PadConfig, Runtime, SliceConfig, Tensor, TensorRead,
    TracedTensor,
};

use crate::support::cpu_runtime;

fn f64_values(tensor: &Tensor) -> Vec<f64> {
    tensor.as_slice::<f64>().expect("f64 payload").to_vec()
}

/// Bind one input read and run the traced output on a fresh serial CPU engine.
fn run_traced_on_read(
    output: &TracedTensor,
    placeholder: &TracedTensor,
    read: TensorRead<'_>,
) -> Tensor {
    let runtime: Runtime = cpu_runtime();
    let shape = read.shape().to_vec();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(output, &[(placeholder, read.dtype(), &shape)])
        .expect("compile the traced output");

    let mut engine = tenferro_cpu::CpuEngine::serial();
    let target = engine.target();
    let reads = [read];
    let prepared = runtime
        .prepare(&program, &reads, std::slice::from_ref(&target))
        .expect("prepare for the bound read");
    let outputs = runtime
        .run(&mut [&mut engine], &prepared, &reads)
        .expect("run the prepared program");
    let mut outputs = outputs
        .into_iter()
        .map(|output| {
            output
                .into_tensor(&mut engine)
                .expect("materialize a traced output")
        })
        .collect::<Vec<_>>();
    assert_eq!(outputs.len(), 1, "expected one traced output");
    outputs.pop().expect("one output")
}

/// The eager value must match the traced value on both bound reads.
fn assert_read_parity(
    label: &str,
    eager: &EagerTensor,
    traced_output: &TracedTensor,
    placeholder: &TracedTensor,
    compact: TensorRead<'_>,
    strided: TensorRead<'_>,
) {
    let eager = eager
        .to_tensor(&mut tenferro_cpu::CpuEngine::serial())
        .expect("materialize the eager output");
    for (flavour, read) in [("compact", compact), ("strided", strided)] {
        let traced = run_traced_on_read(traced_output, placeholder, read);
        assert_eq!(traced.shape(), eager.shape(), "{label} on a {flavour} read");
        assert_eq!(
            f64_values(&traced),
            f64_values(&eager),
            "{label} on a {flavour} read"
        );
    }
}

#[test]
fn eager_and_traced_agree_on_a_borrowed_view_operand() {
    let ctx = EagerRuntime::new().expect("eager runtime");
    // `2 x 3 x 4` column-major, so the transposed rank-3 view below is not
    // compact: shape `[3, 4, 2]` with strides `[2, 6, 1]`.
    let source = Tensor::from_vec_col_major(
        vec![2, 3, 4],
        (0..24).map(|index| index as f64).collect::<Vec<_>>(),
    )
    .expect("source tensor");
    let owned = EagerTensor::from_tensor_in(source, ctx.clone()).expect("eager source");
    let view = ctx
        .with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
            s.transpose(&owned, &[1, 2, 0])
        })
        .expect("untracked transpose");
    assert!(
        matches!(view.tensor_read(), TensorRead::View(_)),
        "the eager input must stay a borrowed view"
    );
    assert_eq!(view.shape(), &[3, 4, 2]);

    let logical = view
        .to_tensor(&mut tenferro_cpu::CpuEngine::serial())
        .expect("materialize the eager input");
    let compact = TensorRead::from_view(TensorRead::from_tensor(&logical).tensor_view());
    let strided = view.tensor_read();
    let placeholder = TracedTensor::input_symbolic_shape(DType::F64, 3).expect("placeholder");

    let eager =
        |f: &dyn Fn(&mut tenferro_ad::EagerSession<'_>) -> tenferro_ad::Result<EagerTensor>| {
            ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| f(s))
                .expect("eager op")
        };

    let out = eager(&|s| s.tril(&view, -1));
    let traced = placeholder.tril(-1).expect("traced tril");
    assert_read_parity(
        "tril",
        &out,
        &traced,
        &placeholder,
        compact.clone(),
        strided.clone(),
    );

    let out = eager(&|s| s.triu(&view, 1));
    let traced = placeholder.triu(1).expect("traced triu");
    assert_read_parity(
        "triu",
        &out,
        &traced,
        &placeholder,
        compact.clone(),
        strided.clone(),
    );

    let out = eager(&|s| s.extract_diag(&view, 1, 2));
    let traced = placeholder.extract_diag(1, 2).expect("traced extract_diag");
    assert_read_parity(
        "extract_diag",
        &out,
        &traced,
        &placeholder,
        compact.clone(),
        strided.clone(),
    );

    let out = eager(&|s| s.embed_diag(&view, 0, 1));
    let traced = placeholder.embed_diag(0, 1).expect("traced embed_diag");
    assert_read_parity(
        "embed_diag",
        &out,
        &traced,
        &placeholder,
        compact.clone(),
        strided.clone(),
    );

    let pad_config = PadConfig {
        edge_padding_low: vec![1, 0, 0],
        edge_padding_high: vec![0, 1, 1],
        interior_padding: vec![0, 1, 0],
    };
    let out = eager(&|s| s.pad(&view, pad_config.clone()));
    let traced = placeholder.pad(pad_config).expect("traced pad");
    assert_read_parity(
        "pad",
        &out,
        &traced,
        &placeholder,
        compact.clone(),
        strided.clone(),
    );

    let out = eager(&|s| s.reverse(&view, &[0, 2]));
    let traced = placeholder.reverse(&[0, 2]).expect("traced reverse");
    assert_read_parity(
        "reverse",
        &out,
        &traced,
        &placeholder,
        compact.clone(),
        strided.clone(),
    );

    let slice_config = SliceConfig {
        starts: vec![1, 0, 0],
        limits: vec![3, 4, 2],
        strides: vec![1, 2, 1],
    };
    let out = eager(&|s| s.slice(&view, slice_config.clone()));
    let traced = placeholder.slice(slice_config).expect("traced slice");
    assert_read_parity(
        "slice",
        &out,
        &traced,
        &placeholder,
        compact.clone(),
        strided.clone(),
    );

    let out = eager(&|s| s.concatenate(&[&view, &view], 0));
    let traced =
        TracedTensor::concatenate(&[&placeholder, &placeholder], 0).expect("traced concat");
    assert_read_parity(
        "concatenate",
        &out,
        &traced,
        &placeholder,
        compact.clone(),
        strided.clone(),
    );

    let out = ctx
        .with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
            let size = s.constant_from(Tensor::from_vec_col_major(vec![], vec![2.0_f64])?)?;
            s.apply_standard_op(StdTensorOp::DynamicTruncate { axis: 0 }, &[&view, &size])
        })
        .expect("eager dynamic_truncate");
    let size = TracedTensor::from_vec_col_major(vec![], vec![2.0_f64]).expect("traced size");
    let traced = placeholder
        .dynamic_truncate(&size, 0)
        .expect("traced dynamic_truncate");
    assert_read_parity(
        "dynamic_truncate",
        &out,
        &traced,
        &placeholder,
        compact.clone(),
        strided.clone(),
    );

    let out = ctx
        .with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
            let reference = s.constant_from(Tensor::from_vec_col_major(
                vec![5, 4, 2],
                vec![0.0_f64; 40],
            )?)?;
            s.apply_standard_op(StdTensorOp::PadToMatch { axis: 0 }, &[&view, &reference])
        })
        .expect("eager pad_to_match");
    let reference = TracedTensor::from_vec_col_major(vec![5, 4, 2], vec![0.0_f64; 40])
        .expect("traced reference");
    let traced = placeholder
        .pad_to_match(&reference, 0)
        .expect("traced pad_to_match");
    assert_read_parity(
        "pad_to_match",
        &out,
        &traced,
        &placeholder,
        compact,
        strided,
    );
}
