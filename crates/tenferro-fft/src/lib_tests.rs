use super::*;

#[test]
fn fft_infer_output_meta_rejects_invalid_trait_inputs_without_panicking() {
    let op = FftOp::new(FftOperation::R2cOnesided, vec![0], None, FftNorm::Backward);
    let shape = [SymDim::from(4usize)];

    assert!(tenferro_ops::ext_op::invoke_extension_shape_inference(&op, &[], &[&shape]).is_err());
    assert!(
        tenferro_ops::ext_op::invoke_extension_shape_inference(&op, &[DType::F64], &[]).is_err()
    );
    assert!(
        tenferro_ops::ext_op::invoke_extension_shape_inference(&op, &[DType::I64], &[&shape])
            .is_err()
    );

    let bad_axis = FftOp::new(FftOperation::C2cForward, vec![2], None, FftNorm::Backward);
    assert!(tenferro_ops::ext_op::invoke_extension_shape_inference(
        &bad_axis,
        &[DType::C64],
        &[&shape]
    )
    .is_err());
}

#[test]
fn symbolic_irfft_rejects_a_length_one_spectrum_like_the_concrete_path() {
    let op = FftOp::new(FftOperation::C2r, vec![0], None, FftNorm::Backward);
    let shape = [SymDim::from(1usize)];

    let symbolic =
        tenferro_ops::ext_op::invoke_extension_shape_inference(&op, &[DType::C64], &[&shape])
            .expect_err("a length-1 one-sided spectrum has output length 0");
    let concrete =
        output_shape_c2r(&[1usize], 0, None).expect_err("the concrete path rejects the same");

    assert_eq!(symbolic.to_string(), concrete.to_string());
}

#[test]
fn symbolic_transform_length_rejects_zero_like_the_concrete_path() {
    for (operation, dtype) in [
        (FftOperation::C2cForward, DType::C64),
        (FftOperation::R2cOnesided, DType::F64),
    ] {
        let concrete = transform_len(&[0usize], 0, None)
            .expect_err("the concrete path rejects a zero transform length");

        let zero_input = FftOp::new(operation, vec![0], None, FftNorm::Backward);
        let shape = [SymDim::from(0usize)];
        let symbolic = tenferro_ops::ext_op::invoke_extension_shape_inference(
            &zero_input,
            &[dtype],
            &[&shape],
        )
        .expect_err("a symbolic zero extent must be rejected the same way");
        assert_eq!(symbolic.to_string(), concrete.to_string(), "{operation:?}");

        let zero_length = FftOp::new(operation, vec![0], Some(vec![0]), FftNorm::Backward);
        let shape = [SymDim::from(4usize)];
        let symbolic = tenferro_ops::ext_op::invoke_extension_shape_inference(
            &zero_length,
            &[dtype],
            &[&shape],
        )
        .expect_err("an explicit zero transform length must be rejected");
        assert_eq!(symbolic.to_string(), concrete.to_string(), "{operation:?}");
    }
}

#[test]
fn checked_shape_product_rejects_overflow_before_allocation() {
    let err = cpu::checked_shape_product("fft", "output", &[usize::MAX, 2])
        .expect_err("overflowing output shape should be rejected");

    assert!(err.to_string().contains("overflows usize"), "{err}");
}

#[test]
fn irfft_default_output_length_rejects_overflow() {
    let err = output_shape_c2r(&[usize::MAX], 0, None)
        .expect_err("default irfft output length should reject overflow");

    assert!(err.to_string().contains("overflows usize"), "{err}");
}

#[test]
fn normalize_axis_handles_large_rank_without_isize_cast_wrap() {
    assert_eq!(normalize_axis("fft", 0, usize::MAX).unwrap(), 0);
    assert_eq!(
        normalize_axis("fft", -1, usize::MAX).unwrap(),
        usize::MAX - 1
    );
    assert!(normalize_axis("fft", isize::MIN, 3).is_err());
}

#[test]
fn out_of_range_axis_error_keeps_its_sign() {
    for axis in [-5_isize, 5] {
        match normalize_axis("fft", axis, 2) {
            Err(tenferro_tensor::Error::Validation {
                source: tenferro_tensor::ValidationError::AxisOutOfBounds { axis: got, rank },
                ..
            }) => assert_eq!((got, rank), (axis, 2)),
            other => panic!("axis {axis}: expected AxisOutOfBounds, got {other:?}"),
        }
    }
}

#[test]
fn axis_lane_layout_rejects_stride_overflow() {
    let err = cpu::LaneLayout::new(&[usize::MAX, 2], 1, 2)
        .expect_err("lane layout should reject stride overflow");

    assert!(err.to_string().contains("overflows usize"), "{err}");
}

#[cfg(feature = "autodiff")]
#[test]
fn fft_semantic_rules_emit_extension_first_jvp_and_length_restoring_transpose() {
    use tenferro_ops::dim_expr::DimExpr;
    use tenferro_runtime::program::{ProgramInputSpec, SemanticOpRef, SemanticProgramBuilder};

    let fft_op = FftOp::new(
        FftOperation::C2cForward,
        vec![0],
        Some(vec![2]),
        FftNorm::Backward,
    );
    let mut source = SemanticProgramBuilder::new();
    let source_input = source
        .input(ProgramInputSpec::new(DType::C64, [DimExpr::Const(4)]))
        .unwrap();
    let source_output = source
        .add_extension(Arc::new(fft_op), &[source_input])
        .unwrap()[0];
    let source = source.finish(&[source_output]).unwrap();
    let operation = source.program.operations().next().unwrap();

    let rules = semantic_ad_rules().unwrap();
    let mut destination = SemanticProgramBuilder::new();
    let primal = destination
        .input(ProgramInputSpec::new(DType::C64, [DimExpr::Const(4)]))
        .unwrap();
    let tangent = destination
        .input(ProgramInputSpec::new(DType::C64, [DimExpr::Const(4)]))
        .unwrap();
    let primal_output = destination
        .add_extension(
            Arc::new(FftOp::new(
                FftOperation::C2cForward,
                vec![0],
                Some(vec![2]),
                FftNorm::Backward,
            )),
            &[primal],
        )
        .unwrap()[0];
    let linearized = rules
        .linearize_operation(
            operation,
            &[primal],
            &[primal_output],
            &[AdValue::Value(tangent)],
            &[true],
            &mut destination,
        )
        .unwrap();
    let AdValue::Value(tangent_output) = linearized.tangent_outputs()[0] else {
        panic!("FFT tangent must be active");
    };
    let cotangent_inputs = rules
        .linear_transpose_operation(
            operation,
            &[primal],
            &[primal_output],
            &[AdValue::Value(tangent_output)],
            &[true],
            linearized.residuals(),
            &mut destination,
        )
        .unwrap();
    let AdValue::Value(cotangent_input) = cotangent_inputs[0] else {
        panic!("FFT cotangent must be active");
    };
    let frozen = destination
        .finish(&[tangent_output, cotangent_input])
        .unwrap();
    let operations: Vec<_> = frozen.program.operations().collect();
    assert!(
        operations
            .iter()
            .filter(|operation| matches!(operation.op(), SemanticOpRef::Extension(_)))
            .count()
            >= 3
    );
    assert!(operations.iter().any(|operation| matches!(
        operation.op(),
        SemanticOpRef::Core(CoreSemanticOp::DynamicTruncate { axis: 0 })
    )));
    assert!(operations.iter().any(|operation| matches!(
        operation.op(),
        SemanticOpRef::Core(CoreSemanticOp::PadToMatch { axis: 0 })
    )));
}

#[cfg(feature = "autodiff")]
#[test]
fn fft_semantic_rules_run_through_whole_program_jvp_and_vjp() {
    use tenferro_ad::AdContext;
    use tenferro_ops::dim_expr::DimExpr;
    use tenferro_runtime::program::{ProgramInputSpec, SemanticOpRef, SemanticProgramBuilder};

    let mut builder = SemanticProgramBuilder::new();
    let input = builder
        .input(ProgramInputSpec::new(DType::C64, [DimExpr::Const(4)]))
        .unwrap();
    let output = builder
        .add_extension(
            Arc::new(FftOp::new(
                FftOperation::C2cForward,
                vec![0],
                Some(vec![2]),
                FftNorm::Backward,
            )),
            &[input],
        )
        .unwrap()[0];
    let source = builder.finish(&[output]).unwrap();
    let ad = AdContext::builder()
        .with_semantic_extension_rules(semantic_ad_rules().unwrap())
        .unwrap()
        .build()
        .unwrap();

    let jvp = ad.jvp_program(&source, &[true]).unwrap();
    assert_eq!(jvp.derivative_input_indices(), &[Some(1)]);
    assert!(matches!(
        jvp.frozen().program.operations().last().unwrap().op(),
        SemanticOpRef::Extension(op) if op.family_id() == FFT_EXTENSION_FAMILY_ID
    ));

    let vjp = ad.vjp_program(&source, &[true], &[true]).unwrap();
    assert_eq!(vjp.derivative_output_indices(), &[Some(0)]);
    assert!(vjp.frozen().program.operations().any(|operation| matches!(
        operation.op(),
        SemanticOpRef::Core(CoreSemanticOp::PadToMatch { axis: 0 })
            | SemanticOpRef::Core(CoreSemanticOp::DynamicTruncate { axis: 0 })
    )));
}
