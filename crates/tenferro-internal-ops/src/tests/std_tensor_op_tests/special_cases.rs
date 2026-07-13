use super::*;
use crate::ad::PrimitiveRuleBuilder;

#[test]
fn test_std_tensor_op_structural_special_cases_cover_identity_and_empty_axes() {
    let (transpose_none_result, transpose_none_graph) =
        run_linearize_case(StdTensorOp::Transpose { perm: vec![1, 0] }, 0, 0, &[false]);
    assert_eq!(transpose_none_result, vec![None]);
    assert!(transpose_none_graph.operations().is_empty());

    let (identity_transpose_result, identity_transpose_graph) =
        run_linearize_case(StdTensorOp::Transpose { perm: vec![0, 1] }, 0, 0, &[true]);
    assert!(identity_transpose_result[0].is_some());
    assert!(identity_transpose_graph.operations().is_empty());

    let reshape = StdTensorOp::Reshape {
        to_shape: shape![2, 2],
    };
    let (reshape_linear_result, reshape_linear_graph) =
        run_linearize_case(reshape.clone(), 0, 0, &[true]);
    assert!(reshape_linear_result[0].is_some());
    assert_eq!(reshape_linear_graph.operations().len(), 1);
    assert_eq!(reshape_linear_graph.operations()[0].operation, reshape);

    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let primal_in = add_input_keys(&mut builder, 904, 1);
    ad_ctx.insert_metadata(
        primal_in[0].clone(),
        TensorMeta::exact(DType::F64, sym_shape(&[2, 2])),
    );
    let tangent = builder.add_input(tensor_input_key(905));
    let identity_reshape = StdTensorOp::Reshape {
        to_shape: shape![2, 2],
    };
    let identity_reshape_result = identity_reshape
        .jvp_rule(&mut builder, &primal_in, &[], &[Some(tangent)], &mut ad_ctx)
        .unwrap();
    let identity_reshape_graph = builder.build();
    assert_eq!(identity_reshape_result, vec![Some(tangent)]);
    assert!(identity_reshape_graph.operations().is_empty());

    let (transpose_result, _, transpose_graph) = run_transpose_case(
        StdTensorOp::Transpose { perm: vec![0, 1] },
        1,
        &[true],
        true,
    );
    assert!(transpose_result[0].is_some());
    assert!(transpose_graph.operations().is_empty());

    let (broadcast_linear_result, broadcast_linear_graph) = run_linearize_case(
        StdTensorOp::BroadcastInDim {
            shape: shape![2, 2, 3],
            dims: vec![0, 2],
        },
        0,
        0,
        &[true],
    );
    assert!(broadcast_linear_result[0].is_some());
    assert_eq!(
        broadcast_linear_graph.operations()[0].operation,
        StdTensorOp::BroadcastInDim {
            shape: shape![2, 2, 3],
            dims: vec![0, 2],
        }
    );

    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let primal_in = add_input_keys(&mut builder, 906, 1);
    ad_ctx.insert_metadata(
        primal_in[0].clone(),
        TensorMeta::exact(DType::F64, sym_shape(&[2, 3])),
    );
    let tangent = builder.add_input(tensor_input_key(907));
    let identity_broadcast = StdTensorOp::BroadcastInDim {
        shape: shape![2, 3],
        dims: vec![0, 1],
    };
    let identity_broadcast_result = identity_broadcast
        .jvp_rule(&mut builder, &primal_in, &[], &[Some(tangent)], &mut ad_ctx)
        .unwrap();
    let identity_broadcast_graph = builder.build();
    assert_eq!(identity_broadcast_result, vec![Some(tangent)]);
    assert!(identity_broadcast_graph.operations().is_empty());

    let (broadcast_none_result, broadcast_none_graph) = run_linearize_case(
        StdTensorOp::BroadcastInDim {
            shape: shape![2, 2, 3],
            dims: vec![0, 2],
        },
        0,
        0,
        &[false],
    );
    assert_eq!(broadcast_none_result, vec![None]);
    assert!(broadcast_none_graph.operations().is_empty());

    let (reshape_transpose_result, _, reshape_transpose_graph) =
        run_transpose_case_with_input_shape(
            reshape,
            1,
            &[true],
            true,
            Some(vec![SymDim::from(4usize)]),
        );
    assert!(reshape_transpose_result[0].is_some());
    assert_eq!(reshape_transpose_graph.operations().len(), 1);
    assert_eq!(
        reshape_transpose_graph.operations()[0].operation,
        StdTensorOp::Reshape {
            to_shape: shape![4],
        }
    );

    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let cotangent = builder.add_input(tensor_input_key(910));
    let result = StdTensorOp::BroadcastInDim {
        shape: shape![2, 3],
        dims: vec![0, 1],
    }
    .transpose_rule(
        &mut builder,
        &[Some(cotangent)],
        &external_inputs(911, 1),
        &linear_mode(&[true]),
        &mut ad_ctx,
    )
    .unwrap();
    assert_eq!(result, vec![Some(cotangent)]);
    assert!(builder.build().operations().is_empty());

    let (tril_result, tril_graph) = run_linearize_case(StdTensorOp::Tril { k: -1 }, 0, 0, &[true]);
    assert!(tril_result[0].is_some());
    assert_eq!(
        tril_graph.operations()[0].operation,
        StdTensorOp::Tril { k: -1 }
    );

    let (triu_result, triu_graph) = run_linearize_case(StdTensorOp::Triu { k: 2 }, 0, 0, &[true]);
    assert!(triu_result[0].is_some());
    assert_eq!(
        triu_graph.operations()[0].operation,
        StdTensorOp::Triu { k: 2 }
    );

    let slice = StdTensorOp::Slice(tenferro_tensor::SliceConfig {
        starts: vec![1],
        limits: vec![5],
        strides: vec![2],
    });
    let (slice_result, slice_graph) = run_linearize_case(slice.clone(), 0, 0, &[true]);
    assert!(slice_result[0].is_some());
    assert_eq!(slice_graph.operations()[0].operation, slice.clone());

    let (transpose_slice_result, _, transpose_slice_graph) =
        run_transpose_case_with_input_shapes(slice, 1, &[true], true, &[&[5]]);
    assert!(transpose_slice_result[0].is_some());
    assert_eq!(transpose_slice_graph.operations().len(), 1);
    assert_eq!(
        transpose_slice_graph.operations()[0].operation,
        StdTensorOp::Pad(PadConfig {
            edge_padding_low: vec![1],
            edge_padding_high: vec![1],
            interior_padding: vec![1],
        })
    );

    let pad = StdTensorOp::Pad(PadConfig {
        edge_padding_low: vec![1],
        edge_padding_high: vec![2],
        interior_padding: vec![1],
    });
    let (transpose_pad_result, _, transpose_pad_graph) =
        run_transpose_case_with_input_shapes(pad, 1, &[true], true, &[&[3]]);
    assert!(transpose_pad_result[0].is_some());
    assert_eq!(transpose_pad_graph.operations().len(), 1);
    assert_eq!(
        transpose_pad_graph.operations()[0].operation,
        StdTensorOp::Slice(tenferro_tensor::SliceConfig {
            starts: vec![1],
            limits: vec![6],
            strides: vec![2],
        })
    );

    let cropped_pad = StdTensorOp::Pad(PadConfig {
        edge_padding_low: vec![-1],
        edge_padding_high: vec![0],
        interior_padding: vec![0],
    });
    let (cropped_pad_result, _, cropped_pad_graph) =
        run_transpose_case_with_input_shapes(cropped_pad, 1, &[true], true, &[&[4]]);
    assert!(cropped_pad_result[0].is_some());
    assert_eq!(cropped_pad_graph.operations().len(), 2);
    assert_eq!(
        cropped_pad_graph.operations()[0].operation,
        StdTensorOp::Slice(tenferro_tensor::SliceConfig {
            starts: vec![0],
            limits: vec![3],
            strides: vec![1],
        })
    );
    assert_eq!(
        cropped_pad_graph.operations()[1].operation,
        StdTensorOp::Pad(PadConfig {
            edge_padding_low: vec![1],
            edge_padding_high: vec![0],
            interior_padding: vec![0],
        })
    );

    let reverse = StdTensorOp::Reverse { axes: vec![0, 2] };
    let (reverse_result, reverse_graph) = run_linearize_case(reverse.clone(), 0, 0, &[true]);
    assert!(reverse_result[0].is_some());
    assert_eq!(reverse_graph.operations()[0].operation, reverse.clone());

    let (transpose_reverse_result, _, transpose_reverse_graph) =
        run_transpose_case(reverse.clone(), 1, &[true], true);
    assert!(transpose_reverse_result[0].is_some());
    assert_eq!(transpose_reverse_graph.operations()[0].operation, reverse);

    let (transpose_tril_result, _, transpose_tril_graph) =
        run_transpose_case(StdTensorOp::Tril { k: 0 }, 1, &[true], true);
    assert!(transpose_tril_result[0].is_some());
    assert_eq!(
        transpose_tril_graph.operations()[0].operation,
        StdTensorOp::Tril { k: 0 }
    );

    let (transpose_triu_result, _, transpose_triu_graph) =
        run_transpose_case(StdTensorOp::Triu { k: 1 }, 1, &[true], true);
    assert!(transpose_triu_result[0].is_some());
    assert_eq!(
        transpose_triu_graph.operations()[0].operation,
        StdTensorOp::Triu { k: 1 }
    );

    let (transpose_transpose_none_result, _, transpose_transpose_none_graph) = run_transpose_case(
        StdTensorOp::Transpose {
            perm: vec![2, 0, 1],
        },
        1,
        &[true],
        false,
    );
    assert_eq!(transpose_transpose_none_result, vec![None]);
    assert!(transpose_transpose_none_graph.operations().is_empty());

    let mut invalid_builder = GraphBuilder::<StdTensorOp>::new();
    let invalid_ct = invalid_builder.add_input(tensor_input_key(401));
    let invalid = StdTensorOp::Transpose { perm: vec![1, 1] }.transpose_rule(
        &mut invalid_builder,
        &[Some(invalid_ct)],
        &external_inputs(931, 1),
        &linear_mode(&[true]),
        &mut ShapeGuardContext::default(),
    );
    assert!(
        invalid.is_err(),
        "transpose transpose rule must reject invalid inverse permutations instead of panicking"
    );

    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let none_broadcast = StdTensorOp::BroadcastInDim {
        shape: shape![2, 2, 3],
        dims: vec![0, 2],
    }
    .transpose_rule(
        &mut builder,
        &[None],
        &external_inputs(930, 1),
        &linear_mode(&[true]),
        &mut ad_ctx,
    )
    .unwrap();
    assert_eq!(none_broadcast, vec![None]);
    assert!(builder.build().operations().is_empty());
}

#[test]
fn test_std_tensor_op_linear_transpose_rules_respect_inactive_inputs() {
    let (add_result, _, add_graph) = run_transpose_case(StdTensorOp::Add, 2, &[true, false], true);
    assert_eq!(add_result.len(), 2);
    assert!(add_result[0].is_some());
    assert_eq!(add_result[1], None);
    assert!(add_graph.operations().is_empty());

    let (neg_result, _, neg_graph) = run_transpose_case(StdTensorOp::Neg, 1, &[false], true);
    assert_eq!(neg_result, vec![None]);
    assert!(neg_graph.operations().is_empty());

    let (transpose_result, _, transpose_graph) = run_transpose_case(
        StdTensorOp::Transpose { perm: vec![1, 0] },
        1,
        &[false],
        true,
    );
    assert_eq!(transpose_result, vec![None]);
    assert!(transpose_graph.operations().is_empty());

    let (reshape_result, _, reshape_graph) = run_transpose_case_with_input_shape(
        StdTensorOp::Reshape {
            to_shape: shape![4],
        },
        1,
        &[false],
        true,
        Some(vec![SymDim::from(2usize), SymDim::from(2usize)]),
    );
    assert_eq!(reshape_result, vec![None]);
    assert!(reshape_graph.operations().is_empty());

    let (tril_result, _, tril_graph) =
        run_transpose_case(StdTensorOp::Tril { k: 0 }, 1, &[false], true);
    assert_eq!(tril_result, vec![None]);
    assert!(tril_graph.operations().is_empty());

    let (truncate_result, _, truncate_graph) = run_transpose_case(
        StdTensorOp::DynamicTruncate { axis: 0 },
        2,
        &[false, false],
        true,
    );
    assert_eq!(truncate_result, vec![None, None]);
    assert!(truncate_graph.operations().is_empty());
}

#[test]
fn test_std_tensor_op_convert_linearize_and_transpose_swap_dtypes() {
    let convert = StdTensorOp::Convert {
        from: DType::F64,
        to: DType::C64,
    };

    let (linear_result, linear_graph) = run_linearize_case(convert.clone(), 0, 0, &[true]);
    assert!(linear_result[0].is_some());
    assert_eq!(linear_graph.operations().len(), 1);
    assert_eq!(linear_graph.operations()[0].operation, convert);

    let (linear_none_result, linear_none_graph) =
        run_linearize_case(convert.clone(), 0, 0, &[false]);
    assert_eq!(linear_none_result, vec![None]);
    assert!(linear_none_graph.operations().is_empty());

    let (transpose_result, _, transpose_graph) =
        run_transpose_case(convert.clone(), 1, &[true], true);
    assert!(transpose_result[0].is_some());
    assert_eq!(transpose_graph.operations().len(), 1);
    assert_eq!(
        transpose_graph.operations()[0].operation,
        StdTensorOp::Convert {
            from: DType::C64,
            to: DType::F64,
        }
    );

    let (transpose_inactive_result, _, transpose_inactive_graph) =
        run_transpose_case(convert, 1, &[false], true);
    assert_eq!(transpose_inactive_result, vec![None]);
    assert!(transpose_inactive_graph.operations().is_empty());
}

#[test]
fn test_std_tensor_op_contraction_special_cases_cover_none_and_scalar_paths() {
    let matmul = StdTensorOp::DotGeneral {
        config: DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
    };
    let (linearize_none_result, linearize_none_graph) =
        run_linearize_case(matmul.clone(), 2, 0, &[false, false]);
    assert_eq!(linearize_none_result, vec![None]);
    assert!(linearize_none_graph.operations().is_empty());

    let (transpose_none_result, _, transpose_none_graph) =
        run_transpose_case(matmul.clone(), 2, &[true, true], false);
    assert_eq!(transpose_none_result, vec![None, None]);
    assert!(transpose_none_graph.operations().is_empty());

    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let cotangent = builder.add_input(tensor_input_key(980));
    seed_dot_general_ref_metadata(&mut ad_ctx, &external_inputs(981, 2));
    let primal_mode_result = matmul
        .transpose_rule(
            &mut builder,
            &[Some(cotangent)],
            &external_inputs(981, 2),
            &OperationRole::Primary,
            &mut ad_ctx,
        )
        .unwrap();
    assert_eq!(primal_mode_result, vec![None, None]);
    assert!(builder.build().operations().is_empty());

    let reduce = StdTensorOp::ReduceSum { axes: vec![1] };
    let (reduce_linearize_none_result, reduce_linearize_none_graph) =
        run_linearize_case(reduce.clone(), 0, 0, &[false]);
    assert_eq!(reduce_linearize_none_result, vec![None]);
    assert!(reduce_linearize_none_graph.operations().is_empty());

    let (reduce_transpose_result, _, reduce_transpose_graph) =
        run_transpose_case_with_input_shapes(reduce.clone(), 1, &[true], true, &[&[2, 3]]);
    assert!(reduce_transpose_result[0].is_some());
    assert_eq!(reduce_transpose_graph.operations().len(), 1);
    // Exact metadata is enough here, so the transpose rule does not retain a
    // runtime shape source input.
    assert_eq!(
        reduce_transpose_graph.operations()[0].operation,
        StdTensorOp::BroadcastInDim {
            shape: shape![2, 3],
            dims: vec![0],
        }
    );

    let (reduce_transpose_none_result, _, reduce_transpose_none_graph) =
        run_transpose_case(reduce, 1, &[true], false);
    assert_eq!(reduce_transpose_none_result, vec![None]);
    assert!(reduce_transpose_none_graph.operations().is_empty());

    let scalar_contract = StdTensorOp::DotGeneral {
        config: DotGeneralConfig {
            lhs_contracting_dims: vec![1, 0],
            rhs_contracting_dims: vec![0, 1],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
    };
    let (scalar_transpose_result, _, scalar_transpose_graph) =
        run_transpose_case(scalar_contract.clone(), 2, &[true, false], true);
    assert!(scalar_transpose_result[0].is_some());
    assert_eq!(scalar_transpose_result[1], None);
    assert_eq!(
        scalar_transpose_graph.operations()[0].operation,
        StdTensorOp::Reshape { to_shape: shape![] }
    );
    assert!(scalar_transpose_graph
        .operations()
        .iter()
        .all(|node| node.operation != StdTensorOp::Conj));
    assert!(matches!(
        scalar_transpose_graph.operations()[1].operation,
        StdTensorOp::DotGeneral { .. }
    ));
    assert_eq!(
        scalar_transpose_graph
            .operations()
            .last()
            .unwrap()
            .operation,
        StdTensorOp::Transpose { perm: vec![1, 0] }
    );

    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let cotangent = builder.add_input(tensor_input_key(990));
    let inputs = external_inputs(991, 2);
    for (input, shape) in inputs.iter().zip([&[2, 3][..], &[3, 2][..]]) {
        let ValueRef::External(key) = input else {
            unreachable!("external_inputs returns external refs")
        };
        ad_ctx.insert_metadata(
            key.clone(),
            TensorMeta::exact(
                DType::C64,
                shape.iter().copied().map(SymDim::from).collect(),
            ),
        );
    }
    let complex_transpose_result = scalar_contract
        .transpose_rule(
            &mut builder,
            &[Some(cotangent)],
            &inputs,
            &linear_mode(&[true, false]),
            &mut ad_ctx,
        )
        .unwrap();
    let complex_transpose_graph = builder.build();
    assert!(complex_transpose_result[0].is_some());
    assert_eq!(complex_transpose_result[1], None);
    assert!(complex_transpose_graph
        .operations()
        .iter()
        .any(|node| node.operation == StdTensorOp::Conj));
}

#[derive(Clone, Debug)]
struct RuleOnlyExt {
    family: &'static str,
}

impl ExtensionOp for RuleOnlyExt {
    fn family_id(&self) -> &'static str {
        self.family
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<RuleOnlyExt>()
            .is_some_and(|rhs| rhs.family == self.family)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        ctx: &mut crate::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        Ok(vec![(ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec())])
    }
}

#[derive(Debug)]
struct RuleOnlyIdentityAd {
    family: &'static str,
}

impl ExtensionLinearizeRule for RuleOnlyIdentityAd {
    fn family_id(&self) -> &'static str {
        self.family
    }

    fn linearize(
        &self,
        _op: &dyn ExtensionOp,
        _builder: &mut dyn PrimitiveRuleBuilder,
        _primal_in: &[ValueKey<StdTensorOp>],
        _primal_out: &[ValueKey<StdTensorOp>],
        tangent_in: &[Option<LocalValueId>],
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        Ok(vec![tangent_in[0]])
    }
}

impl ExtensionLinearTransposeRule for RuleOnlyIdentityAd {
    fn family_id(&self) -> &'static str {
        self.family
    }

    fn linear_transpose(
        &self,
        _op: &dyn ExtensionOp,
        _builder: &mut dyn PrimitiveRuleBuilder,
        cotangent_out: &[Option<LocalValueId>],
        _inputs: &[tidu::PrimitiveTransposeInput<StdTensorOp>],
        _active_mask: &[bool],
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        Ok(vec![cotangent_out[0]])
    }
}

#[test]
fn extension_linearize_uses_registered_rule() {
    let family = "stdtensor.rule_only_identity.v1";
    let rules = ExtensionRuleSet::new()
        .with_linearize(Arc::new(RuleOnlyIdentityAd { family }))
        .expect("extension rule should register");
    let op = StdTensorOp::Extension(Arc::new(RuleOnlyExt { family }));
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default().with_extension_rules(rules);
    let dx = builder.add_input(tensor_input_key(900));
    let result = op
        .jvp_rule(&mut builder, &[], &[], &[Some(dx)], &mut ad_ctx)
        .expect("registered extension rule should linearize");

    assert_eq!(result, vec![Some(dx)]);
}

#[test]
fn extension_transpose_uses_registered_rule() {
    let family = "stdtensor.rule_only_transpose.v1";
    let rules = ExtensionRuleSet::new()
        .with_linear_transpose(Arc::new(RuleOnlyIdentityAd { family }))
        .expect("extension rule should register");
    let op = StdTensorOp::Extension(Arc::new(RuleOnlyExt { family }));
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default().with_extension_rules(rules);
    let ct = builder.add_input(tensor_input_key(901));
    let result = op
        .transpose_rule(
            &mut builder,
            &[Some(ct)],
            &external_inputs(910, 1),
            &linear_mode(&[true]),
            &mut ad_ctx,
        )
        .expect("registered extension rule should transpose");

    assert_eq!(result, vec![Some(ct)]);
}

#[test]
fn extension_linearize_reports_missing_rule() {
    let family = "stdtensor.missing_rule.v1";
    let op = StdTensorOp::Extension(Arc::new(RuleOnlyExt { family }));
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let dx = builder.add_input(tensor_input_key(920));
    let err = op
        .jvp_rule(&mut builder, &[], &[], &[Some(dx)], &mut ad_ctx)
        .expect_err("missing extension rule should be an AD error");

    assert_eq!(err.rule(), ADRuleKind::Jvp);
    assert!(err.to_string().contains(family));
}
