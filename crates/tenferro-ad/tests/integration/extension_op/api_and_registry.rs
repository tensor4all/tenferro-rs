use super::*;

#[test]
fn apply_rejects_wrong_input_count() {
    let err = apply(Arc::new(TestScaleBy2), &[]).unwrap_err();

    assert!(err.to_string().contains("expects 1 inputs, got 0"));
}

#[test]
fn apply_rejects_mismatched_output_metadata_count() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let err = apply(Arc::new(TestBadOutputCount), &[&x]).unwrap_err();

    let message = err.to_string();
    assert!(message.contains("tenferro-tests.bad_output_count.v1"));
    assert!(message.contains("produced 1 output metadata entries"));
    assert!(message.contains("declared 2 outputs"));
}

#[test]
fn apply_eager_rejects_empty_input_list() {
    let err = match apply_eager(Arc::new(TestScaleBy2), &[]) {
        Ok(_) => panic!("empty eager extension input list unexpectedly succeeded"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        RuntimeError::Validation {
            phase: ErrorPhase::Execution,
            source: ValidationError::InvalidArgument {
                argument: "inputs",
                ..
            },
            ..
        }
    ));
}

#[test]
fn apply_eager_rejects_wrong_input_count() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
        ctx,
    )
    .unwrap();

    let err = match apply_eager(Arc::new(TestSwap), &[&x]) {
        Ok(_) => panic!("wrong eager extension input count unexpectedly succeeded"),
        Err(err) => err,
    };

    assert!(err.to_string().contains("expects 2 inputs, got 1"));
}

#[test]
fn apply_standard_op_rejects_wrong_input_count() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
        ctx,
    )
    .unwrap();

    let err = match tenferro_ad::extension::apply_standard_op(StdTensorOp::Add, &[&x]) {
        Ok(_) => panic!("wrong standard op input count unexpectedly succeeded"),
        Err(err) => err,
    };

    assert!(err.to_string().contains("expects 2 inputs, got 1"));
}

#[test]
fn apply_eager_rejects_cross_context_inputs() {
    let lhs_ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let rhs_ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let lhs = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
        lhs_ctx,
    )
    .unwrap();
    let rhs = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap(),
        rhs_ctx,
    )
    .unwrap();

    let err = match apply_eager(Arc::new(TestSwap), &[&lhs, &rhs]) {
        Ok(_) => panic!("cross-context eager extension inputs unexpectedly succeeded"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        tenferro_runtime::error::Error::ContextMismatch { .. }
    ));
}

#[test]
fn apply_eager_reports_missing_extension_runtime() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
        ctx,
    )
    .unwrap();

    let err = match apply_eager(Arc::new(TestScaleBy2), &[&x]) {
        Ok(_) => panic!("unregistered eager extension runtime unexpectedly succeeded"),
        Err(err) => err,
    };

    let message = err.to_string();
    assert!(message.contains("missing runtime"), "{message}");
    assert!(
        message.contains("tenferro-tests.scale_by_2.v1"),
        "{message}"
    );
}

#[test]
fn apply_eager_untracked_forward_returns_untracked_result() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    register_test_eager_runtime(&ctx, "tenferro-tests.scale_by_2.v1");
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap(),
        ctx,
    )
    .unwrap();

    let y = apply_eager(Arc::new(TestScaleBy2), &[&x])
        .expect("untracked eager extension apply")
        .into_iter()
        .next()
        .expect("single eager extension output");

    assert!(!y.tracks_grad());
    assert_eq!(y.shape(), &[3]);
    assert_eq!(
        f64_slice(y.materialized().unwrap().as_ref()),
        &[2.0, 4.0, 6.0]
    );
}

#[test]
fn graph_executor_reports_missing_extension_runtime() {
    let x = TracedTensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    let y = apply(Arc::new(TestScaleBy2), &[&x])
        .unwrap()
        .into_iter()
        .next()
        .unwrap();

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let err = match y.run_with(&mut engine) {
        Ok(_) => panic!("unregistered traced extension runtime unexpectedly succeeded"),
        Err(err) => err,
    };

    let message = err.to_string();
    assert!(message.contains("missing runtime"), "{message}");
    assert!(
        message.contains("tenferro-tests.scale_by_2.v1"),
        "{message}"
    );
}

#[test]
fn apply_eager_rejects_mismatched_output_count() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    register_test_eager_runtime(&ctx, "tenferro-tests.bad_output_count.v1");
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
        ctx,
    )
    .unwrap();

    let err = match apply_eager(Arc::new(TestBadOutputCount), &[&x]) {
        Ok(_) => panic!("bad eager extension output count unexpectedly succeeded"),
        Err(err) => err,
    };

    let message = err.to_string();
    assert!(
        message.contains("tenferro-tests.bad_output_count.v1"),
        "{message}"
    );
    assert!(message.contains("runtime returned 1 outputs"), "{message}");
    assert!(message.contains("op declared 2 outputs"), "{message}");
}

#[test]
fn scale_by_2_grad_through_symbolic_placeholder() {
    // Same loss but with an input_symbolic_shape placeholder so the AD
    // path exercises the deferred zero-tangent policy (spec
    // Section 10).
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let scaled = apply(Arc::new(TestScaleBy2), &[&x])
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    let loss = scaled.reduce_sum(Some(&[0])).unwrap();

    let g = scale_by_2_ad_context().grad(&loss, &x).expect("grad build");
    let bound =
        Tensor::from_vec_col_major(vec![5], vec![10.0_f64, 20.0, 30.0, 40.0, 50.0]).unwrap();
    assert!(
        !compiled_program_contains_extension_with_specs(&g, &[(&x, bound.dtype(), bound.shape())]),
        "symbolic sum(scale_by_2(x)) grad should not retain the forward extension as a shape-only dependency"
    );

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let grad_out = g
        .run_with_inputs_auto(&mut engine, &[(&x, &bound)])
        .expect("grad eval");

    assert_eq!(grad_out.shape(), &[5]);
    assert_eq!(f64_slice(&grad_out), &[2.0, 2.0, 2.0, 2.0, 2.0]);
}

#[test]
fn swap_forward_roundtrip() {
    let a = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let b = TracedTensor::from_vec_col_major(vec![2], vec![100.0_f64, 200.0]).unwrap();
    let outputs = apply(Arc::new(TestSwap), &[&a, &b]).unwrap();
    assert_eq!(outputs.len(), 2);
    let mut iter = outputs.into_iter();
    let out_first = iter.next().unwrap();
    let out_second = iter.next().unwrap();

    let mut engine = GraphExecutor::new(CpuBackend::new());
    register_test_runtime(&mut engine, "tenferro-tests.swap.v1");
    let t0 = out_first.run_with(&mut engine).unwrap().clone();
    let t1 = out_second.run_with(&mut engine).unwrap().clone();

    // (a, b) -> (b, a)
    assert_eq!(f64_slice(&t0), &[100.0, 200.0]);
    assert_eq!(f64_slice(&t1), &[1.0, 2.0]);
}

#[test]
fn swap_grad_routes_cotangents_across_inputs() {
    // loss = sum(out0 + out1)   where (out0, out1) = swap(a, b)
    //       = sum(b + a)
    // => dloss/da = ones_like(a),  dloss/db = ones_like(b)
    let a = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let b = TracedTensor::from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0]).unwrap();
    let swapped = apply(Arc::new(TestSwap), &[&a, &b]).unwrap();
    let mut iter = swapped.into_iter();
    let out0 = iter.next().unwrap();
    let out1 = iter.next().unwrap();
    let combined = (&out0 + &out1).unwrap();
    let loss = combined.reduce_sum(Some(&[0])).unwrap();

    let ad = swap_ad_context();
    let grad_a = ad.grad(&loss, &a).expect("grad a");
    let grad_b = ad.grad(&loss, &b).expect("grad b");

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ga = grad_a.run_with(&mut engine).unwrap().clone();
    let gb = grad_b.run_with(&mut engine).unwrap().clone();

    assert_eq!(f64_slice(&ga), &[1.0, 1.0, 1.0]);
    assert_eq!(f64_slice(&gb), &[1.0, 1.0, 1.0]);
}

#[test]
fn swap_grad_routes_only_through_active_output() {
    // loss = sum(out1)  where (out0, out1) = swap(a, b)  (so out1 = a)
    // => dloss/da = ones_like(a)
    let a = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let b = TracedTensor::from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0]).unwrap();
    let swapped = apply(Arc::new(TestSwap), &[&a, &b]).unwrap();
    let mut iter = swapped.into_iter();
    let _out0 = iter.next().unwrap();
    let out1 = iter.next().unwrap();
    let loss = out1.reduce_sum(Some(&[0])).unwrap();

    let ad = swap_ad_context();
    let grad_a = ad.grad(&loss, &a).expect("grad a");
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ga = grad_a.run_with(&mut engine).unwrap().clone();
    assert_eq!(f64_slice(&ga), &[1.0, 1.0, 1.0]);

    // Even inactive extension paths need the owned extension rule set to
    // linearize the graph before transpose can prove the tangent is absent.
    let maybe = ad.grad_optional(&loss, &b).expect("grad_optional b");
    assert!(
        maybe.is_none(),
        "expected sum(swap(a,b)[1]) to have no gradient wrt b"
    );
}

#[test]
fn extension_carrier_hash_and_eq_are_stable() {
    // Structural identity: two extension carriers with the same payload
    // hash to equal values and compare equal; different payloads (here
    // different families) compare unequal.
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let a: StdTensorOp = StdTensorOp::Extension(Arc::new(TestScaleBy2) as Arc<dyn ExtensionOp>);
    let b: StdTensorOp = StdTensorOp::Extension(Arc::new(TestScaleBy2) as Arc<dyn ExtensionOp>);
    let c: StdTensorOp = StdTensorOp::Extension(Arc::new(TestSwap) as Arc<dyn ExtensionOp>);

    assert_eq!(a, b);
    assert_ne!(a, c);

    let mut ha = DefaultHasher::new();
    a.hash(&mut ha);
    let mut hb = DefaultHasher::new();
    b.hash(&mut hb);
    assert_eq!(ha.finish(), hb.finish());
}

#[test]
fn duplicate_rule_registration_is_rejected() {
    use tenferro_ad::extension::ExtensionRegistryError;

    let mut rules = scale_by_2_rules();
    let err = rules
        .register_linearize(Arc::new(TestScaleBy2Rule))
        .expect_err("second rule registration must error");
    assert!(matches!(
        err,
        ExtensionRegistryError::DuplicateRule {
            family_id: "tenferro-tests.scale_by_2.v1",
            role: ExtensionRuleRole::Linearize
        }
    ));
}

#[test]
fn malformed_family_id_is_rejected() {
    use tenferro_ad::extension::ExtensionRegistryError;

    #[derive(Debug)]
    struct BadRule;
    impl ExtensionLinearizeRule for BadRule {
        fn family_id(&self) -> &'static str {
            "no-version-suffix"
        }
        fn linearize(
            &self,
            _op: &dyn ExtensionOp,
            _builder: &mut dyn PrimitiveRuleBuilder,
            _primal_in: &[ValueKey<StdTensorOp>],
            _primal_out: &[ValueKey<StdTensorOp>],
            _tangent_in: &[Option<LocalValueId>],
            _ctx: &mut ShapeGuardContext,
        ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
            Ok(vec![])
        }
    }

    let err = ExtensionRuleSet::new()
        .with_linearize(Arc::new(BadRule))
        .expect_err("malformed family_id must error");
    assert!(matches!(
        err,
        ExtensionRegistryError::MalformedFamilyId { .. }
    ));
}
