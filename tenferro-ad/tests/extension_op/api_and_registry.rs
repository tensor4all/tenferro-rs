use super::*;

#[test]
fn apply_rejects_wrong_input_count() {
    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _ = apply(Arc::new(TestScaleBy2), &[]);
    }));

    assert!(panic.is_err());
}

#[test]
fn apply_rejects_mismatched_output_metadata_count() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _ = apply(Arc::new(TestBadOutputCount), &[&x]);
    }));

    assert!(panic.is_err());
}

#[test]
fn apply_eager_rejects_empty_input_list() {
    let err = match apply_eager(Arc::new(TestScaleBy2), &[]) {
        Ok(_) => panic!("empty eager extension input list unexpectedly succeeded"),
        Err(err) => err,
    };

    assert!(err.to_string().contains("requires at least one input"));
}

#[test]
fn apply_eager_rejects_wrong_input_count() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![1], vec![1.0_f64]), ctx);

    let err = match apply_eager(Arc::new(TestSwap), &[&x]) {
        Ok(_) => panic!("wrong eager extension input count unexpectedly succeeded"),
        Err(err) => err,
    };

    assert!(err.to_string().contains("expects 2 inputs, got 1"));
}

#[test]
fn apply_eager_rejects_cross_context_inputs() {
    let lhs_ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let rhs_ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let lhs =
        EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![1], vec![1.0_f64]), lhs_ctx);
    let rhs =
        EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![1], vec![2.0_f64]), rhs_ctx);

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
fn apply_eager_rejects_mismatched_output_count() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![1], vec![1.0_f64]), ctx);

    let err = match apply_eager(Arc::new(TestBadOutputCount), &[&x]) {
        Ok(_) => panic!("bad eager extension output count unexpectedly succeeded"),
        Err(err) => err,
    };

    assert!(err.to_string().contains("expected 2 eager outputs"));
}

#[test]
fn scale_by_2_grad_through_symbolic_placeholder() {
    ensure_scale_by_2_registered();

    // Same loss but with an input_symbolic_shape placeholder so the AD
    // path exercises the deferred zero-tangent policy (spec
    // Section 10).
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    let scaled = apply(Arc::new(TestScaleBy2), &[&x])
        .into_iter()
        .next()
        .unwrap();
    let loss = scaled.reduce_sum(&[0]);

    let g = loss.grad(&x).expect("grad build");
    let bound = Tensor::from_vec_col_major(vec![5], vec![10.0_f64, 20.0, 30.0, 40.0, 50.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let grad_out = g
        .run_with_inputs_auto(&mut engine, &[(&x, &bound)])
        .expect("grad eval");

    assert_eq!(grad_out.shape(), &[5]);
    assert_eq!(f64_slice(&grad_out), &[2.0, 2.0, 2.0, 2.0, 2.0]);
}

#[test]
fn swap_forward_roundtrip() {
    ensure_swap_registered();

    let a = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let b = TracedTensor::from_vec_col_major(vec![2], vec![100.0_f64, 200.0]);
    let outputs = apply(Arc::new(TestSwap), &[&a, &b]);
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
    ensure_swap_registered();

    // loss = sum(out0 + out1)   where (out0, out1) = swap(a, b)
    //       = sum(b + a)
    // => dloss/da = ones_like(a),  dloss/db = ones_like(b)
    let a = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]);
    let b = TracedTensor::from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0]);
    let swapped = apply(Arc::new(TestSwap), &[&a, &b]);
    let mut iter = swapped.into_iter();
    let out0 = iter.next().unwrap();
    let out1 = iter.next().unwrap();
    let combined = &out0 + &out1;
    let loss = combined.reduce_sum(&[0]);

    let grad_a = loss.grad(&a).expect("grad a");
    let grad_b = loss.grad(&b).expect("grad b");

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ga = grad_a.run_with(&mut engine).unwrap().clone();
    let gb = grad_b.run_with(&mut engine).unwrap().clone();

    assert_eq!(f64_slice(&ga), &[1.0, 1.0, 1.0]);
    assert_eq!(f64_slice(&gb), &[1.0, 1.0, 1.0]);
}

#[test]
fn swap_grad_routes_only_through_active_output() {
    ensure_swap_registered();

    // loss = sum(out1)  where (out0, out1) = swap(a, b)  (so out1 = a)
    // => dloss/da = ones_like(a)
    let a = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]);
    let b = TracedTensor::from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0]);
    let swapped = apply(Arc::new(TestSwap), &[&a, &b]);
    let mut iter = swapped.into_iter();
    let _out0 = iter.next().unwrap();
    let out1 = iter.next().unwrap();
    let loss = out1.reduce_sum(&[0]);

    let grad_a = loss.grad(&a).expect("grad a");
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ga = grad_a.run_with(&mut engine).unwrap().clone();
    assert_eq!(f64_slice(&ga), &[1.0, 1.0, 1.0]);

    // grad wrt b: sum(out1) does not depend on b; grad_optional returns None.
    let maybe = loss.grad_optional(&b).expect("grad_optional b");
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

    ensure_scale_by_2_registered();
    let err = register_extension_rule(Arc::new(TestScaleBy2Rule))
        .expect_err("second rule registration must error");
    assert!(matches!(
        err,
        ExtensionRegistryError::DuplicateRule {
            family_id: "tenferro-tests.scale_by_2.v1"
        }
    ));
}

#[test]
fn malformed_family_id_is_rejected() {
    use tenferro_ad::extension::ExtensionRegistryError;

    #[derive(Debug)]
    struct BadRule;
    impl ExtensionAdRuleTrait for BadRule {
        fn family_id(&self) -> &'static str {
            "no-version-suffix"
        }
        fn linearize(
            &self,
            _op: &dyn ExtensionOp,
            _builder: &mut dyn OpEmitter<StdTensorOp>,
            _primal_in: &[GlobalValKey<StdTensorOp>],
            _primal_out: &[GlobalValKey<StdTensorOp>],
            _tangent_in: &[Option<LocalValId>],
            _ctx: &mut ShapeGuardContext,
        ) -> ADRuleResult<Vec<Option<LocalValId>>> {
            Ok(vec![])
        }
        fn transpose_rule(
            &self,
            _op: &dyn ExtensionOp,
            _emitter: &mut dyn OpEmitter<StdTensorOp>,
            _cotangent_out: &[Option<LocalValId>],
            _inputs: &[ValRef<StdTensorOp>],
            _mode: &OpMode,
            _ctx: &mut ShapeGuardContext,
        ) -> ADRuleResult<Vec<Option<LocalValId>>> {
            Ok(vec![])
        }
    }

    let err =
        register_extension_rule(Arc::new(BadRule)).expect_err("malformed family_id must error");
    assert!(matches!(
        err,
        ExtensionRegistryError::MalformedFamilyId { .. }
    ));
}
