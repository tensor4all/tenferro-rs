use std::any::Any;
use std::hash::Hasher;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tenferro_ad::semantic_extension::{
    AdValue, ResidualSpec, SemanticAdError, SemanticExtensionRuleSet, SemanticPrimalVjpRequest,
    SemanticPrimalVjpRule,
};
use tenferro_ad::AdContext;
use tenferro_runtime::extension::{
    apply, define_extension_runtime, ExtensionAliasDeclaration, ExtensionEffectDeclaration,
    ExtensionExecutionContext, ExtensionOp, ExtensionShapeContext,
};
use tenferro_runtime::{DType, GraphCompiler, Runtime, Tensor, TracedTensor};
use tenferro_tensor::{TensorBackend, TensorRead};

use crate::support::{cpu_runtime, RunTraced};

const ACTION_FAMILY: &str = "tenferro-ad.test.wilson-action.v1";
const FORCE_FAMILY: &str = "tenferro-ad.test.wilson-force.v1";

static FORCE_EXECUTIONS: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Debug)]
struct ActionOp;

#[derive(Clone, Debug)]
struct ForceOp;

macro_rules! impl_extension_op {
    ($type:ty, $family:expr, $inputs:expr, $outputs:expr) => {
        impl ExtensionOp for $type {
            fn family_id(&self) -> &'static str {
                $family
            }

            fn payload_hash(&self, hasher: &mut dyn Hasher) {
                hasher.write_u8(0);
            }

            fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
                other.as_any().is::<Self>()
            }

            fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
                Arc::new(self.clone())
            }

            fn as_any(&self) -> &dyn Any {
                self
            }

            fn input_count(&self) -> usize {
                $inputs
            }

            fn output_count(&self) -> usize {
                $outputs
            }

            fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
                ExtensionEffectDeclaration::Declared(&[])
            }

            fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
                ExtensionAliasDeclaration::AllFresh
            }

            fn infer_output_meta(
                &self,
                context: &mut ExtensionShapeContext<'_>,
            ) -> tenferro_tensor::Result<Vec<(DType, Vec<tenferro_runtime::SymDim>)>> {
                for input in 1..$inputs {
                    context.require_same_shape(0, input)?;
                }
                let metadata = (context.input_dtype(0)?, context.input_shape(0)?.to_vec());
                Ok(vec![metadata; $outputs])
            }
        }
    };
}

impl_extension_op!(ActionOp, ACTION_FAMILY, 4, 1);
impl_extension_op!(ForceOp, FORCE_FAMILY, 5, 4);

fn execute_action<B: TensorBackend + 'static>(
    _op: &ActionOp,
    inputs: &[TensorRead<'_>],
    _context: &mut ExtensionExecutionContext<'_, B>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let values = inputs
        .iter()
        .map(|input| Ok(input.as_slice::<f64>()?[0]))
        .collect::<tenferro_tensor::Result<Vec<_>>>()?;
    Ok(vec![Tensor::from_vec_col_major(
        inputs[0].shape().to_vec(),
        vec![values.into_iter().product::<f64>()],
    )?])
}

fn execute_force<B: TensorBackend + 'static>(
    _op: &ForceOp,
    inputs: &[TensorRead<'_>],
    _context: &mut ExtensionExecutionContext<'_, B>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    FORCE_EXECUTIONS.fetch_add(1, Ordering::SeqCst);
    let values = inputs
        .iter()
        .map(|input| Ok(input.as_slice::<f64>()?[0]))
        .collect::<tenferro_tensor::Result<Vec<_>>>()?;
    let cotangent = values[4];
    let derivatives = [
        values[1] * values[2] * values[3],
        values[0] * values[2] * values[3],
        values[0] * values[1] * values[3],
        values[0] * values[1] * values[2],
    ];
    derivatives
        .into_iter()
        .map(|value| {
            Tensor::from_vec_col_major(inputs[0].shape().to_vec(), vec![value * cotangent])
        })
        .collect()
}

mod action_runtime {
    use super::*;

    define_extension_runtime! {
        runtime = ActionRuntime,
        family_id = ACTION_FAMILY,
        op_type = ActionOp,
        execute_reads = execute_action,
    }
}

mod force_runtime {
    use super::*;

    define_extension_runtime! {
        runtime = ForceRuntime,
        family_id = FORCE_FAMILY,
        op_type = ForceOp,
        execute_reads = execute_force,
    }
}

#[derive(Debug)]
struct ActionRule;

impl SemanticPrimalVjpRule for ActionRule {
    fn family_id(&self) -> &'static str {
        ACTION_FAMILY
    }

    fn residual_mask(&self) -> ResidualSpec {
        ResidualSpec::all_inputs()
    }

    fn primal_vjp(
        &self,
        request: SemanticPrimalVjpRequest<'_>,
        builder: &mut tenferro_runtime::program::SemanticProgramBuilder,
    ) -> Result<Box<[AdValue]>, SemanticAdError> {
        assert_eq!(request.active_inputs(), &[true, true, true, true]);
        let AdValue::Value(cotangent) = request.cotangent_outputs()[0] else {
            return Ok(vec![AdValue::Absent; 4].into_boxed_slice());
        };
        let mut force_inputs = request.primal_inputs().to_vec();
        force_inputs.push(cotangent);
        let force_outputs = builder.add_extension(Arc::new(ForceOp), &force_inputs)?;
        Ok(force_outputs
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                if request.active_inputs()[index] {
                    AdValue::Value(value)
                } else {
                    AdValue::Absent
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice())
    }
}

#[test]
fn four_input_extension_vjp_emits_one_force_node_and_executes_once() {
    FORCE_EXECUTIONS.store(0, Ordering::SeqCst);
    let rules = SemanticExtensionRuleSet::new()
        .with_primal_vjp(Arc::new(ActionRule))
        .unwrap();
    let ad = AdContext::builder()
        .with_semantic_extension_rules(rules)
        .unwrap()
        .build()
        .unwrap();
    let inputs = [2.0_f64, 3.0, 4.0, 5.0]
        .into_iter()
        .map(|value| TracedTensor::from_vec_col_major(vec![], vec![value]).unwrap())
        .collect::<Vec<_>>();
    let input_refs = inputs.iter().collect::<Vec<_>>();
    let action = apply(Arc::new(ActionOp), &input_refs).unwrap().remove(0);
    let cotangent = TracedTensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap();
    let vjp = ad.vjp_many(&action, &input_refs, &cotangent).unwrap();
    assert_eq!(ad.cache_stats().unwrap().ad_transforms.entries, 1);
    let force_outputs = vjp
        .iter()
        .map(|value| value.as_ref().unwrap())
        .collect::<Vec<_>>();
    let first_graph = force_outputs[0].graph();
    assert!(force_outputs
        .iter()
        .all(|value| Arc::ptr_eq(value.graph(), first_graph)));

    let backend = tenferro_cpu::CpuBackend::new();
    let engine_id = tenferro_cpu::runtime_engine_id().unwrap();
    let mut runtime_builder = Runtime::builder();
    runtime_builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&backend).unwrap())
        .unwrap();
    runtime_builder
        .install_extension_module(
            action_runtime::extension_module::<tenferro_cpu::CpuBackend>(engine_id.clone())
                .unwrap(),
        )
        .unwrap();
    runtime_builder
        .install_extension_module(
            force_runtime::extension_module::<tenferro_cpu::CpuBackend>(engine_id).unwrap(),
        )
        .unwrap();
    let runtime = runtime_builder.build().unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(&force_outputs).unwrap();
    let outputs = runtime.run_compiled(&program, &[]).unwrap();

    let expected = [60.0, 40.0, 30.0, 24.0];
    for (output, expected) in outputs.iter().zip(expected) {
        assert_eq!(output.as_slice::<f64>().unwrap(), &[expected]);
    }
    assert_eq!(FORCE_EXECUTIONS.load(Ordering::SeqCst), 1);
}

#[test]
fn many_jvp_and_vjp_cover_two_inputs_in_one_request() {
    let ad = AdContext::builder().build().unwrap();
    let x = TracedTensor::from_vec_col_major(vec![], vec![2.0_f64]).unwrap();
    let y = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    let dx = TracedTensor::from_vec_col_major(vec![], vec![5.0_f64]).unwrap();
    let dy = TracedTensor::from_vec_col_major(vec![], vec![7.0_f64]).unwrap();
    let output = (&x + &y).unwrap();
    let cotangent = TracedTensor::from_vec_col_major(vec![], vec![11.0_f64]).unwrap();

    let jvp = ad
        .jvp_many(&output, &[(&x, &dx), (&y, &dy)])
        .unwrap()
        .unwrap();
    let vjp = ad.vjp_many(&output, &[&y, &x], &cotangent).unwrap();
    let runtime = cpu_runtime();

    assert_eq!(
        jvp.run_with(&runtime).unwrap().as_slice::<f64>().unwrap(),
        &[12.0]
    );
    assert_eq!(vjp.len(), 2);
    assert_eq!(
        vjp[0]
            .as_ref()
            .unwrap()
            .run_with(&runtime)
            .unwrap()
            .as_slice::<f64>()
            .unwrap(),
        &[11.0]
    );
    assert_eq!(
        vjp[1]
            .as_ref()
            .unwrap()
            .run_with(&runtime)
            .unwrap()
            .as_slice::<f64>()
            .unwrap(),
        &[11.0]
    );
}

#[test]
fn four_input_requests_use_one_transform_and_one_shared_derivative_graph() {
    let ad = AdContext::builder().build().unwrap();
    let inputs = [2.0_f64, 3.0, 4.0, 5.0]
        .into_iter()
        .map(|value| TracedTensor::from_vec_col_major(vec![], vec![value]).unwrap())
        .collect::<Vec<_>>();
    let tangents = [1.0_f64, 2.0, 3.0, 4.0]
        .into_iter()
        .map(|value| TracedTensor::from_vec_col_major(vec![], vec![value]).unwrap())
        .collect::<Vec<_>>();
    let output =
        (&(&inputs[0] * &inputs[1]).unwrap() + &(&inputs[2] * &inputs[3]).unwrap()).unwrap();
    let cotangent = TracedTensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap();

    let jvp = ad
        .jvp_many(
            &output,
            &[
                (&inputs[0], &tangents[0]),
                (&inputs[1], &tangents[1]),
                (&inputs[2], &tangents[2]),
                (&inputs[3], &tangents[3]),
            ],
        )
        .unwrap()
        .unwrap();
    assert_eq!(ad.cache_stats().unwrap().ad_transforms.entries, 1);

    let wrts = inputs.iter().collect::<Vec<_>>();
    let vjp = ad.vjp_many(&output, &wrts, &cotangent).unwrap();
    assert_eq!(ad.cache_stats().unwrap().ad_transforms.entries, 2);
    let first_graph = vjp[0].as_ref().unwrap().graph();
    assert!(vjp
        .iter()
        .flatten()
        .all(|value| std::sync::Arc::ptr_eq(value.graph(), first_graph)));

    let runtime = cpu_runtime();
    assert_eq!(
        jvp.run_with(&runtime).unwrap().as_slice::<f64>().unwrap(),
        &[38.0]
    );
    let expected = [3.0, 2.0, 5.0, 4.0];
    for (value, expected) in vjp.iter().zip(expected) {
        assert_eq!(
            value
                .as_ref()
                .unwrap()
                .run_with(&runtime)
                .unwrap()
                .as_slice::<f64>()
                .unwrap(),
            &[expected]
        );
    }
}

#[test]
fn many_requests_preserve_unreachable_and_duplicate_policies() {
    let ad = AdContext::builder().build().unwrap();
    let x = TracedTensor::from_vec_col_major(vec![], vec![2.0_f64]).unwrap();
    let y = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    let tangent = TracedTensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap();
    let output = (&x * &x).unwrap();
    let cotangent = TracedTensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap();

    let vjp = ad.vjp_many(&output, &[&y, &x, &x], &cotangent).unwrap();
    assert!(vjp[0].is_none());
    assert!(vjp[1].is_some());
    assert!(vjp[2].is_some());
    assert!(ad
        .jvp_many(&output, &[(&y, &tangent), (&x, &tangent)])
        .is_ok());
    let duplicate = ad.jvp_many(&output, &[(&x, &tangent), (&x, &tangent)]);
    assert!(duplicate.is_err());
}

#[test]
fn many_requests_cover_empty_and_seed_metadata_errors() {
    let ad = AdContext::builder().build().unwrap();
    let x = TracedTensor::from_vec_col_major(vec![], vec![2.0_f64]).unwrap();
    let output = (&x * &x).unwrap();
    let seed = TracedTensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap();

    assert!(ad.jvp_many(&output, &[]).unwrap().is_none());
    assert!(ad.vjp_many(&output, &[], &seed).unwrap().is_empty());
    assert_eq!(ad.cache_stats().unwrap().ad_transforms.entries, 0);

    let wrong_shape = TracedTensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    assert!(ad.jvp_many(&output, &[(&x, &wrong_shape)]).is_err());
    assert!(ad.vjp_many(&output, &[&x], &wrong_shape).is_err());
}
