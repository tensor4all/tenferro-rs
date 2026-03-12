use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::Hash;

use chainrules_scalarops::ScalarAd;
use tenferro_algebra::Scalar;
use tenferro_tensor::Tensor;

use crate::{Error, NodeId, Result, TapeId};

pub(super) type PullbackRule<T> =
    Box<dyn Fn(&Tensor<T>) -> Result<Vec<(NodeId, Tensor<T>)>> + 'static>;
pub(super) type BridgeRule<TOut, TIn> =
    Box<dyn Fn(&Tensor<TOut>) -> Result<Vec<(NodeId, Tensor<TIn>)>> + 'static>;
pub(super) type ScalarBridgeRule<TOut, TIn> =
    Box<dyn Fn(&Tensor<TOut>) -> Result<Vec<(NodeId, TIn)>> + 'static>;
pub(super) type ScalarMixedRule<TOut, TIn> =
    Box<dyn Fn(&TOut) -> Result<Vec<(NodeId, TIn)>> + 'static>;
pub(super) type ScalarPullbackRule<T> = Box<dyn Fn(&T) -> Result<Vec<(NodeId, T)>> + 'static>;

pub(super) struct NodeRuleStore<R> {
    pub(super) rules: HashMap<NodeId, R>,
}

impl<R> NodeRuleStore<R> {
    fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }
}

pub(super) type TapeRules<T> = NodeRuleStore<PullbackRule<T>>;
type TapeBridgeRules<TOut, TIn> = NodeRuleStore<BridgeRule<TOut, TIn>>;
type TapeScalarBridgeRules<TOut, TIn> = NodeRuleStore<ScalarBridgeRule<TOut, TIn>>;
type TapeScalarMixedRules<TOut, TIn> = NodeRuleStore<ScalarMixedRule<TOut, TIn>>;
pub(super) type TapeScalarRules<T> = NodeRuleStore<ScalarPullbackRule<T>>;

type UnaryTypeKey = TypeId;
type BinaryTypeKey = (TypeId, TypeId);

#[inline]
fn unary_type_key<T: 'static>() -> UnaryTypeKey {
    TypeId::of::<T>()
}

#[inline]
fn binary_type_key<TOut: 'static, TIn: 'static>() -> BinaryTypeKey {
    (TypeId::of::<TOut>(), TypeId::of::<TIn>())
}

struct TapeRuleStore {
    tensor_rules: HashMap<UnaryTypeKey, Box<dyn Any>>,
    bridge_rules: HashMap<BinaryTypeKey, Box<dyn Any>>,
    scalar_bridge_rules: HashMap<BinaryTypeKey, Box<dyn Any>>,
    scalar_mixed_rules: HashMap<BinaryTypeKey, Box<dyn Any>>,
    scalar_rules: HashMap<UnaryTypeKey, Box<dyn Any>>,
}

impl TapeRuleStore {
    fn new() -> Self {
        Self {
            tensor_rules: HashMap::new(),
            bridge_rules: HashMap::new(),
            scalar_bridge_rules: HashMap::new(),
            scalar_mixed_rules: HashMap::new(),
            scalar_rules: HashMap::new(),
        }
    }

    fn tensor_rules_mut<T: Scalar + 'static>(&mut self) -> &mut TapeRules<T> {
        typed_bucket_mut(
            &mut self.tensor_rules,
            unary_type_key::<T>(),
            TapeRules::<T>::new,
        )
    }

    fn scalar_rules_mut<T: ScalarAd + 'static>(&mut self) -> &mut TapeScalarRules<T> {
        typed_bucket_mut(
            &mut self.scalar_rules,
            unary_type_key::<T>(),
            TapeScalarRules::<T>::new,
        )
    }

    fn bridge_rules_mut<TOut: Scalar + 'static, TIn: Scalar + 'static>(
        &mut self,
    ) -> &mut TapeBridgeRules<TOut, TIn> {
        typed_bucket_mut(
            &mut self.bridge_rules,
            binary_type_key::<TOut, TIn>(),
            TapeBridgeRules::<TOut, TIn>::new,
        )
    }

    fn scalar_bridge_rules_mut<TOut: Scalar + 'static, TIn: ScalarAd + 'static>(
        &mut self,
    ) -> &mut TapeScalarBridgeRules<TOut, TIn> {
        typed_bucket_mut(
            &mut self.scalar_bridge_rules,
            binary_type_key::<TOut, TIn>(),
            TapeScalarBridgeRules::<TOut, TIn>::new,
        )
    }

    fn scalar_mixed_rules_mut<TOut: ScalarAd + 'static, TIn: ScalarAd + 'static>(
        &mut self,
    ) -> &mut TapeScalarMixedRules<TOut, TIn> {
        typed_bucket_mut(
            &mut self.scalar_mixed_rules,
            binary_type_key::<TOut, TIn>(),
            TapeScalarMixedRules::<TOut, TIn>::new,
        )
    }

    fn tensor_rules<T: Scalar + 'static>(&self, tape: TapeId) -> Result<&TapeRules<T>> {
        typed_bucket_ref(
            &self.tensor_rules,
            &unary_type_key::<T>(),
            || Error::InvalidAdTensor {
                message: format!("no reverse rules registered for tape {}", tape.0),
            },
            || Error::InvalidAdTensor {
                message: "reverse tape registry type mismatch".to_string(),
            },
        )
    }

    fn scalar_rules<T: ScalarAd + 'static>(&self, tape: TapeId) -> Result<&TapeScalarRules<T>> {
        typed_bucket_ref(
            &self.scalar_rules,
            &unary_type_key::<T>(),
            || Error::InvalidAdScalar {
                message: format!("no reverse scalar rules registered for tape {}", tape.0),
            },
            || Error::InvalidAdScalar {
                message: "reverse scalar tape registry type mismatch".to_string(),
            },
        )
    }

    fn bridge_rules<TOut: Scalar + 'static, TIn: Scalar + 'static>(
        &self,
    ) -> Result<Option<&TapeBridgeRules<TOut, TIn>>> {
        typed_bucket_opt_ref(&self.bridge_rules, &binary_type_key::<TOut, TIn>(), || {
            Error::InvalidAdTensor {
                message: "reverse tape bridge registry type mismatch".to_string(),
            }
        })
    }

    fn scalar_bridge_rules<TOut: Scalar + 'static, TIn: ScalarAd + 'static>(
        &self,
    ) -> Result<Option<&TapeScalarBridgeRules<TOut, TIn>>> {
        typed_bucket_opt_ref(
            &self.scalar_bridge_rules,
            &binary_type_key::<TOut, TIn>(),
            || Error::InvalidAdScalar {
                message: "reverse scalar bridge registry type mismatch".to_string(),
            },
        )
    }

    fn scalar_mixed_rules<TOut: ScalarAd + 'static, TIn: ScalarAd + 'static>(
        &self,
    ) -> Result<Option<&TapeScalarMixedRules<TOut, TIn>>> {
        typed_bucket_opt_ref(
            &self.scalar_mixed_rules,
            &binary_type_key::<TOut, TIn>(),
            || Error::InvalidAdScalar {
                message: "reverse scalar mixed registry type mismatch".to_string(),
            },
        )
    }
}

fn typed_bucket_mut<S: 'static, K: Eq + Hash>(
    bucket: &mut HashMap<K, Box<dyn Any>>,
    key: K,
    init: impl FnOnce() -> S,
) -> &mut S {
    let entry = bucket.entry(key).or_insert_with(|| Box::new(init()));
    match entry.downcast_mut::<S>() {
        Some(state) => state,
        None => unreachable!("reverse tape registry invariant violated"),
    }
}

fn typed_bucket_ref<'a, S: 'static, K: Eq + Hash>(
    bucket: &'a HashMap<K, Box<dyn Any>>,
    key: &K,
    missing: impl FnOnce() -> Error,
    mismatch: impl FnOnce() -> Error,
) -> Result<&'a S> {
    let state_any = bucket.get(key).ok_or_else(missing)?;
    state_any.downcast_ref::<S>().ok_or_else(mismatch)
}

fn typed_bucket_opt_ref<'a, S: 'static, K: Eq + Hash>(
    bucket: &'a HashMap<K, Box<dyn Any>>,
    key: &K,
    mismatch: impl FnOnce() -> Error,
) -> Result<Option<&'a S>> {
    let Some(state_any) = bucket.get(key) else {
        return Ok(None);
    };
    state_any.downcast_ref::<S>().ok_or_else(mismatch).map(Some)
}

fn with_tape_store_mut<R>(tape: TapeId, f: impl FnOnce(&mut TapeRuleStore) -> R) -> R {
    TAPE_RULE_STORES.with(|stores| {
        let mut stores = stores.borrow_mut();
        let store = stores.entry(tape.0).or_insert_with(TapeRuleStore::new);
        f(store)
    })
}

fn with_tape_store_ref<R>(
    tape: TapeId,
    f: impl FnOnce(Option<&TapeRuleStore>) -> Result<R>,
) -> Result<R> {
    TAPE_RULE_STORES.with(|stores| {
        let stores = stores.borrow();
        f(stores.get(&tape.0))
    })
}

pub(super) fn is_no_tensor_rules_error(err: &Error) -> bool {
    matches!(
        err,
        Error::InvalidAdTensor { message }
            if message.starts_with("no reverse rules registered for tape")
                || message.starts_with("no reverse rule registered for output node")
    )
}

pub(super) fn is_no_scalar_rules_error(err: &Error) -> bool {
    matches!(
        err,
        Error::InvalidAdScalar { message }
            if message.starts_with("no reverse scalar rules registered for tape")
                || message.starts_with("no reverse scalar rule registered for output node")
    )
}

thread_local! {
    static TAPE_RULE_STORES: RefCell<HashMap<u64, TapeRuleStore>> = RefCell::new(HashMap::new());
}

pub(crate) fn register_rule<T: Scalar + 'static>(
    tape: TapeId,
    node: NodeId,
    rule: PullbackRule<T>,
) {
    with_tape_store_mut(tape, |store| {
        store.tensor_rules_mut::<T>().rules.insert(node, rule);
    })
}

pub(crate) fn register_bridge_rule<TOut: Scalar + 'static, TIn: Scalar + 'static>(
    tape: TapeId,
    node: NodeId,
    rule: BridgeRule<TOut, TIn>,
) {
    with_tape_store_mut(tape, |store| {
        store
            .bridge_rules_mut::<TOut, TIn>()
            .rules
            .insert(node, rule);
    })
}

pub(crate) fn register_scalar_bridge_rule<TOut: Scalar + 'static, TIn: ScalarAd + 'static>(
    tape: TapeId,
    node: NodeId,
    rule: ScalarBridgeRule<TOut, TIn>,
) {
    with_tape_store_mut(tape, |store| {
        store
            .scalar_bridge_rules_mut::<TOut, TIn>()
            .rules
            .insert(node, rule);
    })
}

pub(crate) fn register_scalar_mixed_rule<TOut: ScalarAd + 'static, TIn: ScalarAd + 'static>(
    tape: TapeId,
    node: NodeId,
    rule: ScalarMixedRule<TOut, TIn>,
) {
    with_tape_store_mut(tape, |store| {
        store
            .scalar_mixed_rules_mut::<TOut, TIn>()
            .rules
            .insert(node, rule);
    })
}

pub(crate) fn register_scalar_rule<T: ScalarAd + 'static>(
    tape: TapeId,
    node: NodeId,
    rule: ScalarPullbackRule<T>,
) {
    with_tape_store_mut(tape, |store| {
        store.scalar_rules_mut::<T>().rules.insert(node, rule);
    })
}

pub(super) fn bridge_pullback<TOut: Scalar + 'static, TIn: Scalar + 'static>(
    tape: TapeId,
    output_node: NodeId,
    cotangent: &Tensor<TOut>,
) -> Result<Vec<(NodeId, Tensor<TIn>)>> {
    with_tape_store_ref(tape, |store| {
        let Some(store) = store else {
            return Ok(Vec::new());
        };
        let Some(state) = store.bridge_rules::<TOut, TIn>()? else {
            return Ok(Vec::new());
        };
        let Some(rule) = state.rules.get(&output_node) else {
            return Ok(Vec::new());
        };
        rule(cotangent)
    })
}

pub(super) fn bridge_pullback_scalar<TOut: Scalar + 'static, TIn: ScalarAd + 'static>(
    tape: TapeId,
    output_node: NodeId,
    cotangent: &Tensor<TOut>,
) -> Result<Vec<(NodeId, TIn)>> {
    with_tape_store_ref(tape, |store| {
        let Some(store) = store else {
            return Ok(Vec::new());
        };
        let Some(state) = store.scalar_bridge_rules::<TOut, TIn>()? else {
            return Ok(Vec::new());
        };
        let Some(rule) = state.rules.get(&output_node) else {
            return Ok(Vec::new());
        };
        rule(cotangent)
    })
}

pub(super) fn bridge_pullback_scalar_mixed<TOut: ScalarAd + 'static, TIn: ScalarAd + 'static>(
    tape: TapeId,
    output_node: NodeId,
    cotangent: &TOut,
) -> Result<Vec<(NodeId, TIn)>> {
    with_tape_store_ref(tape, |store| {
        let Some(store) = store else {
            return Ok(Vec::new());
        };
        let Some(state) = store.scalar_mixed_rules::<TOut, TIn>()? else {
            return Ok(Vec::new());
        };
        let Some(rule) = state.rules.get(&output_node) else {
            return Ok(Vec::new());
        };
        rule(cotangent)
    })
}

pub(super) fn with_tensor_rules<T: Scalar + 'static, R>(
    tape: TapeId,
    f: impl FnOnce(&TapeRules<T>) -> Result<R>,
) -> Result<R> {
    with_tape_store_ref(tape, |store| {
        let store = store.ok_or_else(|| Error::InvalidAdTensor {
            message: format!("no reverse rules registered for tape {}", tape.0),
        })?;
        f(store.tensor_rules::<T>(tape)?)
    })
}

pub(super) fn with_scalar_rules<T: ScalarAd + 'static, R>(
    tape: TapeId,
    f: impl FnOnce(&TapeScalarRules<T>) -> Result<R>,
) -> Result<R> {
    with_tape_store_ref(tape, |store| {
        let store = store.ok_or_else(|| Error::InvalidAdScalar {
            message: format!("no reverse scalar rules registered for tape {}", tape.0),
        })?;
        f(store.scalar_rules::<T>(tape)?)
    })
}
