use super::*;

pub(super) type PullbackRule<T> =
    Box<dyn Fn(&Tensor<T>) -> Result<Vec<(NodeId, Tensor<T>)>> + 'static>;
pub(super) type BridgeRule<TOut, TIn> =
    Box<dyn Fn(&Tensor<TOut>) -> Result<Vec<(NodeId, Tensor<TIn>)>> + 'static>;
pub(super) type ScalarBridgeRule<TOut, TIn> =
    Box<dyn Fn(&Tensor<TOut>) -> Result<Vec<(NodeId, TIn)>> + 'static>;
pub(super) type ScalarMixedRule<TOut, TIn> =
    Box<dyn Fn(&TOut) -> Result<Vec<(NodeId, TIn)>> + 'static>;
pub(super) type ScalarPullbackRule<T> = Box<dyn Fn(&T) -> Result<Vec<(NodeId, T)>> + 'static>;

pub(super) struct TapeRules<T: Scalar> {
    pub(super) rules: HashMap<NodeId, PullbackRule<T>>,
}

impl<T: Scalar> TapeRules<T> {
    fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }
}

struct TapeBridgeRules<TOut: Scalar, TIn: Scalar> {
    rules: HashMap<NodeId, BridgeRule<TOut, TIn>>,
}

impl<TOut: Scalar, TIn: Scalar> TapeBridgeRules<TOut, TIn> {
    fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }
}

struct TapeScalarBridgeRules<TOut: Scalar, TIn: ScalarAd> {
    rules: HashMap<NodeId, ScalarBridgeRule<TOut, TIn>>,
}

impl<TOut: Scalar, TIn: ScalarAd> TapeScalarBridgeRules<TOut, TIn> {
    fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }
}

struct TapeScalarMixedRules<TOut: ScalarAd, TIn: ScalarAd> {
    rules: HashMap<NodeId, ScalarMixedRule<TOut, TIn>>,
}

impl<TOut: ScalarAd, TIn: ScalarAd> TapeScalarMixedRules<TOut, TIn> {
    fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }
}

pub(super) struct TapeScalarRules<T: ScalarAd> {
    pub(super) rules: HashMap<NodeId, ScalarPullbackRule<T>>,
}

impl<T: ScalarAd> TapeScalarRules<T> {
    fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }
}

type RuleRegistry = HashMap<(u64, TypeId), Box<dyn Any>>;
type BridgeRegistry = HashMap<(u64, TypeId, TypeId), Box<dyn Any>>;
type ScalarBridgeRegistry = HashMap<(u64, TypeId, TypeId), Box<dyn Any>>;
type ScalarMixedRegistry = HashMap<(u64, TypeId, TypeId), Box<dyn Any>>;
type ScalarRuleRegistry = HashMap<(u64, TypeId), Box<dyn Any>>;

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
    static REVERSE_RULE_REGISTRY: RefCell<RuleRegistry> = RefCell::new(HashMap::new());
    static REVERSE_BRIDGE_REGISTRY: RefCell<BridgeRegistry> = RefCell::new(HashMap::new());
    static REVERSE_SCALAR_BRIDGE_REGISTRY: RefCell<ScalarBridgeRegistry> = RefCell::new(HashMap::new());
    static REVERSE_SCALAR_MIXED_REGISTRY: RefCell<ScalarMixedRegistry> = RefCell::new(HashMap::new());
    static REVERSE_SCALAR_RULE_REGISTRY: RefCell<ScalarRuleRegistry> = RefCell::new(HashMap::new());
}

pub(crate) fn register_rule<T: Scalar + 'static>(
    tape: TapeId,
    node: NodeId,
    rule: PullbackRule<T>,
) -> Result<()> {
    REVERSE_RULE_REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let key = (tape.0, TypeId::of::<T>());
        let entry = registry
            .entry(key)
            .or_insert_with(|| Box::new(TapeRules::<T>::new()));
        let typed = entry
            .downcast_mut::<TapeRules<T>>()
            .ok_or_else(|| Error::InvalidAdTensor {
                message: "reverse tape registry type mismatch".to_string(),
            })?;
        typed.rules.insert(node, rule);
        Ok(())
    })
}

pub(crate) fn register_bridge_rule<TOut: Scalar + 'static, TIn: Scalar + 'static>(
    tape: TapeId,
    node: NodeId,
    rule: BridgeRule<TOut, TIn>,
) -> Result<()> {
    REVERSE_BRIDGE_REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let key = (tape.0, TypeId::of::<TOut>(), TypeId::of::<TIn>());
        let entry = registry
            .entry(key)
            .or_insert_with(|| Box::new(TapeBridgeRules::<TOut, TIn>::new()));
        let typed = entry
            .downcast_mut::<TapeBridgeRules<TOut, TIn>>()
            .ok_or_else(|| Error::InvalidAdTensor {
                message: "reverse tape bridge registry type mismatch".to_string(),
            })?;
        typed.rules.insert(node, rule);
        Ok(())
    })
}

pub(crate) fn register_scalar_bridge_rule<TOut: Scalar + 'static, TIn: ScalarAd + 'static>(
    tape: TapeId,
    node: NodeId,
    rule: ScalarBridgeRule<TOut, TIn>,
) -> Result<()> {
    REVERSE_SCALAR_BRIDGE_REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let key = (tape.0, TypeId::of::<TOut>(), TypeId::of::<TIn>());
        let entry = registry
            .entry(key)
            .or_insert_with(|| Box::new(TapeScalarBridgeRules::<TOut, TIn>::new()));
        let typed = entry
            .downcast_mut::<TapeScalarBridgeRules<TOut, TIn>>()
            .ok_or_else(|| Error::InvalidAdScalar {
                message: "reverse scalar bridge registry type mismatch".to_string(),
            })?;
        typed.rules.insert(node, rule);
        Ok(())
    })
}

pub(crate) fn register_scalar_mixed_rule<TOut: ScalarAd + 'static, TIn: ScalarAd + 'static>(
    tape: TapeId,
    node: NodeId,
    rule: ScalarMixedRule<TOut, TIn>,
) -> Result<()> {
    REVERSE_SCALAR_MIXED_REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let key = (tape.0, TypeId::of::<TOut>(), TypeId::of::<TIn>());
        let entry = registry
            .entry(key)
            .or_insert_with(|| Box::new(TapeScalarMixedRules::<TOut, TIn>::new()));
        let typed = entry
            .downcast_mut::<TapeScalarMixedRules<TOut, TIn>>()
            .ok_or_else(|| Error::InvalidAdScalar {
                message: "reverse scalar mixed registry type mismatch".to_string(),
            })?;
        typed.rules.insert(node, rule);
        Ok(())
    })
}

pub(crate) fn register_scalar_rule<T: ScalarAd + 'static>(
    tape: TapeId,
    node: NodeId,
    rule: ScalarPullbackRule<T>,
) -> Result<()> {
    REVERSE_SCALAR_RULE_REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let key = (tape.0, TypeId::of::<T>());
        let entry = registry
            .entry(key)
            .or_insert_with(|| Box::new(TapeScalarRules::<T>::new()));
        let typed =
            entry
                .downcast_mut::<TapeScalarRules<T>>()
                .ok_or_else(|| Error::InvalidAdScalar {
                    message: "reverse scalar tape registry type mismatch".to_string(),
                })?;
        typed.rules.insert(node, rule);
        Ok(())
    })
}

pub(super) fn bridge_pullback<TOut: Scalar + 'static, TIn: Scalar + 'static>(
    tape: TapeId,
    output_node: NodeId,
    cotangent: &Tensor<TOut>,
) -> Result<Vec<(NodeId, Tensor<TIn>)>> {
    REVERSE_BRIDGE_REGISTRY.with(|registry| {
        let registry = registry.borrow();
        let key = (tape.0, TypeId::of::<TOut>(), TypeId::of::<TIn>());
        let Some(state_any) = registry.get(&key) else {
            return Ok(Vec::new());
        };
        let state = state_any
            .downcast_ref::<TapeBridgeRules<TOut, TIn>>()
            .ok_or_else(|| Error::InvalidAdTensor {
                message: "reverse tape bridge registry type mismatch".to_string(),
            })?;
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
    REVERSE_SCALAR_BRIDGE_REGISTRY.with(|registry| {
        let registry = registry.borrow();
        let key = (tape.0, TypeId::of::<TOut>(), TypeId::of::<TIn>());
        let Some(state_any) = registry.get(&key) else {
            return Ok(Vec::new());
        };
        let state = state_any
            .downcast_ref::<TapeScalarBridgeRules<TOut, TIn>>()
            .ok_or_else(|| Error::InvalidAdScalar {
                message: "reverse scalar bridge registry type mismatch".to_string(),
            })?;
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
    REVERSE_SCALAR_MIXED_REGISTRY.with(|registry| {
        let registry = registry.borrow();
        let key = (tape.0, TypeId::of::<TOut>(), TypeId::of::<TIn>());
        let Some(state_any) = registry.get(&key) else {
            return Ok(Vec::new());
        };
        let state = state_any
            .downcast_ref::<TapeScalarMixedRules<TOut, TIn>>()
            .ok_or_else(|| Error::InvalidAdScalar {
                message: "reverse scalar mixed registry type mismatch".to_string(),
            })?;
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
    REVERSE_RULE_REGISTRY.with(|registry| {
        let registry = registry.borrow();
        let key = (tape.0, TypeId::of::<T>());
        let state_any = registry.get(&key).ok_or_else(|| Error::InvalidAdTensor {
            message: format!("no reverse rules registered for tape {}", tape.0),
        })?;
        let state =
            state_any
                .downcast_ref::<TapeRules<T>>()
                .ok_or_else(|| Error::InvalidAdTensor {
                    message: "reverse tape registry type mismatch".to_string(),
                })?;
        f(state)
    })
}

pub(super) fn with_scalar_rules<T: ScalarAd + 'static, R>(
    tape: TapeId,
    f: impl FnOnce(&TapeScalarRules<T>) -> Result<R>,
) -> Result<R> {
    REVERSE_SCALAR_RULE_REGISTRY.with(|registry| {
        let registry = registry.borrow();
        let key = (tape.0, TypeId::of::<T>());
        let state_any = registry.get(&key).ok_or_else(|| Error::InvalidAdScalar {
            message: format!("no reverse scalar rules registered for tape {}", tape.0),
        })?;
        let state = state_any
            .downcast_ref::<TapeScalarRules<T>>()
            .ok_or_else(|| Error::InvalidAdScalar {
                message: "reverse scalar tape registry type mismatch".to_string(),
            })?;
        f(state)
    })
}
