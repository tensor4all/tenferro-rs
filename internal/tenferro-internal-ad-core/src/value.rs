use tenferro_internal_frontend_core::DynTensor;

/// Canonical reverse-mode value handle for tenferro's hard cut to tidu.
pub type DynValue = tidu::Value<DynTensor>;

pub fn new_dyn_value(primal: DynTensor) -> DynValue {
    tidu::Value::new(primal)
}

pub fn new_reverse_leaf(primal: DynTensor) -> DynValue {
    tidu::Value::new(primal).with_requires_grad(true)
}
