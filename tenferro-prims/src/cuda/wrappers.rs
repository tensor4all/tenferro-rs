use std::sync::Arc;

use crate::cuda_ffi::{
    cutensorHandle_t, cutensorOperationDescriptor_t, cutensorPlanPreference_t, cutensorPlan_t,
    cutensorTensorDescriptor_t, CutensorVtable,
};

/// RAII wrapper for `cutensorHandle_t`. Drop calls `cutensorDestroy`.
pub(super) struct HandleWrapper {
    pub(super) raw: cutensorHandle_t,
    pub(super) vtable: Arc<CutensorVtable>,
}

impl Drop for HandleWrapper {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                (self.vtable.destroy)(self.raw);
            }
        }
    }
}

/// RAII wrapper for `cutensorTensorDescriptor_t`.
pub(super) struct TensorDescWrapper {
    pub(super) raw: cutensorTensorDescriptor_t,
    pub(super) vtable: Arc<CutensorVtable>,
}

impl Drop for TensorDescWrapper {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                (self.vtable.destroy_tensor_descriptor)(self.raw);
            }
        }
    }
}

/// RAII wrapper for `cutensorOperationDescriptor_t`.
pub(super) struct OpDescWrapper {
    pub(super) raw: cutensorOperationDescriptor_t,
    pub(super) vtable: Arc<CutensorVtable>,
}

impl Drop for OpDescWrapper {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                (self.vtable.destroy_operation_descriptor)(self.raw);
            }
        }
    }
}

/// RAII wrapper for `cutensorPlanPreference_t`.
pub(super) struct PlanPrefWrapper {
    pub(super) raw: cutensorPlanPreference_t,
    pub(super) vtable: Arc<CutensorVtable>,
}

impl Drop for PlanPrefWrapper {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                (self.vtable.destroy_plan_preference)(self.raw);
            }
        }
    }
}

/// RAII wrapper for `cutensorPlan_t`.
#[derive(Clone, Debug)]
pub(super) struct PlanWrapper {
    pub(super) raw: cutensorPlan_t,
    pub(super) vtable: Arc<CutensorVtable>,
}

impl Drop for PlanWrapper {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                (self.vtable.destroy_plan)(self.raw);
            }
        }
    }
}
