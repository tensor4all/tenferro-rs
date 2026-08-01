use std::borrow::BorrowMut;
use std::ops::DerefMut;

use tenferro_cpu::CpuBackend;
use tenferro_tensor::backend::BackendSession;

fn replace_by_assignment(session: &mut dyn BackendSession) {
    *session = CpuBackend::new();
}

fn replace_via_as_mut(session: &mut dyn BackendSession) {
    let owner: &mut CpuBackend = AsMut::as_mut(session);
    *owner = CpuBackend::new();
}

fn replace_via_borrow_mut(session: &mut dyn BackendSession) {
    let owner: &mut CpuBackend = BorrowMut::borrow_mut(session);
    *owner = CpuBackend::new();
}

fn replace_via_deref_mut(session: &mut dyn BackendSession) {
    let owner: &mut CpuBackend = DerefMut::deref_mut(session);
    *owner = CpuBackend::new();
}

fn replace_via_backend_mut(session: &mut dyn BackendSession) {
    let owner: &mut CpuBackend = session.backend_mut();
    *owner = CpuBackend::new();
}

fn replace_via_parts_mut(session: &mut dyn BackendSession) {
    let (owner, _) = session.parts_mut();
    let owner: &mut CpuBackend = owner;
    *owner = CpuBackend::new();
}

fn main() {}
