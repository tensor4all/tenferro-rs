use std::borrow::BorrowMut;
use std::ops::DerefMut;

use tenferro_tensor::backend::Engine;

fn replace_by_assignment(session: &mut dyn Engine) {
    *session = tenferro_cpu::CpuEngine::serial();
}

fn replace_via_as_mut(session: &mut dyn Engine) {
    let owner: &mut tenferro_cpu::CpuEngine = AsMut::as_mut(session);
    *owner = tenferro_cpu::CpuEngine::serial();
}

fn replace_via_borrow_mut(session: &mut dyn Engine) {
    let owner: &mut tenferro_cpu::CpuEngine = BorrowMut::borrow_mut(session);
    *owner = tenferro_cpu::CpuEngine::serial();
}

fn replace_via_deref_mut(session: &mut dyn Engine) {
    let owner: &mut tenferro_cpu::CpuEngine = DerefMut::deref_mut(session);
    *owner = tenferro_cpu::CpuEngine::serial();
}

fn replace_via_backend_mut(session: &mut dyn Engine) {
    let owner: &mut tenferro_cpu::CpuEngine = session.backend_mut();
    *owner = tenferro_cpu::CpuEngine::serial();
}

fn replace_via_parts_mut(session: &mut dyn Engine) {
    let (owner, _) = <dyn Engine as Engine>::parts_mut(session);
    let owner: &mut tenferro_cpu::CpuEngine = owner;
    *owner = tenferro_cpu::CpuEngine::serial();
}

fn main() {}
