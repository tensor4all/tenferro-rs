use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

use tenferro_tensor::{BackendSessionHost, ErrorKind, SessionEntryError, TensorRead};

use super::*;

fn backend() -> CpuBackend {
    CpuBackend::with_threads(1).expect("one-worker CPU backend")
}

fn vector_a() -> tenferro_tensor::Tensor {
    tenferro_tensor::Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).expect("vector")
}

fn vector_b() -> tenferro_tensor::Tensor {
    tenferro_tensor::Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).expect("vector")
}

fn add_in_session(session: &mut CpuHeldSession<'_>) -> Vec<f64> {
    let (a, b) = (vector_a(), vector_b());
    let value = session
        .with_concrete_session(None, |view| {
            view.add_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b))
        })
        .expect("held concrete operation");
    value.as_slice::<f64>().expect("f64 payload").to_vec()
}

/// The held session must match the scoped entry on the same inputs.
#[test]
fn held_session_matches_the_scoped_entry() {
    let mut backend = backend();
    let mut session = backend.open_session().expect("held session");
    assert_eq!(add_in_session(&mut session), vec![3.0]);
    session.close().expect("affinity restores");

    let (a, b) = (vector_a(), vector_b());
    let scoped = backend
        .with_backend_session(|view| {
            view.add_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b))
        })
        .expect("scoped entry")
        .expect("scoped operation");
    assert_eq!(scoped.as_slice::<f64>().expect("f64 payload"), &[3.0]);
}

/// Nested root entry and any scoped entry inside a held session are rejected
/// typed before they can block, and the session stays usable afterwards.
#[test]
fn nested_entry_is_rejected_and_the_session_survives() {
    let backend = backend();
    let mut other = backend.clone();
    let mut session = backend.open_session().expect("held session");

    assert!(matches!(
        backend.open_session(),
        Err(SessionEntryError::Reentered { .. })
    ));
    assert!(matches!(
        other.with_backend_session(|view| view.reclaim_buffer(vector_a())),
        Err(SessionEntryError::Reentered { .. })
    ));

    assert_eq!(add_in_session(&mut session), vec![3.0]);
    session.close().expect("affinity restores");
}

/// A held root cannot escape an enclosing shared execution scope, whose
/// operation loan also protects the eager owner lock.
#[test]
fn held_root_is_rejected_inside_a_shared_execution_scope() {
    let backend = backend();
    let mut operations = backend.clone();
    let rejected = backend
        .with_execution_scope(|| backend.open_session().is_err())
        .expect("scope entry");
    assert!(rejected, "a held root must not open inside a shared scope");

    // The scope keeps working exactly as before after the rejection.
    let (a, b) = (vector_a(), vector_b());
    let value = backend
        .with_execution_scope(|| {
            operations.with_backend_session(|view| {
                view.add_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b))
            })
        })
        .expect("scope entry")
        .expect("scoped entry")
        .expect("scoped operation");
    assert_eq!(value.as_slice::<f64>().expect("f64 payload"), &[3.0]);
}

/// A child execution handle keeps using the scoped entry with its inherited
/// owner; it does not open a held root of its own.
#[test]
fn child_handles_do_not_open_held_roots() {
    let mut backend = backend();
    backend
        .with_backend_session(|view| {
            let child_backend =
                crate::with_cpu_exec_session(view, |session| session.child_execution().backend())
                    .expect("CPU native session");
            assert!(matches!(
                child_backend.open_session(),
                Err(SessionEntryError::Reentered { .. })
            ));
        })
        .expect("scoped entry");
}

/// Management entry points report the checked-out resources instead of waiting
/// on a lock the session owns, and work again once the session closes.
#[test]
fn management_reports_checked_out_resources_while_held() {
    let backend = backend();
    let session = backend.open_session().expect("held session");
    assert_eq!(
        backend.buffer_pool_len().unwrap_err().kind(),
        ErrorKind::RuntimeState
    );
    assert_eq!(
        backend.buffer_pool_stats().unwrap_err().kind(),
        ErrorKind::RuntimeState
    );
    assert_eq!(
        backend.indexed_plan_cache_stats().unwrap_err().kind(),
        ErrorKind::RuntimeState
    );
    session.close().expect("affinity restores");

    assert_eq!(backend.buffer_pool_len().expect("resources returned"), 0);
    let mut reopened = backend.open_session().expect("held session reopens");
    assert_eq!(add_in_session(&mut reopened), vec![3.0]);
    reopened.close().expect("affinity restores");
}

/// Dropping a session releases the same state as closing it.
#[test]
fn drop_releases_the_session_like_close() {
    let backend = backend();
    {
        let mut session = backend.open_session().expect("held session");
        assert_eq!(add_in_session(&mut session), vec![3.0]);
    }
    assert_eq!(backend.buffer_pool_len().expect("resources returned"), 0);
    let _second = backend.open_session().expect("held session reopens");
}

/// Closing returns the engine resources before it releases the admission
/// reservation, so an entry queued behind the session does not observe a
/// vacant slot.
#[test]
fn a_queued_entry_succeeds_after_the_session_closes() {
    let backend = Arc::new(backend());
    let session = backend.open_session().expect("held session");
    let (sender, receiver) = mpsc::channel();
    let waiter = Arc::clone(&backend);
    let handle = thread::spawn(move || {
        let mut waiter = (*waiter).clone();
        let (a, b) = (vector_a(), vector_b());
        let value = waiter
            .with_backend_session(|view| {
                view.add_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b))
            })
            .expect("queued scoped entry")
            .expect("queued operation");
        sender
            .send(value.as_slice::<f64>().expect("f64 payload").to_vec())
            .expect("send result");
    });
    // Close only once the scoped entry is actually queued behind the session, so
    // the test observes the release order rather than an ordinary reopen.
    assert!(
        backend
            .shared
            .arbiter
            .wait_for_waiter_count_for_test(1, Duration::from_secs(30)),
        "the scoped entry must be queued behind the held session"
    );
    session.close().expect("affinity restores");
    assert_eq!(
        receiver
            .recv_timeout(Duration::from_secs(30))
            .expect("queued entry completes"),
        vec![3.0]
    );
    handle.join().expect("waiter thread");
}

/// A held session is `!Send + !Sync`: admission, the resource checkout and the
/// caller's CPU mask belong to the opening thread.
///
/// The probe is the `assert_not_impl_any` ambiguity idiom from the
/// `static_assertions` crate: a second `AmbiguousIfImpl` impl applies only when
/// the type does satisfy the bound, which makes the method reference ambiguous
/// and fails the build.
#[test]
fn held_session_is_neither_send_nor_sync() {
    const _: fn() = || {
        trait AmbiguousIfImpl<A> {
            fn item() {}
        }
        struct InvalidSend;
        struct InvalidSync;
        impl<T: ?Sized> AmbiguousIfImpl<()> for T {}
        impl<T: ?Sized + Send> AmbiguousIfImpl<InvalidSend> for T {}
        impl<T: ?Sized + Sync> AmbiguousIfImpl<InvalidSync> for T {}
        let _ = <CpuHeldSession<'static> as AmbiguousIfImpl<_>>::item;
    };
}
