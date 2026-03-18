use std::sync::Arc;
use std::thread;

use crate::CompletionEvent;

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn completion_event_is_send() {
    assert_send::<CompletionEvent>();
}

#[test]
fn completion_event_is_sync() {
    assert_sync::<CompletionEvent>();
}

#[test]
fn completion_event_can_be_sent_across_threads() {
    let event = CompletionEvent::noop();

    let handle = thread::spawn(move || {
        let _event = event;
    });

    handle.join().unwrap();
}

#[test]
fn completion_event_can_be_shared_across_threads() {
    let event = Arc::new(CompletionEvent::noop());

    let event_clone = Arc::clone(&event);
    let handle = thread::spawn(move || {
        let _ = &*event_clone;
    });

    handle.join().unwrap();
}
