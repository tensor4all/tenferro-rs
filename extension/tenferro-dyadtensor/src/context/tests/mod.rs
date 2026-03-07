use super::*;

#[test]
fn set_and_restore_context() {
    let guard0 = set_global_context::<u32>(7);
    let value = with_global_context::<u32, _>(|ctx| Ok(*ctx)).unwrap();
    assert_eq!(value, 7);

    let guard1 = set_global_context::<u32>(11);
    let value = with_global_context::<u32, _>(|ctx| Ok(*ctx)).unwrap();
    assert_eq!(value, 11);

    drop(guard1);
    let value = with_global_context::<u32, _>(|ctx| Ok(*ctx)).unwrap();
    assert_eq!(value, 7);

    drop(guard0);
    let missing = with_global_context::<u32, _>(|ctx| Ok(*ctx));
    assert!(matches!(missing, Err(Error::MissingGlobalContext { .. })));
}

#[test]
fn try_with_global_context_when_missing() {
    let value = try_with_global_context::<usize, _>(|ctx| Ok(*ctx)).unwrap();
    assert_eq!(value, None);
}
