use std::collections::BTreeSet;

/// Method names declared between a trait header and its closing brace.
fn method_names<'a>(source: &'a str, header: &str) -> BTreeSet<&'a str> {
    let body = source
        .split_once(header)
        .unwrap_or_else(|| panic!("trait header not found: {header}"))
        .1;
    let body = body.split_once("\n}\n").expect("trait body ends").0;
    body.lines()
        .filter_map(|line| line.strip_prefix("    fn "))
        .filter_map(|rest| rest.split_once('(').map(|(name, _)| name))
        .collect()
}

/// #2214: an operation may not land on the erased surface alone.
///
/// The exemption list is the reviewed set of erased-only operations:
/// `convert` and `cast` change the dtype of a dtype-erased tensor (a typed
/// tensor has one dtype), and `where_select` is on the bool-mask trait.
/// Anything else missing here is a new gap.
#[test]
fn typed_surface_covers_the_erased_surface_except_documented_exemptions() {
    let erased = method_names(
        include_str!("../session_ext.rs"),
        "pub trait TensorSessionOpsExt {",
    );
    let typed = method_names(
        include_str!("../typed_session_ext.rs"),
        "pub trait TypedTensorSessionOpsExt<T: TensorScalar> {",
    );
    let exempt: BTreeSet<&str> = ["convert", "cast", "where_select"].into_iter().collect();
    let missing: Vec<&&str> = erased
        .difference(&typed)
        .filter(|name| !exempt.contains(*name))
        .collect();
    assert!(
        missing.is_empty(),
        "the typed surface is missing {missing:?}; add the wrapper and its read hook, or extend the reviewed exemption list"
    );
}
