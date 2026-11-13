//! Guards of the eager AD registry: the exception table and the generated page.

use super::registry::{api_docs_sync, API_EXCEPTIONS, API_OPS};

#[test]
fn eager_ad_rows_declare_their_session_signature() {
    let names: Vec<&str> = API_OPS.iter().map(|op| op.name).collect();
    assert_eq!(
        names,
        [
            "grad",
            "vjp",
            "jvp",
            "backward",
            "detach",
            "duplicate_value",
            "constant_from",
            "constant_from_host",
            "variable_from"
        ]
    );
    for op in API_OPS {
        assert_eq!(op.layers, ["eager"], "{}", op.name);
        let [(layer, signature)] = op.signatures else {
            panic!("{} declares no single eager signature", op.name);
        };
        assert_eq!(*layer, "eager");
        let prefix = format!(
            "pub fn tenferro_ad::EagerSession<'_>::{}(&mut self",
            op.name
        );
        assert!(signature.starts_with(&prefix), "{signature}");
        assert!(!signature.contains("Engine"), "{signature}");
    }
}

#[test]
fn exception_table_lists_exact_signatures() {
    let ids: Vec<&str> = API_EXCEPTIONS.iter().map(|e| e.id).collect();
    assert_eq!(ids, ["E11", "E12", "E13"]);
    for exception in API_EXCEPTIONS {
        for symbol in exception.symbols {
            assert!(symbol.starts_with("pub fn tenferro_ad::"), "{symbol}");
        }
    }
}

#[test]
fn api_docs_are_current() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/spec/generated/api-ops.md"
    );
    let current = std::fs::read_to_string(path).expect("read docs/spec/generated/api-ops.md");
    let expected = api_docs_sync(&current);
    if std::env::var_os("UPDATE_API_DOCS").is_some() {
        std::fs::write(path, &expected).expect("write docs/spec/generated/api-ops.md");
        return;
    }
    assert!(
        current == expected,
        "docs/spec/generated/api-ops.md is stale; run \
         `UPDATE_API_DOCS=1 cargo test -p tenferro-ad api_docs_are_current`"
    );
}
