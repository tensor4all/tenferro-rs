use std::path::PathBuf;

fn repo_file(path: &str) -> String {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.push("../..");
    root.push(path);
    std::fs::read_to_string(root).expect("source file must be readable")
}

#[test]
fn traced_tensor_graph_and_attached_data_are_accessor_based() {
    let source = repo_file("crates/tenferro-runtime/src/traced.rs");
    assert!(
        !source.contains("pub graph:"),
        "TracedTensor graph storage must not be a public field"
    );
    assert!(
        !source.contains("pub data:"),
        "TracedTensor attached data storage must not be a public field"
    );
    assert!(
        source.contains("pub fn graph(&self)"),
        "TracedTensor should expose graph inspection through an accessor"
    );
    assert!(
        source.contains("pub fn attached_data(&self)"),
        "TracedTensor should expose optional attached data through an accessor"
    );
}
