use std::path::PathBuf;

use tenferro_runtime::{GraphCompiler, GraphOpView, TracedTensor};

fn repo_file(path: &str) -> String {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.push("../..");
    root.push(path);
    std::fs::read_to_string(root).expect("source file must be readable")
}

#[test]
fn graph_program_exposes_read_only_lowering_view_for_owner_crates() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let y = &x + &x;
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();

    let view = program.lowering_view();
    let instructions = view.instructions().collect::<Vec<_>>();

    assert_eq!(program.input_count(), 1);
    assert_eq!(program.output_count(), 1);
    assert_eq!(view.input_slots().len(), 1);
    assert_eq!(view.output_slots().len(), 1);
    assert!(!instructions.is_empty());
    assert!(matches!(instructions[0].op(), GraphOpView::Add));
    assert_eq!(instructions[0].output_slots().len(), 1);
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
