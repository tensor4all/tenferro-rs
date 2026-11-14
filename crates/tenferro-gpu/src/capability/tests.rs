use std::path::PathBuf;

use tenferro_core_ops::PrimitiveOpKind;

use super::*;

#[test]
fn the_descriptor_helpers_set_the_axes_and_output_dtypes() {
    let native = same_dtype(PrimitiveOpKind::Add, DType::I64, N);
    assert_eq!(native.backend, BackendId::Cuda);
    assert_eq!(native.output_dtype, DType::I64);
    assert_eq!(native.result, N);
    assert_eq!(native.read_inputs, F);
    assert_eq!(native.write_output, U);
    assert_eq!(native.strided_output, U);
    assert_eq!(native.accumulation, U);

    // A refused dtype declares every axis unsupported.
    let refused = same_dtype(PrimitiveOpKind::Exp, DType::C64, U);
    assert_eq!(refused.read_inputs, U);
    assert_eq!(refused.result, U);

    let real_output = owned_level(PrimitiveOpKind::Abs, DType::C64, DType::F64, N);
    assert_eq!(real_output.output_dtype, DType::F64);
    assert_eq!(real_output.read_inputs, F);

    let dot = dot_native(DType::C32);
    assert_eq!(dot.op, PrimitiveOpKind::DotGeneral);
    assert_eq!(dot.output_dtype, DType::C32);
    assert_eq!(dot.result, N);
    assert_eq!(dot.read_inputs, F);
    assert_eq!(dot.write_output, N);
    assert_eq!(dot.strided_output, U);
    assert_eq!(dot.accumulation, N);
}

const GUIDE: &str = "../../docs/guides/devices-and-gpu.md";
const START: &str = "<!-- cuda-core-capability:start -->";
const END: &str = "<!-- cuda-core-capability:end -->";
const WRITE_ENV: &str = "TENFERRO_WRITE_CAPABILITY_DOCS";

fn guide_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(GUIDE)
}

fn guide_and_block() -> (String, String) {
    let path = guide_path();
    let text = std::fs::read_to_string(&path).expect("the CUDA guide must be readable");
    let start = text.find(START).expect("guide has the table start marker") + START.len();
    let end = text[start..]
        .find(END)
        .expect("guide has the table end marker")
        + start;
    let block = text[start..end]
        .strip_prefix('\n')
        .unwrap_or(&text[start..end])
        .to_string();
    (text, block)
}

#[test]
fn the_guide_table_is_rendered_from_the_capability_table() {
    let rendered = render_capability_markdown();
    let (text, block) = guide_and_block();
    if block == rendered {
        return;
    }
    if std::env::var_os(WRITE_ENV).is_some() {
        let start = text.find(START).expect("guide has the table start marker") + START.len();
        let end = text[start..]
            .find(END)
            .expect("guide has the table end marker")
            + start;
        // One newline after the start marker, the rows, then the end marker on its own line.
        let rewritten = format!("{}\n{}{}", &text[..start], rendered, &text[end..]);
        std::fs::write(guide_path(), rewritten).expect("the CUDA guide must be writable");
        return;
    }
    panic!(
        "the CUDA guide table differs from `cuda_capabilities()`; regenerate it with\n    \
         {WRITE_ENV}=1 cargo test -p tenferro-gpu the_guide_table_is_rendered\n\
         expected block:\n{rendered}"
    );
}

/// The distinct operations of a table, in first-appearance order.
fn operations(entries: &[OperationCapability]) -> Vec<PrimitiveOpKind> {
    let mut ops: Vec<PrimitiveOpKind> = Vec::new();
    for entry in entries {
        if !ops.contains(&entry.op) {
            ops.push(entry.op);
        }
    }
    ops
}

#[test]
fn the_cpu_and_cuda_tables_cover_the_same_operations() {
    let cuda = operations(cuda_capability_table());
    let cpu = operations(tenferro_cpu::cpu_capabilities());
    // The shared conformance dispatcher has one case per catalogued operation; a one-sided
    // table addition would leave a case missing on the other lane. Order is each table's own.
    assert!(
        cuda.iter().all(|op| cpu.contains(op)) && cpu.iter().all(|op| cuda.contains(op)),
        "the CUDA and CPU tables cover different operations: CUDA {cuda:?}, CPU {cpu:?}"
    );
}
