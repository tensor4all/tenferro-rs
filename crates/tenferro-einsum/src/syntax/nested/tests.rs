use super::*;

fn labels(s: &str) -> Vec<u32> {
    s.bytes().map(u32::from).collect()
}

#[test]
fn parse_group_preserves_intermediate_output_label_order() {
    let nested = NestedEinsum::parse("(ca,ab),cd->bd").unwrap();

    let NestedEinsum::Node {
        subscripts,
        children,
    } = nested
    else {
        panic!("expected root node");
    };

    assert_eq!(subscripts.inputs[0], labels("cb"));

    let NestedEinsum::Node {
        subscripts: inner_subscripts,
        ..
    } = &children[0]
    else {
        panic!("expected grouped child node");
    };

    assert_eq!(inner_subscripts.output, labels("cb"));
}
