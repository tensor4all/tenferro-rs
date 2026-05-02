use tenferro_device::Error;

use super::Subscripts;

#[test]
fn parse_rejects_parenthesized_order_without_discarding_it() {
    let err = Subscripts::parse("ij,(jk,kl)->il").unwrap_err();

    assert!(matches!(
        err,
        Error::InvalidArgument(message)
            if message.contains("parentheses")
                && message.contains("NestedEinsum::parse")
    ));
}
