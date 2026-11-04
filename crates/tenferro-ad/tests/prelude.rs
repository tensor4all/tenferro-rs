use tenferro_ad::prelude::*;
use tenferro_runtime::TraceContext;

#[test]
fn prelude_calls_traced_ad_operation() {
    let trace = TraceContext::new();
    let x = trace.from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    let loss = (&x * &x).unwrap();
    let gradient = loss.grad(&x).unwrap();
    assert_eq!(gradient.rank(), 0);
}
