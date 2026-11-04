use tenferro_ad::prelude::*;

#[test]
fn prelude_calls_traced_ad_operation() {
    let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    let loss = (&x * &x).unwrap();
    let gradient = loss.grad(&x).unwrap();
    assert_eq!(gradient.rank, 0);
}
