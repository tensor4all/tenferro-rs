use tenferro_ad::{EagerRuntime, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = EagerRuntime::new().unwrap();
    let x = ctx.variable_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?)?;
    let loss = x.mul(&x)?.reduce_sum(Some(&[0]))?;
    loss.backward()?;

    assert_eq!(x.grad()?.unwrap().as_slice::<f64>().unwrap(), &[2.0, 4.0]);

    Ok(())
}
