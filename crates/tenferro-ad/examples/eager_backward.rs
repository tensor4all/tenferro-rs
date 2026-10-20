use tenferro_ad::{EagerRuntime, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = EagerRuntime::new()?;
    let (x, loss) = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |session| {
        let x = session.variable_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?)?;
        let squared = session.mul(&x, &x)?;
        let loss = session.reduce_sum(&squared, Some(&[0]))?;
        Ok::<_, tenferro_ad::Error>((x, loss))
    })?;
    loss.backward(&mut tenferro_cpu::CpuEngine::serial())?;

    assert_eq!(x.grad()?.unwrap().as_slice::<f64>().unwrap(), &[2.0, 4.0]);

    Ok(())
}
