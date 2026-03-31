use crate::{Error, Result, Tensor};

#[derive(Debug)]
pub struct JvpResult {
    pub outputs: Vec<Tensor>,
    pub output_tangents: Vec<Option<Tensor>>,
}

pub fn jvp<F>(f: F, primals: &[Tensor], tangents: &[Option<Tensor>]) -> Result<JvpResult>
where
    F: FnOnce(&[Tensor]) -> Result<Vec<Tensor>>,
{
    let outputs = f(primals)?;
    let _ = tangents;
    let _ = outputs;
    Err(Error::UnsupportedAdOp { op: "jvp" })
}
