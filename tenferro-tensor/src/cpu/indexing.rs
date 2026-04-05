use crate::config::{GatherConfig, PadConfig, ScatterConfig, SliceConfig};
use crate::Tensor;

pub fn gather(_input: &Tensor, _config: &GatherConfig) -> Tensor {
    todo!("gather")
}

pub fn scatter(_input: &Tensor, _updates: &Tensor, _config: &ScatterConfig) -> Tensor {
    todo!("scatter")
}

pub fn slice(_input: &Tensor, _config: &SliceConfig) -> Tensor {
    todo!("slice")
}

pub fn dynamic_slice(_input: &Tensor, _starts: &Tensor) -> Tensor {
    todo!("dynamic_slice")
}

pub fn pad(_input: &Tensor, _config: &PadConfig) -> Tensor {
    todo!("pad")
}

pub fn concatenate(_inputs: &[&Tensor], _axis: usize) -> Tensor {
    todo!("concatenate")
}

pub fn reverse(_input: &Tensor, _axes: &[usize]) -> Tensor {
    todo!("reverse")
}
