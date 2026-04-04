use tenferro_ops::config::{GatherConfig, PadConfig, ScatterConfig, SliceConfig};
use tenferro_tensor::v2::Tensor;

pub fn gather(_input: &Tensor, _config: &GatherConfig) -> Tensor {
    todo!("indexing::gather")
}

pub fn scatter(_input: &Tensor, _updates: &Tensor, _config: &ScatterConfig) -> Tensor {
    todo!("indexing::scatter")
}

pub fn slice(_input: &Tensor, _config: &SliceConfig) -> Tensor {
    todo!("indexing::slice")
}

pub fn dynamic_slice(_input: &Tensor, _starts: &Tensor) -> Tensor {
    todo!("indexing::dynamic_slice")
}

pub fn pad(_input: &Tensor, _config: &PadConfig) -> Tensor {
    todo!("indexing::pad")
}

pub fn concatenate(_inputs: &[&Tensor], _axis: usize) -> Tensor {
    todo!("indexing::concatenate")
}

pub fn reverse(_input: &Tensor, _axes: &[usize]) -> Tensor {
    todo!("indexing::reverse")
}
