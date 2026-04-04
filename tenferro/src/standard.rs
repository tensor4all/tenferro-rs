use computegraph::Operand;
use tenferro_ops::config::CompareDir;
use tenferro_tensor::Tensor;

pub fn neg(input: &Tensor) -> Tensor {
    input.neg()
}

pub fn conj(input: &Tensor) -> Tensor {
    Operand::conj(input)
}

pub fn div(_lhs: &Tensor, _rhs: &Tensor) -> Tensor {
    todo!("standard::div")
}

pub fn abs(_input: &Tensor) -> Tensor {
    todo!("standard::abs")
}

pub fn sign(_input: &Tensor) -> Tensor {
    todo!("standard::sign")
}

pub fn maximum(_lhs: &Tensor, _rhs: &Tensor) -> Tensor {
    todo!("standard::maximum")
}

pub fn minimum(_lhs: &Tensor, _rhs: &Tensor) -> Tensor {
    todo!("standard::minimum")
}

pub fn compare(_lhs: &Tensor, _rhs: &Tensor, _dir: &CompareDir) -> Tensor {
    todo!("standard::compare")
}

pub fn select(_pred: &Tensor, _on_true: &Tensor, _on_false: &Tensor) -> Tensor {
    todo!("standard::select")
}

pub fn clamp(_input: &Tensor, _lower: &Tensor, _upper: &Tensor) -> Tensor {
    todo!("standard::clamp")
}

pub fn exp(_input: &Tensor) -> Tensor {
    todo!("standard::exp")
}

pub fn log(_input: &Tensor) -> Tensor {
    todo!("standard::log")
}

pub fn sin(_input: &Tensor) -> Tensor {
    todo!("standard::sin")
}

pub fn cos(_input: &Tensor) -> Tensor {
    todo!("standard::cos")
}

pub fn tanh(_input: &Tensor) -> Tensor {
    todo!("standard::tanh")
}

pub fn sqrt(_input: &Tensor) -> Tensor {
    todo!("standard::sqrt")
}

pub fn rsqrt(_input: &Tensor) -> Tensor {
    todo!("standard::rsqrt")
}

pub fn pow(_lhs: &Tensor, _rhs: &Tensor) -> Tensor {
    todo!("standard::pow")
}

pub fn expm1(_input: &Tensor) -> Tensor {
    todo!("standard::expm1")
}

pub fn log1p(_input: &Tensor) -> Tensor {
    todo!("standard::log1p")
}
