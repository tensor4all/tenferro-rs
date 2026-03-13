use super::*;

pub(crate) fn stable_inverse_gap<R: num_traits::Float>(left: R, right: R, eta: R) -> R {
    let denom = right * right - left * left;
    if denom == R::zero() {
        R::zero()
    } else {
        let sign = if denom >= R::zero() {
            R::one()
        } else {
            -R::one()
        };
        R::one() / (denom + eta * sign)
    }
}

pub(crate) fn stable_inverse_sigma<R: num_traits::Float>(sigma: R, eta: R) -> R {
    if sigma.abs() > eta {
        R::one() / sigma
    } else {
        R::zero()
    }
}

pub(crate) fn imag_axis_component<T: LinalgScalar>(value: T) -> AdResult<T> {
    let half = scalar_from::<T::Real>(0.5).map_err(to_ad_err)?;
    Ok((value - value.conj()) * T::from_real(half))
}

pub(crate) fn real_diagonal_from_scalar<T: LinalgScalar>(value: T) -> T::Real {
    value.real_part()
}
