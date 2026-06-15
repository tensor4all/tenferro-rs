//! Simple first-order optimizers for in-place parameter updates.

use tenferro_tensor::Tensor;

/// Stochastic gradient descent with a fixed learning rate.
///
/// Updates each parameter tensor in place: `param -= lr * grad`.
pub(crate) struct Sgd {
    lr: f64,
}

impl Sgd {
    /// Create a new SGD optimizer with learning rate `lr`.
    pub(crate) fn new(lr: f64) -> Self {
        Self { lr }
    }

    /// Perform one in-place parameter update.
    ///
    /// `params` and `grads` must have the same length and matching element
    /// counts. Each parameter buffer is updated elementwise.
    pub(crate) fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        assert_eq!(
            params.len(),
            grads.len(),
            "params and grads must have the same length"
        );
        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            let p = param
                .as_slice_mut::<f64>()
                .expect("parameter tensor must be f64");
            let g = grad.as_slice::<f64>().expect("gradient tensor must be f64");
            assert_eq!(
                p.len(),
                g.len(),
                "parameter and gradient buffers must have the same length"
            );
            for i in 0..p.len() {
                p[i] -= self.lr * g[i];
            }
        }
    }
}

#[cfg(test)]
mod tests;
