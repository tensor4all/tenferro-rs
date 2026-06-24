//! Simple first-order optimizers for in-place parameter updates.

use tenferro_tensor::Tensor;

/// Stochastic gradient descent with a fixed learning rate.
///
/// Updates each parameter tensor in place: `param -= lr * grad`.
// Kept for reference and unit tests; the training loop currently uses Adam.
#[allow(dead_code)]
pub(crate) struct Sgd {
    lr: f64,
}

#[allow(dead_code)]
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

/// Adam optimizer with first- and second-moment estimates.
///
/// Buffers `m` and `v` are allocated lazily on the first call to `step` and
/// reused across iterations.
pub(crate) struct Adam {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    t: usize,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
}

impl Adam {
    /// Create a new Adam optimizer with the given learning rate.
    ///
    /// Default hyperparameters: beta1=0.9, beta2=0.999, eps=1e-8.
    pub(crate) fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    /// Override the learning rate used by subsequent `step` calls.
    ///
    /// This lets the training loop apply a learning-rate schedule without
    /// discarding the accumulated first- and second-moment estimates.
    pub(crate) fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    /// Perform one in-place parameter update using Adam.
    pub(crate) fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        assert_eq!(
            params.len(),
            grads.len(),
            "params and grads must have the same length"
        );

        // Lazy initialization of first- and second-moment buffers.
        if self.m.is_empty() {
            for param in params.iter() {
                let shape: Vec<usize> = param.shape().to_vec();
                let len = shape.iter().product::<usize>();
                self.m.push(
                    Tensor::from_vec_col_major(shape.clone(), vec![0.0_f64; len])
                        .expect("Adam first-moment buffer shape matches data length"),
                );
                self.v.push(
                    Tensor::from_vec_col_major(shape, vec![0.0_f64; len])
                        .expect("Adam second-moment buffer shape matches data length"),
                );
            }
        }

        self.t += 1;
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for ((param, grad), (m, v)) in params
            .iter_mut()
            .zip(grads.iter())
            .zip(self.m.iter_mut().zip(self.v.iter_mut()))
        {
            let p = param
                .as_slice_mut::<f64>()
                .expect("parameter tensor must be f64");
            let g = grad.as_slice::<f64>().expect("gradient tensor must be f64");
            let m_buf = m.as_slice_mut::<f64>().expect("m buffer must be f64");
            let v_buf = v.as_slice_mut::<f64>().expect("v buffer must be f64");
            assert_eq!(
                p.len(),
                g.len(),
                "parameter and gradient lengths must match"
            );
            assert_eq!(
                p.len(),
                m_buf.len(),
                "parameter and m-buffer lengths must match"
            );
            assert_eq!(
                p.len(),
                v_buf.len(),
                "parameter and v-buffer lengths must match"
            );

            for i in 0..p.len() {
                m_buf[i] = self.beta1 * m_buf[i] + (1.0 - self.beta1) * g[i];
                v_buf[i] = self.beta2 * v_buf[i] + (1.0 - self.beta2) * g[i] * g[i];
                let m_hat = m_buf[i] / bias_correction1;
                let v_hat = v_buf[i] / bias_correction2;
                p[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
    }
}

/// Step-decayed learning rate schedule.
///
/// Returns `base` during the first half of training, `base / 2` during the
/// third quarter, and `base / 4` during the final quarter. Decaying the
/// learning rate sharpens convergence of the PINN once Adam has reached a
/// coarse minimum, which is what keeps the predicted soliton from losing
/// amplitude at later times.
pub(crate) fn step_decay_lr(epoch: usize, total_epochs: usize, base: f64) -> f64 {
    if epoch < total_epochs / 2 {
        base
    } else if epoch < 3 * total_epochs / 4 {
        base * 0.5
    } else {
        base * 0.25
    }
}

#[cfg(test)]
mod tests;
