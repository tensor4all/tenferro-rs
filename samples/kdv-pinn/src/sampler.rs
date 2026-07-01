//! Sampling utilities for collocation, initial-condition, and boundary points.
//!
//! All samplers produce column-major `Tensor`s with shape `[n, 1]` so they can
//! be stacked directly into `[n, 2]` space-time inputs for the network.

use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use tenferro_tensor::Tensor;

/// Space-time domain sampler for the KdV PINN.
pub(crate) struct Sampler {
    x_min: f64,
    x_max: f64,
    t_min: f64,
    t_max: f64,
}

impl Sampler {
    pub(crate) fn new(x_min: f64, x_max: f64, t_min: f64, t_max: f64) -> Self {
        Self {
            x_min,
            x_max,
            t_min,
            t_max,
        }
    }

    /// Sample `n` collocation points in the interior of the space-time domain.
    ///
    /// Returns two tensors of shape `[n, 1]`: the sampled `x` column and the
    /// sampled `t` column. Both columns are built from the same random draws.
    pub(crate) fn collocation<R: Rng>(&self, n: usize, rng: &mut R) -> (Tensor, Tensor) {
        let x_dist = Uniform::new(self.x_min, self.x_max);
        let t_dist = Uniform::new(self.t_min, self.t_max);
        let mut x = Vec::with_capacity(n);
        let mut t = Vec::with_capacity(n);
        for _ in 0..n {
            x.push(x_dist.sample(rng));
            t.push(t_dist.sample(rng));
        }
        (
            Tensor::from_vec_col_major(vec![n, 1], x)
                .expect("collocation x data length matches shape"),
            Tensor::from_vec_col_major(vec![n, 1], t)
                .expect("collocation t data length matches shape"),
        )
    }

    /// Sample `n` initial-condition points at `t = 0`.
    ///
    /// Returns `(x, u)` tensors of shape `[n, 1]` where `u` is the exact KdV
    /// single-soliton profile at `t = 0`.
    pub(crate) fn initial<R: Rng>(&self, n: usize, rng: &mut R) -> (Tensor, Tensor) {
        let x_dist = Uniform::new(self.x_min, self.x_max);
        let mut x = Vec::with_capacity(n);
        let mut u = Vec::with_capacity(n);
        for _ in 0..n {
            let xi = x_dist.sample(rng);
            x.push(xi);
            u.push(2.0 * (1.0 / xi.cosh()).powi(2));
        }
        (
            Tensor::from_vec_col_major(vec![n, 1], x).expect("initial x data length matches shape"),
            Tensor::from_vec_col_major(vec![n, 1], u)
                .expect("initial target data length matches shape"),
        )
    }

    /// Sample `n` boundary points on `x = x_min` and `x = x_max`.
    ///
    /// Exactly half of the samples use `x_min` and the other half use `x_max`
    /// (the first `n/2` points are at `x_min`, the remainder at `x_max`). Times
    /// are sampled uniformly in `[t_min, t_max]` independently for each point.
    pub(crate) fn boundary<R: Rng>(&self, n: usize, rng: &mut R) -> (Tensor, Tensor, Tensor) {
        let t_dist = Uniform::new(self.t_min, self.t_max);
        let mut x = Vec::with_capacity(n);
        let mut t = Vec::with_capacity(n);
        let mut u = Vec::with_capacity(n);
        let n_min = n / 2;
        for i in 0..n {
            let ti = t_dist.sample(rng);
            let xi = if i < n_min { self.x_min } else { self.x_max };
            x.push(xi);
            t.push(ti);
            u.push(2.0 * (1.0 / ((xi - 4.0 * ti).cosh())).powi(2));
        }
        (
            Tensor::from_vec_col_major(vec![n, 1], x)
                .expect("boundary x data length matches shape"),
            Tensor::from_vec_col_major(vec![n, 1], t)
                .expect("boundary t data length matches shape"),
            Tensor::from_vec_col_major(vec![n, 1], u)
                .expect("boundary target data length matches shape"),
        )
    }
}

#[cfg(test)]
mod tests;
