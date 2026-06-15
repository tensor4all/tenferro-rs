#![allow(dead_code)]

use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use tenferro_tensor::Tensor;

pub struct Sampler {
    x_min: f64,
    x_max: f64,
    t_min: f64,
    t_max: f64,
}

impl Sampler {
    pub fn new(x_min: f64, x_max: f64, t_min: f64, t_max: f64) -> Self {
        Self {
            x_min,
            x_max,
            t_min,
            t_max,
        }
    }

    pub fn collocation<R: Rng>(&self, n: usize, rng: &mut R) -> Tensor {
        let x_dist = Uniform::new(self.x_min, self.x_max);
        let t_dist = Uniform::new(self.t_min, self.t_max);
        let mut data = Vec::with_capacity(n * 2);
        for _ in 0..n {
            data.push(x_dist.sample(rng));
            data.push(t_dist.sample(rng));
        }
        Tensor::from_vec_col_major(vec![n, 2], data)
    }

    pub fn initial<R: Rng>(&self, n: usize, rng: &mut R) -> (Tensor, Tensor) {
        let x_dist = Uniform::new(self.x_min, self.x_max);
        let mut x = Vec::with_capacity(n);
        let mut u = Vec::with_capacity(n);
        for _ in 0..n {
            let xi = x_dist.sample(rng);
            x.push(xi);
            u.push(2.0 * (1.0 / xi.cosh()).powi(2));
        }
        (
            Tensor::from_vec_col_major(vec![n, 1], x),
            Tensor::from_vec_col_major(vec![n, 1], u),
        )
    }

    pub fn boundary<R: Rng>(&self, n: usize, rng: &mut R) -> (Tensor, Tensor, Tensor) {
        let t_dist = Uniform::new(self.t_min, self.t_max);
        let side_dist = Uniform::new(0.0, 1.0);
        let mut x = Vec::with_capacity(n);
        let mut t = Vec::with_capacity(n);
        let mut u = Vec::with_capacity(n);
        for _ in 0..n {
            let ti = t_dist.sample(rng);
            let xi = if side_dist.sample(rng) < 0.5 {
                self.x_min
            } else {
                self.x_max
            };
            x.push(xi);
            t.push(ti);
            u.push(2.0 * (1.0 / ((xi - 4.0 * ti).cosh())).powi(2));
        }
        (
            Tensor::from_vec_col_major(vec![n, 1], x),
            Tensor::from_vec_col_major(vec![n, 1], t),
            Tensor::from_vec_col_major(vec![n, 1], u),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collocation_has_correct_shape() {
        let sampler = Sampler::new(-5.0, 5.0, 0.0, 1.0);
        let mut rng = rand::thread_rng();
        let xt = sampler.collocation(32, &mut rng);
        assert_eq!(xt.shape(), &[32, 2]);
    }

    #[test]
    fn initial_has_correct_shape_and_zero_time() {
        let sampler = Sampler::new(-5.0, 5.0, 0.0, 1.0);
        let mut rng = rand::thread_rng();
        let (x, u) = sampler.initial(16, &mut rng);
        assert_eq!(x.shape(), &[16, 1]);
        assert_eq!(u.shape(), &[16, 1]);
        let t = x.as_slice::<f64>().unwrap();
        assert!(t.iter().all(|&v| v >= -5.0 && v <= 5.0));
    }
}
