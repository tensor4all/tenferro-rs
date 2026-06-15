use tenferro_runtime::{DType, DotGeneralConfig, TracedTensor};
use tenferro_tensor::Tensor;

/// A fully-connected layer using `TracedTensor` placeholders for weight and bias.
#[allow(dead_code)]
pub(crate) struct Linear {
    pub weight: TracedTensor,
    pub bias: TracedTensor,
}

#[allow(dead_code)]
impl Linear {
    /// Create a new `Linear` layer with the given input/output feature sizes.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weight = TracedTensor::input_concrete_shape(DType::F64, &[in_features, out_features]);
        let bias = TracedTensor::input_concrete_shape(DType::F64, &[out_features]);
        Self { weight, bias }
    }

    /// Apply the layer to an input tensor.
    pub fn forward(&self, x: &TracedTensor) -> TracedTensor {
        let y = x.dot_general(
            &self.weight,
            DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        );
        let bias_broadcast = self.bias.reshape(&[
            1,
            self.bias
                .try_concrete_shape()
                .expect("placeholder shape is concrete")[0],
        ]);
        y.add(&bias_broadcast)
    }
}

/// A multi-layer perceptron built from `Linear` layers with `tanh` activations.
#[allow(dead_code)]
pub(crate) struct Mlp {
    layers: Vec<Linear>,
}

#[allow(dead_code)]
impl Mlp {
    /// Create a new `Mlp` from a slice of layer sizes.
    pub fn new(layer_sizes: &[usize]) -> Self {
        assert!(
            layer_sizes.len() >= 2,
            "Mlp needs at least input and output sizes"
        );
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            layers.push(Linear::new(layer_sizes[i], layer_sizes[i + 1]));
        }
        Self { layers }
    }

    /// Run a forward pass through the network.
    pub fn forward(&self, x: &TracedTensor) -> TracedTensor {
        let mut y = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            y = layer.forward(&y);
            if i < self.layers.len() - 1 {
                y = y.tanh();
            }
        }
        y
    }

    /// Return references to all parameter placeholders.
    pub fn parameters(&self) -> Vec<&TracedTensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.push(&layer.weight);
            params.push(&layer.bias);
        }
        params
    }

    /// Initialize concrete `Tensor` buffers for every layer weight and bias.
    ///
    /// Weights use Xavier-style uniform initialization; biases are zeros.
    pub(crate) fn init_tensors(&self, rng: &mut impl rand::Rng) -> Vec<Tensor> {
        use rand::distributions::{Distribution, Uniform};
        let mut tensors = Vec::new();
        for layer in &self.layers {
            let shape = layer.weight.try_concrete_shape().unwrap();
            let fan_in = shape[0];
            let fan_out = shape[1];
            let scale = (6.0 / (fan_in + fan_out) as f64).sqrt();
            let dist = Uniform::new(-scale, scale);
            let w_data: Vec<f64> = (0..shape.iter().product::<usize>())
                .map(|_| dist.sample(rng))
                .collect();
            tensors.push(Tensor::from_vec_col_major(shape.clone(), w_data));

            let b_shape = layer.bias.try_concrete_shape().unwrap();
            let b_data = vec![0.0_f64; b_shape.iter().product::<usize>()];
            tensors.push(Tensor::from_vec_col_major(b_shape.clone(), b_data));
        }
        tensors
    }

    /// Return `(placeholder, dtype, shape)` tuples for every parameter.
    pub fn input_specs(&self) -> Vec<(&TracedTensor, DType, Vec<usize>)> {
        self.parameters()
            .iter()
            .map(|p| {
                (
                    *p,
                    DType::F64,
                    p.try_concrete_shape()
                        .expect("placeholder shape is concrete"),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenferro_cpu::CpuBackend;
    use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
    use tenferro_tensor::Tensor;

    #[test]
    fn mlp_forward_shape() {
        let net = Mlp::new(&[2, 16, 16, 1]);
        let x = TracedTensor::input_concrete_shape(DType::F64, &[4, 2]);
        let y = net.forward(&x);
        assert_eq!(y.rank, 2);
        assert_eq!(y.try_concrete_shape(), Some(vec![4, 1]));

        let specs = net.input_specs();
        let bindings: Vec<(&TracedTensor, DType, &[usize])> = specs
            .iter()
            .map(|(p, dtype, shape)| (*p, *dtype, shape.as_slice()))
            .chain(std::iter::once((&x, DType::F64, &[4, 2][..])))
            .collect();

        let mut compiler = GraphCompiler::new();
        let program = compiler.compile_with_input_specs(&y, &bindings).unwrap();
        let mut executor = GraphExecutor::new(CpuBackend::new());

        let x_tensor = Tensor::from_vec_col_major(vec![4, 2], vec![0.0_f64; 8]);
        let param_tensors: Vec<Tensor> = specs
            .iter()
            .map(|(_, _, shape)| {
                let len = shape.iter().product::<usize>();
                Tensor::from_vec_col_major(shape.clone(), vec![0.1_f64; len])
            })
            .collect();
        let mut bind_tensors: Vec<(&TracedTensor, &Tensor)> = net
            .parameters()
            .iter()
            .zip(param_tensors.iter())
            .map(|(p, t)| (*p, t))
            .collect();
        bind_tensors.push((&x, &x_tensor));

        let out = executor.run_with_inputs(&program, &bind_tensors).unwrap();
        assert_eq!(out.shape(), &[4, 1]);
    }
}
