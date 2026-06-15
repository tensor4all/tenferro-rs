use tenferro_runtime::{DType, DotGeneralConfig, TracedTensor};
use tenferro_tensor::Tensor;

/// A fully-connected layer using `TracedTensor` placeholders for weight and bias.
pub(crate) struct Linear {
    pub weight: TracedTensor,
    pub bias: TracedTensor,
}

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
pub(crate) struct Mlp {
    layers: Vec<Linear>,
}

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
            let shape = layer
                .weight
                .try_concrete_shape()
                .expect("weight placeholder must have a concrete shape");
            let fan_in = shape[0];
            let fan_out = shape[1];
            let scale = (6.0 / (fan_in + fan_out) as f64).sqrt();
            let dist = Uniform::new(-scale, scale);
            let w_data: Vec<f64> = (0..shape.iter().product::<usize>())
                .map(|_| dist.sample(rng))
                .collect();
            tensors.push(Tensor::from_vec_col_major(shape.clone(), w_data));

            let b_shape = layer
                .bias
                .try_concrete_shape()
                .expect("bias placeholder must have a concrete shape");
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
mod tests;
