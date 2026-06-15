use tenferro_runtime::{DType, DotGeneralConfig, TracedTensor};

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
