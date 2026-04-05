use crate::config::{GatherConfig, PadConfig, ScatterConfig, SliceConfig};
use crate::types::{dispatch_tensor, flat_to_multi, Tensor, TypedTensor};

pub fn gather(_input: &Tensor, _config: &GatherConfig) -> Tensor {
    todo!("gather")
}

pub fn scatter(_input: &Tensor, _updates: &Tensor, _config: &ScatterConfig) -> Tensor {
    todo!("scatter")
}

pub fn slice(input: &Tensor, config: &SliceConfig) -> Tensor {
    dispatch_tensor!(input, tensor => typed_slice(tensor, config))
}

pub fn dynamic_slice(_input: &Tensor, _starts: &Tensor) -> Tensor {
    todo!("dynamic_slice")
}

pub fn pad(_input: &Tensor, _config: &PadConfig) -> Tensor {
    todo!("pad")
}

pub fn concatenate(_inputs: &[&Tensor], _axis: usize) -> Tensor {
    todo!("concatenate")
}

pub fn reverse(input: &Tensor, axes: &[usize]) -> Tensor {
    dispatch_tensor!(input, tensor => typed_reverse(tensor, axes))
}

fn typed_slice<T: Copy + Clone>(input: &TypedTensor<T>, config: &SliceConfig) -> TypedTensor<T> {
    let rank = input.shape.len();
    assert_eq!(config.starts.len(), rank, "slice: starts rank mismatch");
    assert_eq!(config.limits.len(), rank, "slice: limits rank mismatch");
    assert_eq!(config.strides.len(), rank, "slice: strides rank mismatch");

    let out_shape: Vec<usize> = input
        .shape
        .iter()
        .enumerate()
        .map(|(axis, &dim)| {
            let start = config.starts[axis];
            let limit = config.limits[axis];
            let stride = config.strides[axis];
            assert!(start <= limit, "slice: start exceeds limit on axis {axis}");
            assert!(limit <= dim, "slice: limit out of bounds on axis {axis}");
            assert!(stride > 0, "slice: stride must be positive on axis {axis}");
            let span = limit - start;
            (span + stride - 1) / stride
        })
        .collect();

    let out_len: usize = out_shape.iter().product();
    let mut out_data = Vec::with_capacity(out_len);
    let mut out_idx = vec![0usize; rank];
    let mut in_idx = vec![0usize; rank];

    for flat in 0..out_len {
        flat_to_multi(flat, &out_shape, &mut out_idx);
        for axis in 0..rank {
            in_idx[axis] = config.starts[axis] + out_idx[axis] * config.strides[axis];
        }
        out_data.push(*input.get(&in_idx));
    }

    TypedTensor::from_vec(out_shape, out_data)
}

fn typed_reverse<T: Copy + Clone>(input: &TypedTensor<T>, axes: &[usize]) -> TypedTensor<T> {
    let rank = input.shape.len();
    let mut reverse_axis = vec![false; rank];
    for &axis in axes {
        assert!(axis < rank, "reverse: axis out of bounds");
        reverse_axis[axis] = true;
    }

    let out_len = input.n_elements();
    let mut out_data = Vec::with_capacity(out_len);
    let mut out_idx = vec![0usize; rank];
    let mut in_idx = vec![0usize; rank];

    for flat in 0..out_len {
        flat_to_multi(flat, &input.shape, &mut out_idx);
        for axis in 0..rank {
            in_idx[axis] = if reverse_axis[axis] {
                input.shape[axis] - 1 - out_idx[axis]
            } else {
                out_idx[axis]
            };
        }
        out_data.push(*input.get(&in_idx));
    }

    TypedTensor::from_vec(input.shape.clone(), out_data)
}
