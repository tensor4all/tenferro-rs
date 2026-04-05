use crate::config::{GatherConfig, PadConfig, ScatterConfig, SliceConfig};
use crate::types::{dispatch_tensor, flat_to_multi, Tensor, TypedTensor};

trait TensorAsTyped<T> {
    fn as_typed(&self) -> Option<&TypedTensor<T>>;
}

impl TensorAsTyped<f32> for Tensor {
    fn as_typed(&self) -> Option<&TypedTensor<f32>> {
        match self {
            Tensor::F32(tensor) => Some(tensor),
            _ => None,
        }
    }
}

impl TensorAsTyped<f64> for Tensor {
    fn as_typed(&self) -> Option<&TypedTensor<f64>> {
        match self {
            Tensor::F64(tensor) => Some(tensor),
            _ => None,
        }
    }
}

impl TensorAsTyped<num_complex::Complex<f32>> for Tensor {
    fn as_typed(&self) -> Option<&TypedTensor<num_complex::Complex<f32>>> {
        match self {
            Tensor::C32(tensor) => Some(tensor),
            _ => None,
        }
    }
}

impl TensorAsTyped<num_complex::Complex<f64>> for Tensor {
    fn as_typed(&self) -> Option<&TypedTensor<num_complex::Complex<f64>>> {
        match self {
            Tensor::C64(tensor) => Some(tensor),
            _ => None,
        }
    }
}

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

pub fn concatenate(inputs: &[&Tensor], axis: usize) -> Tensor {
    let first = inputs
        .first()
        .copied()
        .expect("concatenate requires at least one input");
    dispatch_tensor!(first, tensor => typed_concatenate_from_dyn_inputs(tensor, inputs, axis))
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

fn typed_concatenate_from_dyn_inputs<T>(
    _first: &TypedTensor<T>,
    inputs: &[&Tensor],
    axis: usize,
) -> TypedTensor<T>
where
    T: Copy + Clone,
    Tensor: TensorAsTyped<T>,
{
    let typed_inputs = collect_typed_inputs(inputs);
    typed_concatenate(&typed_inputs, axis)
}

fn collect_typed_inputs<'a, T>(inputs: &[&'a Tensor]) -> Vec<&'a TypedTensor<T>>
where
    Tensor: TensorAsTyped<T>,
{
    inputs
        .iter()
        .map(|tensor| {
            TensorAsTyped::<T>::as_typed(*tensor)
                .expect("concatenate: dtype mismatch across inputs")
        })
        .collect()
}

fn typed_concatenate<T: Copy + Clone>(inputs: &[&TypedTensor<T>], axis: usize) -> TypedTensor<T> {
    let first = inputs[0];
    let rank = first.shape.len();
    assert!(axis < rank, "concatenate: axis out of bounds");

    let mut out_shape = first.shape.clone();
    let mut axis_extent = 0usize;
    for input in inputs {
        assert_eq!(input.shape.len(), rank, "concatenate: rank mismatch");
        for dim in 0..rank {
            if dim == axis {
                axis_extent += input.shape[dim];
            } else {
                assert_eq!(
                    input.shape[dim], first.shape[dim],
                    "concatenate: non-concat dimensions must match"
                );
            }
        }
    }
    out_shape[axis] = axis_extent;

    let segment_ends: Vec<usize> = inputs
        .iter()
        .scan(0usize, |sum, input| {
            *sum += input.shape[axis];
            Some(*sum)
        })
        .collect();

    let out_len: usize = out_shape.iter().product();
    let mut out_data = Vec::with_capacity(out_len);
    let mut out_idx = vec![0usize; rank];
    let mut in_idx = vec![0usize; rank];

    for flat in 0..out_len {
        flat_to_multi(flat, &out_shape, &mut out_idx);
        let concat_idx = out_idx[axis];
        let input_pos = segment_ends
            .iter()
            .position(|&end| concat_idx < end)
            .expect("concatenate: output index must map to an input");
        let axis_base = if input_pos == 0 {
            0
        } else {
            segment_ends[input_pos - 1]
        };

        in_idx.copy_from_slice(&out_idx);
        in_idx[axis] -= axis_base;
        out_data.push(*inputs[input_pos].get(&in_idx));
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
