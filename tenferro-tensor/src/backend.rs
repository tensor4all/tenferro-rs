use strided_kernel::{col_major_strides, reduce_axis, zip_map2_into, StridedArray, StridedView};
use tenferro_algebra::Semiring;

use crate::config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use crate::{Buffer, Tensor, TypedTensor};

pub(crate) fn typed_view<T: Copy>(tensor: &TypedTensor<T>) -> StridedView<'_, T> {
    match &tensor.buffer {
        Buffer::Host(data) => {
            let strides = col_major_strides(&tensor.shape);
            StridedView::new(data, &tensor.shape, &strides, 0).expect("contiguous host tensor")
        }
        Buffer::Backend(_) => todo!("typed_view for backend buffers"),
    }
}

pub(crate) fn typed_array<T: Clone>(shape: &[usize], fill: T) -> StridedArray<T> {
    let total: usize = shape.iter().product();
    let strides = col_major_strides(shape);
    StridedArray::from_parts(vec![fill; total], shape, &strides, 0)
        .expect("column-major output array")
}

pub(crate) fn tensor_from_array<T: Clone>(array: StridedArray<T>) -> TypedTensor<T> {
    TypedTensor::from_vec(array.dims().to_vec(), array.into_data())
}

/// Standard runtime backend over dynamic [`Tensor`] values.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::{cpu::CpuBackend, TensorBackend};
///
/// let mut backend = CpuBackend::new();
/// ```
pub trait TensorBackend {
    fn add(&mut self, lhs: &Tensor, rhs: &Tensor) -> Tensor;
    fn mul(&mut self, lhs: &Tensor, rhs: &Tensor) -> Tensor;
    fn neg(&mut self, input: &Tensor) -> Tensor;
    fn conj(&mut self, input: &Tensor) -> Tensor;
    fn div(&mut self, lhs: &Tensor, rhs: &Tensor) -> Tensor;
    fn abs(&mut self, input: &Tensor) -> Tensor;
    fn sign(&mut self, input: &Tensor) -> Tensor;
    fn maximum(&mut self, lhs: &Tensor, rhs: &Tensor) -> Tensor;
    fn minimum(&mut self, lhs: &Tensor, rhs: &Tensor) -> Tensor;
    fn compare(&mut self, lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> Tensor;
    fn select(&mut self, pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> Tensor;
    fn clamp(&mut self, input: &Tensor, lower: &Tensor, upper: &Tensor) -> Tensor;

    fn exp(&mut self, input: &Tensor) -> Tensor;
    fn log(&mut self, input: &Tensor) -> Tensor;
    fn sin(&mut self, input: &Tensor) -> Tensor;
    fn cos(&mut self, input: &Tensor) -> Tensor;
    fn tanh(&mut self, input: &Tensor) -> Tensor;
    fn sqrt(&mut self, input: &Tensor) -> Tensor;
    fn rsqrt(&mut self, input: &Tensor) -> Tensor;
    fn pow(&mut self, lhs: &Tensor, rhs: &Tensor) -> Tensor;
    fn expm1(&mut self, input: &Tensor) -> Tensor;
    fn log1p(&mut self, input: &Tensor) -> Tensor;

    fn transpose(&mut self, input: &Tensor, perm: &[usize]) -> Tensor;
    fn reshape(&mut self, input: &Tensor, shape: &[usize]) -> Tensor;
    fn broadcast_in_dim(&mut self, input: &Tensor, shape: &[usize], dims: &[usize]) -> Tensor;
    fn extract_diagonal(&mut self, input: &Tensor, axis_a: usize, axis_b: usize) -> Tensor;
    fn embed_diagonal(&mut self, input: &Tensor, axis_a: usize, axis_b: usize) -> Tensor;
    fn tril(&mut self, input: &Tensor, k: i64) -> Tensor;
    fn triu(&mut self, input: &Tensor, k: i64) -> Tensor;

    fn reduce_sum(&mut self, input: &Tensor, axes: &[usize]) -> Tensor;
    fn reduce_prod(&mut self, input: &Tensor, axes: &[usize]) -> Tensor;
    fn reduce_max(&mut self, input: &Tensor, axes: &[usize]) -> Tensor;
    fn reduce_min(&mut self, input: &Tensor, axes: &[usize]) -> Tensor;

    fn dot_general(&mut self, lhs: &Tensor, rhs: &Tensor, config: &DotGeneralConfig) -> Tensor;

    fn gather(&mut self, operand: &Tensor, start_indices: &Tensor, config: &GatherConfig)
        -> Tensor;
    fn scatter(
        &mut self,
        operand: &Tensor,
        scatter_indices: &Tensor,
        updates: &Tensor,
        config: &ScatterConfig,
    ) -> Tensor;
    fn slice(&mut self, input: &Tensor, config: &SliceConfig) -> Tensor;
    fn dynamic_slice(&mut self, input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) -> Tensor;
    fn pad(&mut self, input: &Tensor, config: &PadConfig) -> Tensor;
    fn concatenate(&mut self, inputs: &[&Tensor], axis: usize) -> Tensor;
    fn reverse(&mut self, input: &Tensor, axes: &[usize]) -> Tensor;

    fn cholesky(&mut self, input: &Tensor) -> Tensor;
    fn triangular_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> Tensor;
    fn lu(&mut self, input: &Tensor) -> Vec<Tensor>;
    fn svd(&mut self, input: &Tensor) -> Vec<Tensor>;
    fn qr(&mut self, input: &Tensor) -> Vec<Tensor>;
    fn eigh(&mut self, input: &Tensor) -> Vec<Tensor>;
    fn eig(&mut self, input: &Tensor) -> Vec<Tensor>;
    fn solve(&mut self, a: &Tensor, b: &Tensor) -> Tensor;
}

/// Algebra-generic backend over typed tensors.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_tensor::{cpu::CpuBackend, SemiringBackend};
///
/// fn needs_semiring_backend<B: SemiringBackend<Standard<f64>>>(_backend: &mut B) {}
/// let mut backend = CpuBackend::new();
/// needs_semiring_backend(&mut backend);
/// ```
pub trait SemiringBackend<Alg: Semiring> {
    fn batched_gemm(
        &mut self,
        lhs: &TypedTensor<Alg::Scalar>,
        rhs: &TypedTensor<Alg::Scalar>,
        config: &DotGeneralConfig,
    ) -> TypedTensor<Alg::Scalar>;

    fn add(
        &mut self,
        lhs: &TypedTensor<Alg::Scalar>,
        rhs: &TypedTensor<Alg::Scalar>,
    ) -> TypedTensor<Alg::Scalar> {
        assert_eq!(lhs.shape, rhs.shape, "add: shape mismatch");
        let mut out = typed_array(&lhs.shape, Alg::zero());
        zip_map2_into(
            &mut out.view_mut(),
            &typed_view(lhs),
            &typed_view(rhs),
            |x, y| Alg::add(x, y),
        )
        .expect("semiring add");
        tensor_from_array(out)
    }

    fn mul(
        &mut self,
        lhs: &TypedTensor<Alg::Scalar>,
        rhs: &TypedTensor<Alg::Scalar>,
    ) -> TypedTensor<Alg::Scalar> {
        assert_eq!(lhs.shape, rhs.shape, "mul: shape mismatch");
        let mut out = typed_array(&lhs.shape, Alg::zero());
        zip_map2_into(
            &mut out.view_mut(),
            &typed_view(lhs),
            &typed_view(rhs),
            |x, y| Alg::mul(x, y),
        )
        .expect("semiring mul");
        tensor_from_array(out)
    }

    fn reduce_sum(
        &mut self,
        input: &TypedTensor<Alg::Scalar>,
        axes: &[usize],
    ) -> TypedTensor<Alg::Scalar> {
        if axes.is_empty() {
            return input.clone();
        }

        let output_shape: Vec<usize> = input
            .shape
            .iter()
            .enumerate()
            .filter(|(axis, _)| !axes.contains(axis))
            .map(|(_, &dim)| dim)
            .collect();

        let strides = col_major_strides(&input.shape);
        let mut current =
            StridedArray::from_parts(input.host_data().to_vec(), &input.shape, &strides, 0)
                .expect("semiring reduce input");

        let mut sorted_axes = axes.to_vec();
        sorted_axes.sort_unstable_by(|a, b| b.cmp(a));
        for axis in sorted_axes {
            current = reduce_axis(
                &current.view(),
                axis,
                |x| x,
                |a, b| Alg::add(a, b),
                Alg::zero(),
            )
            .expect("semiring reduce_sum");
        }
        TypedTensor::from_vec(output_shape, current.into_data())
    }
}
