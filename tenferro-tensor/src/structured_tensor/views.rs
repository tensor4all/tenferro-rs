use tenferro_algebra::Conjugate;

use super::{validate_permutation, StructuredTensor};

impl<T: tenferro_algebra::Scalar> StructuredTensor<T> {
    /// Returns the same logical tensor with permuted logical axes.
    ///
    /// This permutes both the logical axes and the compressed payload class
    /// order, then rebuilds the canonical axis-class representation.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, StructuredTensor, Tensor};
    ///
    /// let dense = Tensor::<f64>::from_slice(
    ///     &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     &[2, 3],
    ///     MemoryOrder::ColumnMajor,
    /// )
    /// .unwrap();
    /// let x = StructuredTensor::from_dense(dense);
    /// let y = x.permute_logical(&[1, 0]).unwrap();
    /// assert_eq!(y.logical_dims(), &[3, 2]);
    /// ```
    pub fn permute_logical(&self, perm: &[usize]) -> tenferro_device::Result<Self> {
        validate_permutation(
            perm,
            self.logical_dims.len(),
            "StructuredTensor::permute_logical",
        )?;

        let permuted_dims: Vec<usize> = perm.iter().map(|&axis| self.logical_dims[axis]).collect();
        let permuted_raw_classes: Vec<usize> =
            perm.iter().map(|&axis| self.axis_classes[axis]).collect();

        let mut seen_classes = vec![false; self.class_count()];
        let mut class_order = Vec::with_capacity(self.class_count());
        for &class_id in &permuted_raw_classes {
            if !seen_classes[class_id] {
                seen_classes[class_id] = true;
                class_order.push(class_id);
            }
        }

        let mut remap = vec![usize::MAX; self.class_count()];
        for (new_class, &old_class) in class_order.iter().enumerate() {
            remap[old_class] = new_class;
        }
        let canonical_classes: Vec<usize> = permuted_raw_classes
            .iter()
            .map(|&old_class| remap[old_class])
            .collect();

        let payload = self.payload.permute(&class_order)?;
        Self::new(permuted_dims, canonical_classes, payload)
    }

    /// Returns the same structured tensor with payload conjugation toggled.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use num_complex::Complex64;
    /// use tenferro_tensor::{MemoryOrder, StructuredTensor, Tensor};
    ///
    /// let payload = Tensor::<Complex64>::from_slice(
    ///     &[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
    ///     &[2],
    ///     MemoryOrder::ColumnMajor,
    /// )
    /// .unwrap();
    /// let x = StructuredTensor::from_diagonal_vector(payload, 2).unwrap();
    /// let y = x.conj();
    /// assert_eq!(y.logical_dims(), x.logical_dims());
    /// ```
    pub fn conj(&self) -> Self
    where
        T: Conjugate,
    {
        Self {
            payload: self.payload.conj(),
            logical_dims: self.logical_dims.clone(),
            axis_classes: self.axis_classes.clone(),
        }
    }
}
