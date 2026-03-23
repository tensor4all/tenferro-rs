use super::*;

fn view_as_real_strides(strides: &[isize]) -> Result<Vec<isize>> {
    let mut output = Vec::with_capacity(strides.len() + 1);
    for (axis, &stride) in strides.iter().enumerate() {
        output.push(stride.checked_mul(2).ok_or_else(|| {
            Error::StrideError(format!(
                "view_as_real: stride overflow for axis {axis} with stride {stride}"
            ))
        })?);
    }
    output.push(1);
    Ok(output)
}

fn view_as_complex_strides(dims: &[usize], strides: &[isize]) -> Result<Vec<isize>> {
    if dims.is_empty() || strides.is_empty() {
        return Err(Error::InvalidArgument(
            "view_as_complex requires at least one dimension".into(),
        ));
    }
    if dims.len() != strides.len() {
        return Err(Error::InvalidArgument(format!(
            "view_as_complex rank mismatch: dims={} strides={}",
            dims.len(),
            strides.len()
        )));
    }

    let mut output = Vec::with_capacity(strides.len().saturating_sub(1));
    for (axis, (&dim, &stride)) in dims[..dims.len() - 1]
        .iter()
        .zip(&strides[..strides.len() - 1])
        .enumerate()
    {
        if dim != 1 && stride % 2 != 0 {
            return Err(Error::InvalidArgument(format!(
                "view_as_complex requires even strides on all leading dimensions, got stride {stride} at axis {axis}"
            )));
        }
        output.push(stride / 2);
    }
    Ok(output)
}

impl Tensor<Complex32> {
    /// Return a zero-copy real view of a complex tensor.
    ///
    /// The last logical axis is expanded to length `2`, exposing the real and
    /// imaginary parts as adjacent real-valued elements.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use num_complex::Complex32;
    /// use tenferro_device::LogicalMemorySpace;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let z = Tensor::<Complex32>::from_slice(
    ///     &[Complex32::new(1.0, 2.0)],
    ///     &[1],
    ///     MemoryOrder::ColumnMajor,
    /// ).unwrap();
    /// let r = z.view_as_real().unwrap();
    /// assert_eq!(r.dims(), &[1, 2]);
    /// assert_eq!(r.logical_memory_space(), LogicalMemorySpace::MainMemory);
    /// ```
    pub fn view_as_real(&self) -> Result<Tensor<f32>> {
        self.wait();
        if self.is_conjugated() {
            return Err(Error::InvalidArgument(
                "view_as_real requires a resolved complex tensor".into(),
            ));
        }

        let mut dims = self.dims().to_vec();
        dims.push(2);
        let strides = view_as_real_strides(self.strides())?;
        let offset = self.offset().checked_mul(2).ok_or_else(|| {
            Error::StrideError("view_as_real: offset overflow when reinterpreting tensor".into())
        })?;
        let buffer_len =
            self.buffer().len().checked_mul(2).ok_or_else(|| {
                Error::StrideError("view_as_real: storage length overflow".into())
            })?;
        let buffer = self.buffer().reinterpret_as::<f32>(buffer_len)?;
        validate_layout_against_len(&dims, &strides, offset, buffer.len())?;
        Ok(Tensor::from_parts(
            buffer,
            Arc::from(dims),
            Arc::from(strides),
            offset,
            self.logical_memory_space(),
            self.preferred_compute_device(),
            None,
            false,
            None,
        ))
    }
}

impl Tensor<Complex64> {
    /// Return a zero-copy real view of a complex tensor.
    ///
    /// The last logical axis is expanded to length `2`, exposing the real and
    /// imaginary parts as adjacent real-valued elements.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use num_complex::Complex64;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let z = Tensor::<Complex64>::from_slice(
    ///     &[Complex64::new(1.0, 2.0)],
    ///     &[1],
    ///     MemoryOrder::ColumnMajor,
    /// ).unwrap();
    /// let r = z.view_as_real().unwrap();
    /// assert_eq!(r.dims(), &[1, 2]);
    /// ```
    pub fn view_as_real(&self) -> Result<Tensor<f64>> {
        self.wait();
        if self.is_conjugated() {
            return Err(Error::InvalidArgument(
                "view_as_real requires a resolved complex tensor".into(),
            ));
        }

        let mut dims = self.dims().to_vec();
        dims.push(2);
        let strides = view_as_real_strides(self.strides())?;
        let offset = self.offset().checked_mul(2).ok_or_else(|| {
            Error::StrideError("view_as_real: offset overflow when reinterpreting tensor".into())
        })?;
        let buffer_len =
            self.buffer().len().checked_mul(2).ok_or_else(|| {
                Error::StrideError("view_as_real: storage length overflow".into())
            })?;
        let buffer = self.buffer().reinterpret_as::<f64>(buffer_len)?;
        validate_layout_against_len(&dims, &strides, offset, buffer.len())?;
        Ok(Tensor::from_parts(
            buffer,
            Arc::from(dims),
            Arc::from(strides),
            offset,
            self.logical_memory_space(),
            self.preferred_compute_device(),
            None,
            false,
            None,
        ))
    }
}

impl Tensor<f32> {
    /// Return a zero-copy complex view of a real tensor whose last dimension
    /// stores paired real and imaginary components.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let r = Tensor::<f32>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let z = r.view_as_complex().unwrap();
    /// assert_eq!(z.dims(), &[] as &[usize]);
    /// ```
    pub fn view_as_complex(&self) -> Result<Tensor<Complex32>> {
        self.wait();
        if self.ndim() == 0 {
            return Err(Error::InvalidArgument(
                "view_as_complex requires at least one dimension".into(),
            ));
        }
        if self.dims().last().copied() != Some(2) {
            return Err(Error::InvalidArgument(
                "view_as_complex requires the last dimension to have size 2".into(),
            ));
        }
        if self.strides().last().copied() != Some(1) {
            return Err(Error::InvalidArgument(
                "view_as_complex requires the last stride to be 1".into(),
            ));
        }
        if self.offset() % 2 != 0 {
            return Err(Error::InvalidArgument(
                "view_as_complex requires an even element offset".into(),
            ));
        }

        let dims = self.dims()[..self.ndim() - 1].to_vec();
        let strides = view_as_complex_strides(self.dims(), self.strides())?;
        let offset = self.offset() / 2;
        let source_len = self.buffer().len() / 2;
        let buffer = self.buffer().reinterpret_as::<Complex32>(source_len)?;
        validate_layout_against_len(&dims, &strides, offset, buffer.len())?;
        Ok(Tensor::from_parts(
            buffer,
            Arc::from(dims),
            Arc::from(strides),
            offset,
            self.logical_memory_space(),
            self.preferred_compute_device(),
            None,
            false,
            None,
        ))
    }
}

impl Tensor<f64> {
    /// Return a zero-copy complex view of a real tensor whose last dimension
    /// stores paired real and imaginary components.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let r = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let z = r.view_as_complex().unwrap();
    /// assert_eq!(z.dims(), &[] as &[usize]);
    /// ```
    pub fn view_as_complex(&self) -> Result<Tensor<Complex64>> {
        self.wait();
        if self.ndim() == 0 {
            return Err(Error::InvalidArgument(
                "view_as_complex requires at least one dimension".into(),
            ));
        }
        if self.dims().last().copied() != Some(2) {
            return Err(Error::InvalidArgument(
                "view_as_complex requires the last dimension to have size 2".into(),
            ));
        }
        if self.strides().last().copied() != Some(1) {
            return Err(Error::InvalidArgument(
                "view_as_complex requires the last stride to be 1".into(),
            ));
        }
        if self.offset() % 2 != 0 {
            return Err(Error::InvalidArgument(
                "view_as_complex requires an even element offset".into(),
            ));
        }

        let dims = self.dims()[..self.ndim() - 1].to_vec();
        let strides = view_as_complex_strides(self.dims(), self.strides())?;
        let offset = self.offset() / 2;
        let source_len = self.buffer().len() / 2;
        let buffer = self.buffer().reinterpret_as::<Complex64>(source_len)?;
        validate_layout_against_len(&dims, &strides, offset, buffer.len())?;
        Ok(Tensor::from_parts(
            buffer,
            Arc::from(dims),
            Arc::from(strides),
            offset,
            self.logical_memory_space(),
            self.preferred_compute_device(),
            None,
            false,
            None,
        ))
    }
}
