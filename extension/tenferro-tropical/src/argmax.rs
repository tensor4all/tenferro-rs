//! Argmax tracking for tropical backward pass (automatic differentiation).
//!
//! In tropical semirings, the "gradient" of a contraction is determined by
//! which elements "won" the max (or min) comparisons. [`ArgmaxTracker`]
//! records the winner indices during the forward pass so that the backward
//! pass can route gradients to the correct elements.
//!
//! This is the tropical analogue of storing intermediate activations for
//! backpropagation in standard neural network training.

/// Tracks winner indices from tropical forward-pass operations.
///
/// During a tropical contraction `C[i,j] = max_k (A[i,k] + B[k,j])`,
/// the tracker records which `k` achieved the maximum for each `(i,j)`.
/// The backward pass uses these indices to route gradients.
///
/// # Examples
///
/// ```
/// use tenferro_ext_tropical::ArgmaxTracker;
///
/// // Create a tracker for a 3x5 output
/// let tracker = ArgmaxTracker::new(&[3, 5]);
///
/// // After forward pass, query the winner index for output element (1, 2)
/// let k_winner = tracker.winner_index(&[1, 2]);
/// assert_eq!(k_winner, 0); // initialized to 0
/// ```
pub struct ArgmaxTracker {
    /// Shape of the output tensor.
    output_shape: Vec<usize>,
    /// Winner indices (flat storage, one per output element).
    /// Each entry records the contraction index that achieved the optimum.
    indices: Vec<usize>,
}

impl ArgmaxTracker {
    /// Create a new tracker for an output of the given shape.
    ///
    /// All winner indices are initialized to 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ext_tropical::ArgmaxTracker;
    ///
    /// let tracker = ArgmaxTracker::new(&[3, 5]);
    /// assert_eq!(tracker.output_shape(), &[3, 5]);
    /// ```
    pub fn new(output_shape: &[usize]) -> Self {
        let total: usize = output_shape.iter().product();
        Self {
            output_shape: output_shape.to_vec(),
            indices: vec![0; total],
        }
    }

    /// Return the output shape.
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    /// Return the winner indices as a flat slice.
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Return a mutable reference to the winner indices.
    pub fn indices_mut(&mut self) -> &mut [usize] {
        &mut self.indices
    }

    /// Look up the winner index for a given multi-dimensional output position.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ext_tropical::ArgmaxTracker;
    ///
    /// let tracker = ArgmaxTracker::new(&[3, 5]);
    /// let k = tracker.winner_index(&[1, 2]);
    /// assert_eq!(k, 0); // initialized to 0
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `position` has the wrong rank or any index is out of bounds
    /// for the tracked output shape.
    pub fn winner_index(&self, position: &[usize]) -> usize {
        assert_eq!(
            position.len(),
            self.output_shape.len(),
            "winner_index: expected {} indices, got {}",
            self.output_shape.len(),
            position.len()
        );

        // Column-major linear index
        let mut linear = 0;
        let mut stride = 1;
        for (i, &p) in position.iter().enumerate() {
            assert!(
                p < self.output_shape[i],
                "winner_index: index {} out of bounds for axis {} with size {}",
                p,
                i,
                self.output_shape[i]
            );
            linear += p * stride;
            stride *= self.output_shape[i];
        }
        self.indices[linear]
    }
}
