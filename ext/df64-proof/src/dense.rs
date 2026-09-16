//! Dense matrix helpers in the external scalar.
//!
//! The factorization and its derivatives are the contribution's own numerical
//! bodies, so they need the small amount of dense linear algebra that the reverse
//! and forward rules of a factorization require: products, transposes, triangular
//! extraction, and a triangular solve. Every helper is column-major and uses the
//! external scalar's own arithmetic, so nothing here narrows to `f64` on the way.

use crate::Df64;

/// A dense column-major matrix with its row count.
pub(crate) struct Matrix {
    pub(crate) rows: usize,
    pub(crate) data: Vec<Df64>,
}

impl Matrix {
    pub(crate) fn new(rows: usize, data: Vec<Df64>) -> Self {
        Self { rows, data }
    }

    pub(crate) fn columns(&self) -> usize {
        self.data.len().checked_div(self.rows).unwrap_or(0)
    }

    pub(crate) fn at(&self, row: usize, column: usize) -> Df64 {
        self.data[row + column * self.rows]
    }
}

/// Product `a b` of two column-major matrices.
pub(crate) fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
    let columns = b.columns();
    let mut data = vec![Df64::zero(); a.rows * columns];
    for column in 0..columns {
        for row in 0..a.rows {
            let mut sum = Df64::zero();
            for inner in 0..b.rows {
                sum = sum + a.at(row, inner) * b.at(inner, column);
            }
            data[row + column * a.rows] = sum;
        }
    }
    Matrix::new(a.rows, data)
}

/// Transpose of a column-major matrix.
pub(crate) fn transpose(a: &Matrix) -> Matrix {
    let columns = a.columns();
    let mut data = vec![Df64::zero(); columns * a.rows];
    for column in 0..columns {
        for row in 0..a.rows {
            data[column + row * columns] = a.at(row, column);
        }
    }
    Matrix::new(columns, data)
}

/// Elementwise difference `a - b`.
pub(crate) fn subtract(a: &Matrix, b: &Matrix) -> Matrix {
    let data = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(left, right)| *left - *right)
        .collect();
    Matrix::new(a.rows, data)
}

/// Elementwise sum `a + b`.
pub(crate) fn add(a: &Matrix, b: &Matrix) -> Matrix {
    let data = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(left, right)| *left + *right)
        .collect();
    Matrix::new(a.rows, data)
}

/// Upper triangle including the diagonal, with zeros elsewhere.
pub(crate) fn upper_triangle(a: &Matrix) -> Matrix {
    let columns = a.columns();
    let mut data = vec![Df64::zero(); a.data.len()];
    for column in 0..columns {
        for row in 0..=column.min(a.rows.saturating_sub(1)) {
            data[row + column * a.rows] = a.at(row, column);
        }
    }
    Matrix::new(a.rows, data)
}

/// Lower triangle including the diagonal.
pub(crate) fn lower_triangle(a: &Matrix) -> Matrix {
    let columns = a.columns();
    let mut data = vec![Df64::zero(); a.data.len()];
    for column in 0..columns {
        for row in column..a.rows {
            data[row + column * a.rows] = a.at(row, column);
        }
    }
    Matrix::new(a.rows, data)
}

/// Lower triangle strictly below the diagonal.
pub(crate) fn strictly_lower_triangle(a: &Matrix) -> Matrix {
    let columns = a.columns();
    let mut data = vec![Df64::zero(); a.data.len()];
    for column in 0..columns {
        for row in (column + 1)..a.rows {
            data[row + column * a.rows] = a.at(row, column);
        }
    }
    Matrix::new(a.rows, data)
}

/// Solve `x r = b` for `x`, where `r` is upper triangular and square.
///
/// # Errors
///
/// Returns `None` under the same conditions as [`solve_upper`].
pub(crate) fn solve_upper_from_the_right(r: &Matrix, b: &Matrix) -> Option<Matrix> {
    let order = r.rows;
    if r.columns() != order || b.columns() != order {
        return None;
    }
    // Column j of the solution depends only on the earlier columns, because R is
    // upper triangular, so the columns are produced left to right.
    let rows = b.rows;
    let mut data = vec![Df64::zero(); rows * order];
    for column in 0..order {
        let diagonal = r.at(column, column);
        if diagonal.hi == 0.0 {
            return None;
        }
        for row in 0..rows {
            let mut sum = b.at(row, column);
            for inner in 0..column {
                sum = sum - data[row + inner * rows] * r.at(inner, column);
            }
            data[row + column * rows] = sum.ratio(diagonal);
        }
    }
    Some(Matrix::new(rows, data))
}

/// A zero matrix of the given shape.
pub(crate) fn zeros(rows: usize, columns: usize) -> Matrix {
    Matrix::new(rows, vec![Df64::zero(); rows * columns])
}
