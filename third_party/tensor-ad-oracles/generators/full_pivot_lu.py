"""Local full-pivot LU reference helpers.

PyTorch exposes partial-pivot LU but no full-pivot LU operator. These helpers
therefore separate the discrete full-pivot search from the differentiable
fixed-permutation LU map used by the oracle generator.
"""

from __future__ import annotations


SAMPLE_SHAPES: tuple[tuple[int, int], ...] = ((3, 3), (2, 4), (4, 2))


def make_sample_matrix(torch, *, dtype, shape: tuple[int, int]):
    """Return a deterministic full-rank matrix with stable full-pivot choices."""
    values = {
        (3, 3): [
            [1.0, -2.0, 3.0],
            [4.0, 11.0, -6.0],
            [7.0, -8.0, 31.0],
        ],
        (2, 4): [
            [1.0, -2.0, 9.0, 3.0],
            [4.0, 23.0, -6.0, 8.0],
        ],
        (4, 2): [
            [1.0, -2.0],
            [9.0, 4.0],
            [3.0, 29.0],
            [5.0, -6.0],
        ],
    }[shape]
    real = torch.tensor(values, dtype=torch.float64, device="cpu")
    if dtype.is_complex:
        imag = torch.flip(real, dims=(0, 1)) * 0.03
        return (real + 1j * imag).to(dtype=dtype)
    return real.to(dtype=dtype)


def _swap_items(values: list[int], left: int, right: int) -> None:
    values[left], values[right] = values[right], values[left]


def _swap_rows(tensor, left: int, right: int) -> None:
    if left == right:
        return
    replacement = tensor[[right, left], :].clone()
    tensor[[left, right], :] = replacement


def _swap_cols(tensor, left: int, right: int) -> None:
    if left == right:
        return
    replacement = tensor[:, [right, left]].clone()
    tensor[:, [left, right]] = replacement


def full_pivot_metadata(torch, a) -> dict[str, object]:
    """Compute full-pivot row/column metadata from a detached matrix."""
    work = a.detach().clone()
    rows, cols = work.shape
    rank_extent = min(rows, cols)
    row_perm = list(range(rows))
    col_perm = list(range(cols))
    swap_count = 0

    for index in range(rank_extent):
        submatrix_abs = work[index:, index:].abs()
        flat_pivot = int(torch.argmax(submatrix_abs).item())
        pivot_col_count = cols - index
        pivot_row = index + flat_pivot // pivot_col_count
        pivot_col = index + flat_pivot % pivot_col_count

        if pivot_row != index:
            _swap_rows(work, index, pivot_row)
            _swap_items(row_perm, index, pivot_row)
            swap_count += 1
        if pivot_col != index:
            _swap_cols(work, index, pivot_col)
            _swap_items(col_perm, index, pivot_col)
            swap_count += 1

        pivot = work[index, index]
        if float(pivot.abs().item()) == 0.0:
            raise ValueError("full-pivot LU sample is rank deficient")
        if index + 1 < rows and index + 1 < cols:
            multipliers = work[index + 1 :, index] / pivot
            work[index + 1 :, index] = multipliers
            work[index + 1 :, index + 1 :] = (
                work[index + 1 :, index + 1 :]
                - multipliers[:, None] * work[index, index + 1 :]
            )

    return {
        "row_perm": row_perm,
        "col_perm": col_perm,
        "parity": -1 if swap_count % 2 else 1,
        "status": "success",
    }


def fixed_permutation_lu(torch, a, *, row_perm: list[int], col_perm: list[int]):
    """Return differentiable `(L, U)` for fixed full-pivot permutations."""
    permuted = a[row_perm, :][:, col_perm]
    rows, cols = permuted.shape
    rank_extent = min(rows, cols)
    l_columns = []
    u_rows = []

    for index in range(rank_extent):
        if l_columns:
            l_previous = torch.stack(l_columns, dim=1)
            u_previous = torch.stack(u_rows, dim=0)
            residual = permuted - l_previous @ u_previous
        else:
            residual = permuted

        u_row = torch.cat(
            [
                permuted.new_zeros(index),
                residual[index, index:],
            ],
            dim=0,
        )
        pivot = residual[index, index]
        l_column = torch.cat(
            [
                permuted.new_zeros(index),
                permuted.new_ones(1),
                residual[index + 1 :, index] / pivot,
            ],
            dim=0,
        )
        l_columns.append(l_column)
        u_rows.append(u_row)

    return {
        "l": torch.stack(l_columns, dim=1),
        "u": torch.stack(u_rows, dim=0),
    }


def fixed_permutation_lu_tuple(torch, a, *, row_perm: list[int], col_perm: list[int]):
    """Return `(L, U)` in the stable output order used by probe generation."""
    output = fixed_permutation_lu(
        torch,
        a,
        row_perm=row_perm,
        col_perm=col_perm,
    )
    return output["l"], output["u"]
