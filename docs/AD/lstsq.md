# Least Squares AD Notes (`lstsq`)

## Public contract

`lstsq(a, b)` returns

- `solution = pinv(a) @ b`
- `residuals = ||a @ solution - b||_F^2` per right-hand side when `m > n` and the solve is full-rank
- `residuals = []` otherwise

The auxiliary metadata `rank` and `singular_values` are not differentiated.

## First-order source of truth

The first-order rules are expressed in terms of the pseudoinverse:

### JVP for the solution

For `x = pinv(A) b`,

$$
dx = d(pinv(A))\, b + pinv(A)\, db
$$

In the implementation, `d(pinv(A))` is provided by `pinv_frule`.

### JVP for the residual summaries

Let

$$
r = A x - b, \qquad dr = dA\,x - db
$$

Then, for each right-hand side,

$$
d\,\mathrm{residuals} = 2 \sum \mathrm{Re}(r \odot \overline{dr})
$$

For the current real-valued `lstsq` AD path, this reduces to

$$
d\,\mathrm{residuals} = 2 \sum r \odot dr
$$

### VJP for the solution

Let `gx` be the cotangent of `solution`. Since `x = pinv(A) b`,

- the cotangent for `pinv(A)` is `gx @ b^H`
- the cotangent for `b` from this path is `pinv(A)^H @ gx`
- the cotangent for `A` from this path is given by `pinv_rrule`

This is the path used by the implementation.

### VJP for the residual summaries

Let `gr` be the cotangent of the summary residual outputs, broadcast per RHS. Then

$$
\bar{A}_{res} = 2 (gr \odot r)\, x^H
$$

$$
\bar{b}_{res} = -2 (gr \odot r)
$$

The full VJP is the sum of the solution-path and residual-summary-path contributions.

## Verification policy

- `frule/rrule` are the semantic source of truth
- oracle replay must exist before the rule is considered mainline
- `LinearizedOp::jvp/vjp` is expected to be a thin adapter over these rules
