# Dense Scalar OpInfo Design

## Goal

Expand `tensor-ad-oracles` beyond linalg so it can publish dense scalar/tensor
scalarizable operation families drawn from generic PyTorch `OpInfo`
definitions, while preserving the current oracle contract and replay model.

## Scope

This expansion targets dense CPU families only:

- elementwise unary ops
- elementwise binary ops
- dense reduction ops
- dense `special.*` families represented through generic PyTorch `OpInfo`

Included dtypes:

- `float32`
- `float64`
- `complex64`
- `complex128`

Second-order data policy:

- publish first-order data for all four dtypes
- publish scalarized HVP only for families with
  `supports_fwgrad_bwgrad=True` and only when the family/dtype combination is
  numerically stable enough to admit

Explicit non-goals for this phase:

- sparse families
- nested families
- masked families
- device-specific/backend-specific variants
- non-dense layout-specific cases

## Why This Is Feasible

The current database model is already generic enough for dense scalarizable
ops:

- materialized `inputs`
- `observable`
- paired probes
- `pytorch_ref`
- `fd_ref`
- optional scalarized HVP

The main missing piece is machine-readable call metadata for
non-differentiable scalar/integer/bool parameters such as `dim`, `keepdim`,
`ord`, `dtype`, scalar exponents, and similar kwargs.

## Data Model Changes

Keep the existing case structure and add call metadata:

- `op_args`
- `op_kwargs`

Rule:

- differentiable tensor arguments live in `inputs`
- non-differentiable parameters live in `op_args` / `op_kwargs`

This allows generic PyTorch `OpInfo` families to be reconstructed without
forking the schema into a separate scalar-op format.

The `observable` field remains the differentiation target:

- elementwise ops mostly use `identity`
- reductions also usually use `identity`
- any representation-sensitive family can still use a processed observable if
  needed

## Oracle Contract

Every success case still requires:

- `pytorch_ref`
- `fd_ref`

First-order contract remains:

- `Jv_torch ~= Jv_fd`
- `<bar_y, Jv_fd> ~= <J*bar_y_torch, v>`

Second-order contract remains scalarized:

- `phi(x) = <bar_y, observable(x)>`
- publish `H_phi(x) v`
- require `pytorch_ref.hvp ~= fd_ref.hvp`

## Tolerance Policy

Do not reuse PyTorch tolerances as canonical DB tolerances.

Instead:

- use measured `torch` vs FD residuals to derive DB tolerances
- keep the existing “tighten if current tolerance is more than ten orders of
  magnitude looser than observed residual” policy
- retain PyTorch AD tolerances as an upper safety bound for cross-oracle
  disagreement

This means:

- DB tolerances remain empirically tight
- cross-oracle disagreement does not exceed what upstream PyTorch itself
  tolerates in AD tests

## Inventory Strategy

Add a new generic upstream inventory alongside the current linalg-specific one.

The generic inventory should normalize dense AD-relevant generic `OpInfo`
entries and record:

- `name`
- `variant_name`
- `sample_inputs_func`
- `supports_forward_ad`
- `supports_fwgrad_bwgrad`
- `gradcheck_fast_mode`
- `gradcheck_wrapper`
- `output_process_fn_grad`
- explicit tolerance overrides
- family class (`unary`, `binary`, `reduction`, generic dense op)

Then a separate family-mapping layer translates upstream entries into DB
families while preserving upstream naming as much as possible.

## Runtime Strategy

The runtime should generalize from linalg-specific call reconstruction to a
generic call-spec executor:

- rebuild differentiable tensor inputs from `inputs`
- rebuild non-differentiable `op_args` / `op_kwargs`
- call the pinned upstream operator
- apply the selected `observable`
- generate paired probes and oracles exactly as today

This keeps the replay validator and generator aligned on one execution path.

## Testing Strategy

Implement in this order:

1. inventory normalization tests
2. schema tests for `op_args` / `op_kwargs`
3. generic runtime invocation tests
4. representative unary/binary/reduction replay tests
5. global regeneration, replay, and tolerance audit

## Recommendation

Implement this as one feature branch and one generator expansion, but do it
inventory-first and mechanically from upstream `OpInfo`. Do not hand-curate
hundreds of scalar families one by one.
