# Scalarized HVP Design

## Goal

Extend `tensor-ad-oracles` with second-order oracle data for PyTorch families that
explicitly support forward-over-backward AD, while preserving the current
first-order database contract and keeping the stored data compact and
gauge-aware.

## Context

The current database stores, per success probe:

- `direction = v`
- `cotangent = bar_y`
- `pytorch_ref.jvp`
- `pytorch_ref.vjp`
- `fd_ref.jvp`

and validates:

- `Jv_torch ~= Jv_fd`
- `<bar_y, Jv_fd> ~= <J*bar_y_torch, v>`

This is sufficient for first-order derivative correctness, but it does not
cover second-order behavior. PyTorch already classifies many linalg families
with `supports_fwgrad_bwgrad=True`, which provides a natural upstream contract
for second-order coverage.

## Approaches Considered

### 1. Full second-order observable data

Store tensor-valued second-order directional outputs for the observable itself.

Pros:

- conceptually direct for tensor-valued outputs

Cons:

- schema becomes much larger and less clear
- for spectral ops, second-order output tensors are more exposed to gauge and
  representation ambiguity
- difficult to validate independently with finite differences

### 2. Scalarized HVP

Define

`phi(x) = <bar_y, observable(x)>`

and store

`H_phi(x) v`

for the same paired probe `(v, bar_y)`.

Pros:

- works uniformly for scalar and tensor-valued outputs
- reuses the existing probe structure
- keeps spectral ops within the same observable/gauge discipline already used
  for first-order data
- finite-difference validation is straightforward

Cons:

- does not expose every possible second-order output representation

### 3. PyTorch-only HVP

Store only the PyTorch HVP and verify it live in CI.

Pros:

- simplest to generate

Cons:

- weak oracle independence
- fails the repository principle that success-case references should not depend
  on a single source

## Recommendation

Use **scalarized HVP** and require **both PyTorch and finite-difference HVP**
for every family that is explicitly marked upstream with
`supports_fwgrad_bwgrad=True`, except for upstream-explicit xfail or unsupported
families.

This keeps the schema coherent with the current first-order model and gives a
meaningful second-order contract that remains independent of one oracle source.

## Data Model

For each existing success probe, add optional second-order fields:

- `pytorch_ref.hvp`
- `fd_ref.hvp`

Each `hvp` is an input-space tensor map with the same keys and shapes as the
current `vjp` map.

The mathematical definition is:

- `phi(x) = <cotangent, observable(x)>`
- `hvp = H_phi(x) v`

This means:

- for scalar-output ops, scalarized HVP is the ordinary HVP
- for tensor-output ops, `cotangent` chooses the scalar functional that is
  differentiated twice

## Oracle Construction

### PyTorch oracle

Construct:

- `phi(inputs)` from the current probe cotangent and current observable
- `grad_phi(inputs)` using `torch.func.grad`
- `H_phi(x) v` using `torch.func.jvp(grad_phi, inputs, direction)`

For complex outputs, `phi` must be real-valued. The existing real-part inner
product rule used for adjoint validation should be reused.

### Finite-difference oracle

Construct:

- `grad_phi(x + h v)`
- `grad_phi(x - h v)`

and compute:

- `fd_hvp = (grad_phi(x + h v) - grad_phi(x - h v)) / (2 h)`

The finite-difference step should initially reuse the existing dtype-aware step
policy, but second-order tolerances should be tracked separately from
first-order tolerances.

## Scope Rules

HVP data should only be published for families that meet all of these:

- `supports_fwgrad_bwgrad=True` upstream
- first-order success case already materializes
- no upstream-explicit xfail for `fwgrad_bwgrad`
- no local known instability class requiring omission

Expected-error spectral families remain first-order/error-only and do not gain
HVP data.

## Tolerance Policy

Second-order tolerances should not reuse first-order thresholds.

The schema should split comparison policy into:

- `comparison.first_order`
- `comparison.second_order`

The second-order values should be derived from measured maxima of:

- `max_abs_diff(pytorch_hvp, fd_hvp)`
- `max_rel_diff(pytorch_hvp, fd_hvp)`

using the same power-of-ten rounding and safety-factor policy already used for
the current tolerance audit.

## Replay And CI

Replay validation should extend success-case checks with:

- stored `pytorch_ref.hvp` reproducibility
- stored `fd_ref.hvp` reproducibility
- live `pytorch_ref.hvp ~= fd_ref.hvp` within the second-order tolerance

CI should continue to separate concerns:

- `check_replay.py` validates live first- and second-order oracle agreement
- `check_regeneration.py` validates regenerated case content, while still
  treating tolerance proposal fields as environment-sensitive
- `check_tolerances.py` should grow a second-order audit pass rather than
  assuming only JVP/VJP residuals exist

## Testing Strategy

Add second-order coverage in this order:

1. schema-only unit tests for the new `hvp` fields and split comparison blocks
2. pure helper tests for scalarized HVP construction and FD HVP computation
3. generator tests on a small stable family such as `solve/identity`
4. replay tests for one HVP-enabled family
5. full DB regeneration after the end-to-end path is stable

## Non-Goals

This design does not attempt to:

- store full tensor-valued second-order observable data
- add HVP to families without upstream `supports_fwgrad_bwgrad=True`
- solve every numerically unstable second-order family in one step
