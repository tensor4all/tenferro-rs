# Extension Tutorials And Sample CI Design

## Context

tenferro currently has runnable tutorial binaries under `docs/tutorial-code`, a
longer KdV PINN sample in the root workspace as `kdv_pinn`, and an existing
standalone extension crate under `ext/tropical`. The tutorial index does not yet
teach sparse tensors as an extension example, and `ext/tropical` is not surfaced
as one of the ordered tutorials.

The desired split is:

- KdV remains a tutorial/sample, but CI only compile-checks it because the full
  training run is intentionally long.
- Tropical and sparse extension examples are tutorial material whose tests run
  in CI.
- Sparse tensor support is demonstrated as an extension crate, not added to the
  core dense tensor model.

## Goals

1. Treat tropical as a first-class tutorial by adding a tutorial page that points
   at the existing `ext/tropical` extension crate and its runnable tests.
2. Add a sparse tensor extension tutorial that demonstrates sparse-sparse
   contraction and AD through tenferro's extension mechanism.
3. Move the long-running KdV PINN sample out of the root workspace into a sample
   location and compile-check it in CI without executing training.
4. Update CI so runnable extension tutorials execute, while KdV is compile-only.
5. Keep public documentation aligned with the actual public surface and avoid
   claiming a general sparse tensor subsystem.

## Non-Goals

- Do not add sparse tensor variants to `tenferro-tensor::Tensor`.
- Do not make sparse tensors a standard operation family in the core crates.
- Do not implement a production sparse optimizer, sparse GPU backend, or broad
  sparse format suite.
- Do not run KdV training in CI.
- Do not move `ext/tropical` into `docs/tutorial-code`; it should remain an
  extension crate example.

## Repository Layout

The implementation should use this layout:

```text
ext/
  tropical/                 # existing standalone extension crate
  sparse/                   # new standalone sparse extension tutorial crate
samples/
  kdv-pinn/                 # moved from root kdv_pinn
docs/
  tutorials/
    tropical-extension.md
    sparse-extension.md
    kdv-pinn.md             # updated paths and compile-only note
```

The root `Cargo.toml` should no longer list KdV as a workspace member. The KdV
sample should become its own standalone workspace package with explicit package
metadata and path dependencies pointing back to `../../crates/...`.

`ext/sparse` should also be a standalone workspace package. This keeps extension
examples independent from the root workspace while allowing CI to run them with
manifest-path commands.

## Sparse Extension Design

The sparse tutorial should implement a deliberately small COO-style sparse
matrix/tensor wrapper. Its job is to teach extension mechanics, not to become a
general sparse library.

The public wrapper should separate sparse metadata from differentiable values:

- `SparseCooTensor` for eager/concrete values.
- `SparseCooTracedTensor` for traced values.
- A dense tenferro `Tensor` payload for integer coordinates.
- A tenferro `Tensor` or `TracedTensor` for nonzero values.
- Shape metadata stored as ordinary Rust `Vec<usize>`.

The tutorial may store sparse payload metadata as dense tenferro tensors. For a
COO matrix, the coordinate tensor should be a small integer dense tensor with
shape `[rank, nnz]`, where each column is one logical sparse coordinate. Values
should be a dense tensor with shape `[nnz]`.

The extension operation should operate on values as graph inputs while carrying
the fixed sparse structure in the operation payload:

- input 0: left nonzero values
- input 1: right nonzero values
- payload: left coordinates, right coordinates, left shape, right shape,
  contraction axes, and derived output coordinates
- output 0: output nonzero values with shape `[output_nnz]`

This design keeps tensor-valued differentiable data as operation inputs and uses
payload tensors only for fixed sparse structure. The output sparsity pattern is
computed when the sparse operation is constructed, so graph shape inference can
return a fixed output value length.

The tutorial contraction should support a simple deterministic contract, such as
matrix multiplication:

```text
C[i, k] = sum_j A[i, j] * B[j, k]
```

The implementation should expose this as `sparse_matmul`. A more general
axis-labelled `contract` helper is out of scope for this pass.

## Sparse AD Design

The sparse contraction is bilinear in the values:

```text
dC = contract(dA, B) + contract(A, dB)
```

The extension crate should register AD rules through `ExtensionRuleSet`.

The JVP rule should emit a sparse-contract JVP extension op that accepts the
primal values plus any active tangent values and returns one output tangent
value tensor.

The VJP rule should emit sparse-contract VJP extension ops for active inputs:

- gradient with respect to left values uses the output cotangent values and the
  right primal values under the fixed sparse contract plan.
- gradient with respect to right values uses the left primal values and the
  output cotangent values under the same plan.

The tests should prove both rules on a small matrix example:

- forward sparse contraction equals a dense reference.
- traced execution through `GraphExecutor` returns the expected nonzero values.
- gradient of `sum(sparse_matmul(a, b).values)` with respect to left and right
  values equals the exact dense-reference gradient for the fixed sparse pattern.

## Tropical Tutorial Design

`ext/tropical` should remain an extension crate example. Add a tutorial page
that explains:

- tropical is an out-of-tree extension crate, not a core tenferro operation;
- the crate defines domain-specific algebra and extension runtime registration;
- users run it with `cargo test --manifest-path ext/tropical/Cargo.toml`;
- AD examples require the crate's `autodiff` feature.

CI should run tropical tests as executable tutorial coverage. The implementation
should prefer a command that covers the tutorial's AD path, such as:

```bash
cargo test --manifest-path ext/tropical/Cargo.toml --release --features autodiff
```

If a default-feature command is also needed for non-AD coverage, run it as a
second explicit command rather than relying on root `--workspace`.

## KdV Sample Design

Move `kdv_pinn` to `samples/kdv-pinn` and update documentation paths and
commands. The tutorial page remains under `docs/tutorials/kdv-pinn.md`, but it
should state that CI compile-checks the sample and does not run full training.

The documented run command should become:

```bash
cargo run --manifest-path samples/kdv-pinn/Cargo.toml --release
```

The documented fast compile check should become:

```bash
cargo check --manifest-path samples/kdv-pinn/Cargo.toml --release --all-targets
```

The sample's package name should remain `kdv_pinn` to avoid unnecessary internal
renaming. Its file paths in the tutorial structure table should use
`samples/kdv-pinn/src/...`.

## CI Design

The PR workspace test job should continue running the root workspace:

```bash
cargo nextest run --workspace --release --no-fail-fast
cargo test --doc --workspace --release
```

Because KdV is no longer a root workspace member, these commands will not run
KdV tests or training.

Add explicit sample/tutorial commands:

```bash
cargo check --manifest-path samples/kdv-pinn/Cargo.toml --release --all-targets
cargo test --manifest-path ext/tropical/Cargo.toml --release --features autodiff
cargo test --manifest-path ext/sparse/Cargo.toml --release --features autodiff
```

For same-repo PRs, run the explicit sample/tutorial commands once per existing
backend matrix entry. The examples do not need a new matrix-independent job in
this change.

## Documentation Updates

Update:

- `docs/tutorials/index.md` to include tropical and sparse extension tutorials.
- `docs/_quarto.yml` sidebar to include both pages.
- `docs/guides/custom-operations.md` to point to the tropical and sparse
  tutorials as worked extension examples.
- `docs/tutorials/kdv-pinn.md` to reflect the new path and compile-only CI
  treatment.
- Root `README.md` only if it mentions KdV, tropical, or the tutorial list in a
  way that would become stale.

Documentation should avoid saying that tenferro has general sparse tensor
support. The precise claim is that sparse tensors can be implemented as an
extension crate using dense tenferro tensors for fixed sparse metadata and
nonzero-value payloads.

## Verification

Targeted verification for the implementation should include:

```bash
cargo fmt --all --check
cargo nextest run --workspace --release --no-fail-fast
cargo test --doc --workspace --release
cargo check --manifest-path samples/kdv-pinn/Cargo.toml --release --all-targets
cargo test --manifest-path ext/tropical/Cargo.toml --release --features autodiff
cargo test --manifest-path ext/sparse/Cargo.toml --release --features autodiff
python3 scripts/check-docs-site.py
```

If `cargo nextest` is unavailable locally, use:

```bash
cargo test --workspace --release --no-fail-fast
```

## Risks

- A sparse tutorial can accidentally look like a supported sparse subsystem.
  The docs should call it an extension example and keep the API intentionally
  small.
- Embedding fixed sparse metadata in an extension payload is useful for this
  tutorial, but large production sparse structures would need more careful cache
  and ownership design.
- Moving KdV out of the root workspace changes how broad local commands cover
  it. The explicit CI `cargo check --manifest-path` command is the new coverage
  contract.
- Tropical currently lives outside the root workspace. CI must explicitly test
  it, otherwise tutorial status would not imply execution coverage.
