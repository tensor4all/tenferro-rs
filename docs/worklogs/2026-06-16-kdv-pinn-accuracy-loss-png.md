# Work Log: KdV PINN Accuracy Improvement + Loss Curve PNG

Date: 2026-06-16
Branch: `kdv-pinn`

Follow-up to [`2026-06-15-kdv-pinn-sample.md`](2026-06-15-kdv-pinn-sample.md).

## Goal

Reduce the prediction error of the `kdv_pinn` sample — the predicted soliton
visibly lost amplitude over time at `t = 0.5` — and add a PNG plot of the
training-loss curve.

The prior work log's residual risk explicitly predicted that "further reduction
would require a larger network, longer training, adaptive collocation sampling,
or a learning-rate schedule." This session pursued the learning-rate schedule
and denser sampling.

## Context Read

- Re-read [`2026-06-15-kdv-pinn-sample.md`](2026-06-15-kdv-pinn-sample.md): the
  baseline used MLP `[2, 64, 64, 1]`, `N_COL = 512`, `N_IC = N_BC = 64`,
  `EPOCHS = 1000`, fixed `LR = 0.001`, giving an L2 relative error at `t = 0.5`
  of ≈15%.
- Confirmed the KdV residual (`u_t + 6 u u_x + u_xxx`), the initial condition
  (`2 sech^2(x)`), and the boundary data (`2 sech^2(x - 4t)`) are mutually
  consistent — the amplitude decay is a training/convergence issue, not a bug.
- Verified the `plotters` 0.3 APIs used: `IntoLogRange::log_scale()` (re-exported
  in the prelude) for the log y-axis, and `BitMapBackend::new` for PNG output
  (the `image` feature is already enabled because the GIF backend uses it).

## Chosen Design

- **Adam learning-rate step schedule**: added `Adam::set_lr` so the schedule can
  change the rate without discarding the moment buffers, and a pure
  `step_decay_lr(epoch, total, base)` helper that returns `base` for the first
  half of training, `base / 2` for the third quarter, and `base / 4` for the
  final quarter. The training loop calls `opt.set_lr(step_decay_lr(...))` each
  epoch. The loss curve shows a clear inflection at each decay point.
- **Denser sampling + longer training**: `N_COL` 512 → 1024 (stronger interior
  PDE enforcement), `N_IC` / `N_BC` 64 → 128 (better-anchored initial/boundary
  conditions, cheap because they carry no high-order AD), `EPOCHS` 1000 → 3000.
- **Loss-curve PNG**: `plot::write_loss_png` renders the per-epoch loss with a
  **logarithmic y-axis** (the loss spans ~3.5 decades). `plot::loss_axis_bounds`
  computes the axis range, ignoring non-positive / non-finite values (a log axis
  cannot show them) and falling back to a safe `(1e-8, 1.0)` decade when no
  positive value exists. `main.rs` records `loss_history` each epoch.
- **CLI**: generalized `gif_path_from_args` into `arg_value(flag)` and added a
  `--loss-png <path>` flag alongside the existing `--gif <path>`.
- **TDD**: `set_lr`, `step_decay_lr`, and `loss_axis_bounds` were written
  test-first (`optimizer/tests.rs`, `plot/tests.rs`); the renderers
  (`write_loss_png`, like the existing `write_comparison_gif`) are exercised by
  the actual run rather than a unit test.

## Rejected Alternatives

- **Deeper network `[2, 64, 64, 64, 1]`**: tried at the user's request, then
  reverted. On this machine the 4-layer run executed mostly single-threaded and
  exceeded 9 minutes for 1000 epochs with no accuracy guarantee, whereas the
  3-layer 3000-epoch configuration reliably reaches ≈2%. Extra depth remains an
  open option but was not worth the runtime/uncertainty in this session.
- **Dialed-back "≈1 minute" configuration** (`N_COL = 512`, `EPOCHS = 700`):
  produced an L2 error of ≈68% — badly under-trained, and sensitive to a poor
  random seed (initial loss 3.2 vs ~1.0 on a good seed). A converged result is
  not achievable in ≈1 minute on this hardware.
- **L-BFGS second-stage optimization** (the standard PINN polish): rejected for
  this session because the codebase has no L-BFGS optimizer and adding one is a
  large change out of scope for an example tweak.

## Key Adjustments During Implementation

- **Runtime reality**: wall-clock is far higher than first estimated — roughly
  0.17–0.28 s/epoch depending on threading, so even the original 1000-epoch
  baseline already took minutes, and the 3000-epoch configuration takes ~8.5
  minutes (real ≈513 s). Threading behavior is inconsistent between runs (one
  3000-epoch run used many cores, `sys ≈ 1700 s`; a 700-epoch run ran nearly
  single-threaded), so epoch count cannot be used to hit a precise time target.
- **One-time graph-compilation cost**: ~1 min 40 s elapses building and
  compiling the loss program plus the six third-order-AD gradient programs
  before `epoch 0` prints. This is not a hang.
- **Seed variance**: with `rand::thread_rng()` the final L2 error varies roughly
  1.5–2% between runs depending on initialization.

## Residual Risks

- **Runtime**: ~8.5 minutes per full run on the development machine. Acceptable
  for a sample but not interactive.
- **Seed variance**: no fixed seed; results vary run to run (≈1.5–2% at
  `t = 0.5`). A `--seed` flag would make runs reproducible.
- **Depth unexplored**: a deeper or wider network plus the schedule may push the
  error lower but was not characterized here.
- **Pre-existing clippy warnings**: `src/pde/tests.rs` and `src/sampler/tests.rs`
  carry 5 style/complexity warnings (`needless_range_loop`,
  `manual_range_contains`) that predate this session and were left untouched to
  keep the diff scoped.

## Verification

- `cargo fmt --all --check` ✅
- `cargo test -p kdv_pinn --release` ✅ 26/26 tests pass (21 prior + 5 new)
- `cargo clippy -p kdv_pinn --release --all-targets` — no warnings in code added
  this session; 5 pre-existing test-file warnings remain (see Residual Risks).
- `cargo run -p kdv_pinn --release -- --gif kdv_pinn.gif --loss-png loss.png` ✅
  final loss `6.51e-4`, L2 relative error at `t = 0.5` `1.99%` (down from ≈15%),
  produced `kdv_pinn.gif` and `loss.png` (log-scale loss curve, visually
  verified).
