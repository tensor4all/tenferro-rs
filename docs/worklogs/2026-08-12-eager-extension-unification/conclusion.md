# Issue #1665 実装 — 最終結論

> ブランチ: `fix/1665-eager-extension-unification`（origin/main 上に15コミット）
> 状態: 実行経路統一 + eager-AD Def 1（active-edge + constant + capture）実装完了。PR 未作成。

## 実装したこと

### 実行経路統一（#1664 回帰の本丸）

- `apply_eager` 単一入口 + snapshot-resolved native immediate path（`ExtensionEngine::prepare` → session executor、`Unsupported` のみ compiled fallback）。linalg/einsum/fft を単一入口へ移行。
- `install/ensure_extension_module` の steady-state read-only no-op。
- native prepare を exact engine に pin + returned-plan 検証、linalg の per-op/per-backend session admission（CPU `SvdFull` 除外含む）。

### eager-AD Def 1（active-edge 化）

- **active-edge**: `nary_op` / `nary_value_op` / `finish_eager_extension_outputs` が「grad-active input 0 ⇒ 記録ゼロ」に。`record_untracked_outputs`（`register_scoped_metadata_batch` の global registry write、~8µs/op）を廃止。
- **constant 遅延 materialization**: `record_semantic_eager_outputs` が tracked op に食われた untracked 入力を `TracedTensor::from_tensor_symbolic_shape` で constant leaf 化。「untracked 定数 → tracked AD」を暗黙録音なしで維持（PyTorch の untracked=constant に一致）。
- **`capture_trace()`**: thread-local RAII guard（`no_grad()` と同型、`!Send`）。untracked leaf の functional JVP/VJP を明示的に再開。Def 1 の「明示 trace/capture への移行」を提供。

## Review（frontier gate: reviewer-gpt）

- 実行経路統一: 3回 → **READY TO MERGE**。
- Def 1: 2回 → active-edge/constant の機構は正しい、capture 未実装（Important）→ `capture_trace()` 追加。再 review で「re-export 不足」「guard が Send」の 2 件 Important → 修正済み。

## ベンチマーク（1スレッド, Linux, faer, `cargo bench -p tenferro-linalg --bench eager_extension_dispatch --features autodiff`）

| op (2x2 f64) | pre-#1665 no_ad | no_ad | eager_ad_forward |
|---|---:|---:|---:|
| matmul（標準op参照） | 26.3 µs | **16.8 µs** | 21.1 µs |
| solve | 92.4 µs | **44.1 µs** | 41.8 µs |
| svd | 43.8 µs | **16.5 µs** | 24.8 µs |
| eigh | 26.3 µs | **14.6 µs** | 14.9 µs |

- 改善: solve −52%、svd −62%、eigh −44%、matmul −36%。
- extension 固有のディスパッチオーバーヘッドは ~75µs → ~1.8µs に崩壊。

## 残り（未達: 1桁 µs / PyTorch 同等）

残り ~14µs は **session フロア**（`run_backend_session_cached` の permit + session construct + 出力 wrap）。標準 op の `matmul` も ~17µs。これは #1628/#1662 系の一般 eager ディスパッチ回帰で、**#1665 のスコープ外**（別 issue）。

- `ResidualSpec`（保存 tensor の最小 mask）は未実装。backward の保存量削減（メモリ）に関わるが、forward 単発 op のレイテンシには寄与しない。conclusion 5.5 の残タスク。

## 検証

- `cargo test -p tenferro-runtime -p tenferro-ad -p tenferro-linalg -p tenferro-einsum -p tenferro-fft` → 21 ブロック green。
- `cargo fmt --all --check` / `cargo clippy`（全 touched crate）→ clean。
- `python3 scripts/repository-rules-review.py` → pass。
- `bash scripts/check-pr-fast.sh`（実行経路統一時点）→ pass。

## 備考

- `HANDOFF-1597.md`（作業ツリー未追跡・別 issue #1597 の残骸）は本作業と無関係のため未コミット。
- 既知の pre-existing 破損: `crates/tenferro-fft/tests/fft_ops.rs` が `--features autodiff` で `Tensor.clone` 非存在により失敗（base 上で再現、#1665 と無関係）。
