# Issue #1665 実装 — 最終結論

> ブランチ: `fix/1665-eager-extension-unification`（origin/main 上に27コミット）
> 状態: **#1665 の in-scope 全項目 実装完了・review pass・全 gate green**

## 実装したこと（4項目すべて）

### 1. 実行経路統一（#1664 回帰の本丸）
- 単一公開入口 `apply_eager` + snapshot-resolved native immediate path
- **native-session / native-context の両経路**（session executor / 必須 `execute` via `ErasedExecutionContext`）
- steady-state install/ensure no-op（read-only snapshot 検査）
- linalg/einsum/fft を単一入口へ移行、旧 direct bridge を wrapper 化
- per-op/per-backend session admission（CPU `SvdFull` 除外含む）

### 2. eager-AD Def 1（active-edge 化）
- **active-edge**: grad-active input 0 ⇒ 記録ゼロ（`record_untracked_outputs` の ~8µs 廃止）
- **constant 遅延 materialization**: untracked 定数 → tracked AD を暗黙録音なしで維持
- **`capture_trace()`**: untracked leaf の functional AD を明示再開（`!Send` thread-local guard）

### 3. deferred AD materialization
- `extension::apply` を `append_raw_op`（raw `Graph<StdTensorOp>` carrier、O(inputs)/op、analysis なし）+ `analyze_extension_graph` に分解
- eager forward は append のみ。analysis（`register_scoped_graph_analysis` / `infer_output_meta`）は初回 AD 要求時
- **symbolic leaf metadata を `TracedTensor.leaf_metas` に保持**し、deferred analysis が symbolic で seed（fingerprint が traced path と一致）
- `metadata_scopes` Vec + per-op registry write を廃止
- 既存 `compile_ad_source` / `AdTransformCache` / prepared-derivative cache を再利用

### 4. ResidualSpec（保存 tensor 限定）
- `ResidualSpec`（primal input/output index の bitmask）を AD rule 表面に追加
- 全 rule に mask 宣言（add: none / mul, div: all inputs / abs 等 unary: input(0) / reshape: metadata-only / linalg: all inputs + all outputs）
- **未宣言アクセス検出器**（`TransposeInputRef::check_declared` の debug_assert）
- 移行順序: 検出器 → 全 rule に mask → AD 数値テスト全通し

## ベンチマーク（1スレッド, Linux, faer, `cargo bench -p tenferro-linalg --bench eager_extension_dispatch --features autodiff`）

| op (2x2 f64) | pre-#1665 no_ad | no_ad | eager_ad（tracked） |
|---|---:|---:|---:|
| matmul（標準op参照） | 26.3 µs | ~16-20 µs | ~1.2x no_ad |
| solve | 92.4 µs | **~41-45 µs** | ~1.2-1.3x no_ad |
| svd | 43.8 µs | **~17-19 µs** | ~1.2x no_ad |
| eigh | 26.3 µs | **~15-16 µs** | ~1.2x no_ad |

- no-AD: −44〜−60%。
- **eager-AD tracked forward は ~2x → ~1.2x no_ad**（deferred materialization で +13〜35µs → +3〜11µs）。
- extension 固有のディスパッチオーバーヘッド: ~75µs → ~1.8µs。

## Review（frontier gate: reviewer-gpt）

| 変更 | レビュー結果 |
|---|---|
| 実行経路統一 | 3回 → READY TO MERGE |
| Def 1（active-edge + capture） | 2回 → 全 Important 修正 |
| deferred materialization | 2回 → symbolic-metadata 修正 → **READY TO MERGE** |
| ResidualSpec | 2回 → linalg output mask 修正 → **PASS** |

## 検証

- `cargo test -p tenferro-internal-ops -p tenferro-runtime -p tenferro-ad -p tenferro-linalg -p tenferro-einsum -p tenferro-fft` → 24 ブロック green（`--features autodiff` 含む）
- ext/sparse + ext/tropical → green
- `cargo fmt --all --check` / `cargo clippy`（CI パリティ含む）/ `check-pr-fast.sh` / `repository-rules-review.py` → all pass

## 残り

- **session フロア（~14µs）**: #1667 として別 issue 化（#1628/#1662 系、本 PR のスコープ外）
- **PR**: ベンチマーク改善確認済み。本 issue の PR を作成する

## 備考

- `HANDOFF-1597.md`（作業ツリー未追跡・別 issue #1597 の残骸）は無関係のため未コミット。
- 既知の pre-existing 破損: `crates/tenferro-fft/tests/fft_ops.rs` が `--features autodiff` で `Tensor.clone` 非存在により失敗（base 上で再現、#1665 と無関係）。
- tidu-rs への AD 抽出は、tenferro 内製 AD が安定してから別途実施（ユーザー決定）。
