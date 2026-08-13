# Issue #1665 実装 — 最終結論

> ブランチ: `fix/1665-eager-extension-unification`（origin/main 上に5コミット）
> 状態: 実装完了・review pass・ベンチマーク改善確認済み。PR 未作成。

## 実装したこと（5コミット）

| commit | 内容 |
|---|---|
| `25950e78` | `eager_extension_dispatch` criterion bench（no-AD / eager-AD 両パス、solve/svd/eigh/matmul 2x2） |
| `ca5bd395` | `install_extension_module` / `ensure_extension_module_for_engine` の steady-state read-only no-op（公開 snapshot の読み取りのみで install mutex + reconfigure を回避） |
| `d4d44ef5` | `apply_eager` 単一入口 + snapshot-resolved native immediate path（`ExtensionEngine::prepare` → session executor、`Unsupported` のみ compiled へ fallback）。linalg/einsum/fft を単一入口へ移行 |
| `ccfa046b` | review 指摘修正: native prepare を exact engine に pin（first-slot 探索を廃止）+ non-executable/ingress/returned-plan 検証 + linalg の per-op/per-backend session admission |
| `1bf764b9` | review 指摘修正: CPU `SvdFull` を session admission から除外（BLAS provider に in-session full SVD が無いため） |

## Review（frontier gate: reviewer-gpt）

- 1回目: **BLOCK** — 2件 Important（native prepare が compiled の provider選択/検証を再現しない / linalg が Unsupported op を over-admit）。
- 2回目: **BLOCK** — Finding 1 解消、CUDA admission 解消。ただし CPU BLAS `SvdFull` の over-admit が残る。
- 3回目: **READY TO MERGE** — 全指摘解消、`supports_session()==true ⇒ 実行が Unsupported を返さない` 契約が成立。

## ベンチマーク結果（1スレッド, Linux, faer, `cargo bench -p tenferro-linalg --bench eager_extension_dispatch --features autodiff`）

| op (2x2 f64) | pre-#1665 no_ad | no_ad | eager_ad_forward |
|---|---:|---:|---:|
| matmul（標準op参照） | 26.3 µs | ~24 µs | ~31 µs |
| solve | 92.4 µs | **~60 µs** | ~84 µs |
| svd | 43.8 µs | **~27 µs** | ~28 µs |
| eigh | 26.3 µs | ~25 µs | ~22 µs |

- **extension 固有のディスパッチオーバーヘッドは ~75µs → ~1.8µs に崩壊**（deepseek の計測: validate 0.03 + reads 0.87 + input_signature 0.34 + prepare 0.42 + install 0.13 µs）。
- solve/svd は 30〜40% 改善。solve の残余は op 固有コスト + 共有フロア。

## 未達: PyTorch 同等（1桁 µs）

単一 op の残り ~20µs は**共有 eager-op フロア**で、内訳は2つに分かれる:

1. `session_execute` ~11µs（backend lock + session open + 実計算）— **#1628/#1662 系の別 issue**（一般 eager ディスパッチ）。
2. `apply.finish` ~8µs（出力 wrap + `register_scoped_metadata_batch` の global registry write）— **eager-AD Def 1 の対象**。現行は「grad-active input 0 でも全出力を metadata batch に登録」する implicit all-untracked 録音（`record_eager_outputs_from_metadata` の `requires_grad == false` でも `eager_val_key()` + `register_scoped_metadata_batch` を実行）。Def 1 の「grad-active input 0 ⇒ autograd node 0」で消せる。

**フロアはベンチマークの artifacts ではない**ことを確認: `eager_dispatch_baseline` の `lazy/dot_general_f64/2`（`to_tensor()` なし）も ~22µs。つまり `apply.finish` の ~8µs は #1665 の eager-AD 部分（Def 1）で回収可能だが、残り ~11µs は別 issue の session フロア。

## 残タスク

- **PyTorch 同等（1桁 µs）**: 上記共有フロアの削減（session 再利用 / 出力 wrap 削減）。#1665 のスコープ外。
- eager-AD の Def 1（active-edge 化・ResidualSpec 最小保存）: 設計は #1665 最終コメントにあり、**未実装**。今回の forward 高速化とは独立の breaking change。
- PR: ベンチマーク改善（extension 28〜40%）は確認済み。作成可否はユーザー判断待ち。

## 備考

- `HANDOFF-1597.md`（作業ツリー未追跡・別 issue #1597 の残骸）は本作業と無関係のため未コミット。
- 既知の pre-existing 破損: `crates/tenferro-fft/tests/fft_ops.rs` が `--features autodiff` で `Tensor.clone` 非存在により失敗（base 上で再現、#1665 と無関係）。
