# Def 1 — explicit trace/capture API 設計提案（unblock 用）

> 状態: **提案（設計確定待ち）**。#1665 の eager-AD Def 1（active-edge 化）を実装するために
> 必要な「明示 trace/capture API」の具体形を、現行コードの実装根拠に基づいて提案する。

## 1. 現行の暗黙録音（Def 1 が廃止するもの）の実態

- `EagerTensor::detach()`（`crates/tenferro-ad/src/eager.rs:3322`）は `requires_grad: false` だが
  `semantic_trace: Some(TracedTensor::from_tensor_symbolic_shape(value))` を作る。
- `record_semantic_eager_outputs`（`eager.rs:4006`）は「全 input が `semantic_trace: Some`」のときのみ
  演算を semantic graph に適用して伝播する。したがって detached/untracked 中間値の下流も semantic
  trace を持ち、後から `vjp`/`grad`/`backward` で functional AD できる（＝「untracked 中間値を後から
  wrt に選ぶ」を暗黙に支えている）。
- 一方 `record_eager_outputs_from_metadata`（`eager.rs:4133`）は `requires_grad == false` でも
  `eager_val_key()` + `register_scoped_metadata_batch`（global registry への mutex write）を実行する。
  これが eager-AD forward の ~8µs/op（#1664 の `record_untracked_outputs` に相当）。
- 現行の active-edge の片鱗: graph composite パス（`eager.rs:3769`）は `!any(requires_grad)` で
  記録を早出ししているが、single-op の `nary_op`（`eager_ops.rs:1256`）と extension の
  `finish_eager_extension_outputs`（`extension.rs:420`）は早出しせず、常に `record_eager_outputs` を
  呼ぶ。この不整合が Def 1 の対象。

## 2. 提案 API（既存 `no_grad()` guard の流儀に合わせる）

`no_grad()` は thread-local の `EAGER_NO_GRAD_DEPTH` を RAII で増減する。これと同型の
**capture ガード**を追加する:

```rust
// EagerRuntime::capture_trace(&self) -> EagerTraceCaptureGuard
let _capture = ctx.capture_trace();
let x = /* untracked leaf */;
let y = x.mul(&x)?;          // capture 中は semantic trace を明示的に記録
let dy = ctx.vjp(&y, &x, &seed)?;  // capture した graph で functional AD
// guard drop で capture 終了
```

- **capture ガード内**: `!any_requires_grad` の op でも `record_eager_outputs`（semantic trace 伝播 +
  metadata 登録）を実行（現行 branch 2 の挙動を「明示 opt-in」に限定）。
- **capture ガード外（通常時）**: `!any_requires_grad` ⇒ 記録ゼロ（`trace: None`、`semantic_trace: None`、
  metadata 登録なし、autograd node 0）。これが active-edge の本丸で、~8µs/op を回収。
- 実装は既存 `EAGER_NO_GRAD_DEPTH` と同様の thread-local depth + RAII。`record_semantic_eager_outputs`
  と `record_eager_outputs_from_metadata` の早出し判定に depth を加えるだけで、既存の
  semantic-trace / metadata 機構はそのまま再利用できる（新 IR・新 trait 不要）。

## 3. active-edge 化（Def 1 本体）の変更点

1. `eager_ops.rs:1256` の `!any_requires_grad` 分岐: capture ガード外なら `record_eager_outputs` を
   skip し、`new_unregistered_result_with_semantic_trace(..., semantic_trace: None, metadata_scopes: Vec::new())`
   へ（`eager.rs:3769` の graph パスと一致させる）。
2. `extension.rs:420` `finish_eager_extension_outputs`: 現行 `!eager_grad_recording_enabled()` 早出しに
   `|| (!capture_active && !any(requires_grad))` を追加。
3. `record_eager_outputs_from_metadata`: `requires_grad` が false かつ capture 外なら metadata batch を
   登録しない。

## 4. 移行手順（conclusion 5.4 の順序に整合）

1. `capture_trace()` ガードを追加（新 public API）。
2. 現行の「`detach()` → 後から functional AD」に依存するテストを `capture_trace()` に移行
   （migration test 先行）。
3. active-edge 化（上記 1〜3）を適用。
4. 既存 AD 数値テスト（JVP/VJP/HVP）を回帰 gate として全通し。

## 5. 未確定事項（ユーザー/保守者へ）

- `capture_trace()` の戻り値は「graph を返す」形（`EagerTraceCapture` が閉区間の `Graph<StdTensorOp>` を
  返す）か、「guard で囲んだ区間の後に `ctx.vjp(...)` が capture 済み graph を使う」形か。前者の方が
  functional AD の「同じ graph を複数回微分」に直結するが、API が大きい。後者は `no_grad()` と最小差。
- capture ガードと `no_grad()` の相互作用（capture 中の `no_grad()` は semantic trace も止めるべき）。
- スレッド安全性（`no_grad()` は thread-local。capture も thread-local でよいか、cross-thread capture が
  必要か）。

## 6. ベンチマークへの影響

- active-edge 化で `eager_ad_forward`（requires_grad だが grad 消費なし）の ~8µs が消え、no_ad と
  ほぼ同値（~12µs = session フロア）になる見込み。
- 残る ~11µs は session フロア（別 issue #1628/#1662 系）で、本提案の対象外。

## 7. 実験で確定した事実（branch 2 skip を実装して回帰テストで検証）

`nary_op` branch 2 の `record_eager_outputs` を skip + `finish_eager_extension_outputs` に
`|| !any(requires_grad)` を追加（active-edge 化のみ）を試し、**失敗は唯一 1 テスト**だった:

- `crates/tenferro-ad/tests/integration/eager_tensor.rs:314`
  `untracked_eager_intermediate_can_later_feed_tracked_ad` — `scale = plain.add(&plain)`（untracked 定数）
  を `x.mul(&scale)`（tracked）に食わせて backward → `x.grad() == scale`。

つまり Def 1 は **3 部構成**で、branch 2 skip だけでは「untracked 定数を tracked 鎖に食わせる」
（PyTorch で言う定数 leaf）が壊れる:

1. **branch 2 の記録 skip**（active-edge。上記実験で確認）
2. **branch 3 での constant 遅延 materialization** — tracked op に untracked 入力が入った時、その入力の
   concrete value から constant leaf を `TracedTensor::from_tensor_symbolic_shape` で遅延生成
   （`detach()` と同じ要領）。`record_semantic_eager_outputs`（`eager.rs:4006`）の
   `collect::<Option>` 全入力必須を「untracked 入力は constant leaf に」に変える。これで定数ケースが
   暗黙録音なしで成立（PyTorch の「untracked = constant、grad は流れない」に一致）。
3. **明示 trace/capture API**（§2）— 後から `wrt` に選ぶ functional AD のみ。

**重要**: 定数ケース（テスト 1 件）は「branch 2 の暗黙録音」でなく「branch 3 の遅延 constant
materialization」で維持すべき。branch 2 skip 単独では壊れるため、Def 1 は 1+2 を一体で実施し、
3 は functional-AD-over-wrt 専用。
