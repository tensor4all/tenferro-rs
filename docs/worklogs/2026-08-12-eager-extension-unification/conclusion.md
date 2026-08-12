# 最終結論: eager extension 経路統一と single-op オーバーヘッド削減（＋ eager AD の最小保存）

> デベート: `/Users/hiroshi/projects/tensor4all/tenferro-rs/.pi-meetings/2026-08-12-eager_extension_経路統一_single-op_削減`
> 参加者: GPT, DeepSeek（Researcher は fireworks API key 欠如で不参加）
> サイクル: Cycle 1〜5。ユーザーフィードバック3回（AD不要・concrete計算のみ / eager AD の保存最小化 / Def 1 採用）
> 状態: **完全収束**

## 問題と大前提

v0.3.0 の eager extension 実行は2経路が併存（generic `apply_eager` → SemanticProgram 構築毎回、direct session bridge `apply_eager_with_extension_session`）。single-op で 46〜238μs。

大前提: **外部ユーザーが機能を拡張しやすいこと**。
ユーザー方針: **breaking change を許容して最善の設計を目指す**。

---

## 1. 実行経路の統一

1. **単一公開入口 `apply_eager`**。call-site callback bridge（`apply_eager_with_extension_session` 系）は公開経路から外す。
2. **targeted ensure の steady-state no-op 化**（独立コミット）。真の対象は `CandidateConfig::from_snapshot` の全クローン + install mutex の回避。install-or-replace の意味論は不変。
3. `apply_eager` 内に **snapshot-resolved native immediate path**。別 slot cache・新 trait・static eligibility bit・one-core lowering・general plan cache は**追加しない**。
4. `Prepared` は `execute_in_session` または必須 `execute` で即時実行、`Unsupported` のみ prepared graph へ昇格。`prepare(Unsupported)` は allocation/planning 前に早出し。

## 2. 三段階の能力契約（外部拡張容易性）

| レベル | 作者が実装するもの | 得られるもの | 不要なもの |
|---|---|---|---|
| **Concrete-only**（AD 不要・Tensor 計算のみ） | typed executor 1本 + 任意で wrapper、必要なら狭い backend trait | backend-explicit immediate 計算 | `ExtensionOp`、module、engine、AD rule |
| **Runtime-integrated / no AD** | 上記 + `ExtensionOp` + `infer_output_meta` + effects/aliases + stable adapter | unified `apply_eager`、traced/compiled、provider selection、cache | AD rule |
| **Runtime-integrated + AD** | 上記 + canonical AD rule 1個 + `ResidualSpec` | JVP/VJP/linearize/transpose | forward executor の再実装 |

- concrete-only に `ExtensionOp` も AD rule も不要（einsum `concrete.rs` の `TensorTensordotExt` が実在証明）。依存は `tenferro-tensor` のみ。**ただし既存 `TensorBackend` primitive の合成に限定**（新規 leaf kernel は外部実装不能の可能性を recipe に明記）。
- Tier 1→2 は崖（`infer_output_meta` 必須 + effects/aliases + erased shim）。実コスト表をドキュメントに明記。
- stable adapter は engine/prepared/erased shim を隠す。internal macro の公開安定化は v0.3.0 の必須条件にしないが、out-of-tree fixture が public API だけで tier 2/3 を実装できること。

## 3. eager AD の設計（Cycle 4-5、最終合意）

### 3.1 forward は grad の有無で分岐しない

`requires_grad` は immediate eligibility に入れない。forward は no-grad と同じ snapshot-resolved `prepare → execute` を通る。

### 3.2 AD tape の carrier は既存 `Graph<StdTensorOp>`

新しい `EagerTapeIR` / `SemanticRecordNode` は**作らない**。`extension::apply` を次の二段へ因数分解:

1. `append_raw_extension_op`: graph parent、ordered input edge、op node、output ids だけを追加（O(1)/op）
2. `analyze_extension_graph`: leaf schema から canonical metadata/constraints を推論

trace は 1→2、eager autograd は 1 のみ（analysis は初回 AD 要求まで遅延）。`metadata_scopes: Vec<_>` 持ち回りと per-op global registry read/write を廃止。leaf/binding 情報は一度だけ value に付け、親は Arc/stable id で共有。

### 3.3 保存 tensor は `ResidualSpec` で限定（PyTorch 級）

`ResidualSpec` = semantic primal の input/output index + 小 shape/dtype metadata mask。

- `add`: tensor residual 0 / `mul`: 相手 input のみ / `exp`: output のみ / reshape: metadata のみ
- AD rule なしの extension: tensor residual 0。AD 要求時のみ typed missing-rule error
- provider/prepared/session/snapshot/algorithm plan の保存は **0**
- 外部作者が書くのは typed executor 一本 + canonical rule 一つ + residual mask 一つ（PyTorch `save_for_backward` と同程度）

### 3.4 初回 AD 要求で一度だけ canonical materialization

`backward`/VJP/JVP/linearize の最初に、root から reachable raw Graph を一度 walk:

1. cut leaf から canonical symbolic leaves（concrete extent は binding data、fingerprint に焼き込まない）
2. 共通 analyzer で `infer_output_meta`/effects/aliases を topological order に一度だけ
3. 共通 `SemanticProgramBuilder` / freeze
4. active roots/wrt で normalize/slice
5. 既存 `AdTransformCache` + 共通 semantic rule
6. derivative も runtime-owned prepared executor を通す

O(reachable)、per-op 再解析ゼロ。per-root transform cache 新設なし。

### 3.5 Def 1（採用済み、breaking change）

**ordinary eager autograd を PyTorch 型にする:**

- `no_grad`: tape なし
- recording enabled でも grad-active input が 0: autograd node 0
- 1 個以上の grad-active input: active edge だけを持つ node を 1 個追加
- untracked 中間値を後から `wrt` に選ぶ functional JVP/VJP は**明示 trace/capture API** で囲む

→ 現行の「全 untracked history を暗黙保存」を廃止する **AD API の breaking change**。v0.3.0 migration note に明記し、`detach` / 明示 trace への移行テストを perf 実装より先に置く。

### 3.6 経路同一の定義（Def 2）

「同じ」とは **raw op/Graph、canonical analyzer/`SemanticProgram`/rule、runtime prepared executor が同じ** という意味。差を許すのは phase の実行時刻のみ。eager から materialize した AD source と full trace の `semantic_eq` 比較は、**同じ active roots/wrt で slice した normalized AD source** に対して行う。

## 4. 最終 gate（受け入れ条件）

### 保存量
- all-untracked chain: autograd node 0、saved tensor 0
- grad-active op: raw Graph node は multi-output でも 1
- saved tensor count/bytes は `ResidualSpec` と完全一致
- provider/prepared/session/snapshot handle 保存 0

### forward capture cost
- `extension::apply` 再入 0、`infer_output_meta` 0、global registry read/write 0、scope Vec merge/dedup 0
- work は O(arity + selected residual count)、unary chain depth 非依存
- **multi-thread capture が lock で直列化されない**（不変 Arc ノード append、stable-id は atomic、residual は Arc 共有。実測スケーリング gate）

### AD materialization
- canonical inference は reachable op あたり最大 1 回/AD request、O(reachable)
- eager active-sliced source と equivalent trace source が `semantic_eq`/fingerprint/JVP/VJP/HVP/typed error で一致
- shape-churn（長さ 2/3）で同一 symbolic fingerprint
- **data-dependent shape（nonzero/topk 等）の op は `ResidualSpec` に concrete shape/dtype 宣言を必須化**。未宣言アクセスは fixture で落とす

### extension execution
- `native-session` / `native-context` を別々に direct baseline へ非劣性
- eligible path で SemanticProgram/fingerprint/compiler/schedule/admission 0、session entry 最大1
- `prepare(Unsupported)` は allocation 前、promoted path は非退行

### external authoring
- no-AD fixture は tape API/ResidualSpec を実装しない
- AD fixture は typed executor + canonical rule + residual mask で eager/trace 双方通過
- concrete-only fixture は runtime/AD 依存なし、合成限定を明記

### ResidualSpec 移行手順
既存 AD rule 全件に mask 宣言 → 未宣言検出 fixture/`debug_assert` を先に置く → 既存 AD 数値テストを回帰 gate として全通し。recompute/checkpoint 一般化は v0.3.0 に混ぜない。

## 5. 実装順序

1. baseline を no-grad / all-untracked / grad-active / first AD / N forward + one AD に分け、recorder 5 内訳（GraphBuilder / analysis+infer / registry read-write / scope Vec / concrete meta 登録）を counter 化
2. targeted ensure の read-only hit を独立修正・計測
3. eager/compiled 共通の runtime-owned prepared execution helper。`requires_grad` に関係なく immediate forward
4. active-edge semantics 化（Def 1）。detach / 明示 trace の migration test 先行
5. canonical AD rule に最小 residual mask を追加
6. `extension::apply` から raw append と analysis を分離。scope Vec と per-op registry 登録を除去
7. eager root materialization を共通 analyzer / `SemanticProgramBuilder` に接続
8. first-party wrapper + out-of-tree fixture を単一 `apply_eager` へ移し、call-site direct bridge を公開経路から外す

## 6. 残る承認済み breaking change

- **Def 1**: all-untracked の暗黙録音を廃止（PyTorch 型 active-edge）。functional AD の全履歴 capture は明示 trace API へ。migration note 必須。

## 関連 issue / ファイル

- #1664（v0.3.0 eager regression 報告）
- #1665（本設計の実装 issue）
- #1661（LLM 向けドキュメント改善）
- デベート全記録: `docs/worklogs/2026-08-12-eager-extension-unification/`（transcript.md 同梱）
