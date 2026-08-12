# 結論: eager extension 経路統一と single-op オーバーヘッド削減

> デベート: `/Users/hiroshi/projects/tensor4all/tenferro-rs/.pi-meetings/2026-08-12-eager_extension_経路統一_single-op_削減`
> 参加者: GPT, DeepSeek（Researcher は fireworks API key 欠如で不参加）
> サイクル: Cycle 1〜3（Cycle 3 でユーザーフィードバック「AD 不要・concrete 計算のみ」を反映）

## 問題

v0.3.0 の eager extension 実行には2経路が併存する:

- generic 経路: `apply_eager` → `SemanticProgram` 構築 → compile → `Runtime::run_compiled`（毎回フルパイプライン。single-op で 46〜238μs）
- direct session bridge: `apply_eager_with_extension_session`（標準 linalg/einsum/fft が使用。毎回 `install_extension_module` → `reconfigure` 相当のコスト）

大前提: **外部ユーザーが機能を拡張しやすいこと**。

## 合意スコープ

### 実行経路の統一

1. **単一公開入口 `apply_eager`** に統一。call-site callback bridge（`apply_eager_with_extension_session` 系）は公開経路から外す。
2. **targeted ensure の steady-state no-op 化** を独立コミット。真の対象は `CandidateConfig::from_snapshot` の全クローン + install mutex の回避。install-or-replace の意味論は不変。
3. `apply_eager` 内に **snapshot-resolved native immediate path** を追加。別 slot cache・新 trait・static eligibility bit・one-core lowering・general plan cache は**追加しない**。
4. `Prepared` は `execute_in_session` または必須 `execute` で即時実行し、`Unsupported` のみ prepared graph へ昇格。

### 三段階の能力契約（外部ユーザーの拡張容易性）

| レベル | 作者が実装するもの | 得られるもの | 不要なもの |
|---|---|---|---|
| **Concrete-only**（AD 不要・Tensor 計算のみ） | typed executor 1本 + 任意で `Tensor`/`TypedTensor` wrapper、必要なら狭い backend trait | backend-explicit immediate 計算 | `ExtensionOp`、module、engine、AD rule |
| **Runtime-integrated / no AD** | 上記 + semantic `ExtensionOp` + stable adapter | unified `apply_eager`、traced/compiled、provider selection、runtime cache | AD rule |
| **Runtime-integrated + AD** | 上記 + optional AD rule 1個 | JVP/VJP/linearize/transpose | forward executor の再実装 |

- **concrete-only に `ExtensionOp` も AD rule も不要**（einsum `concrete.rs` の `TensorTensordotExt` が実在証明）。依存は原則 `tenferro-tensor` のみ。
- 二層の関係は「one kernel source of truth」: concrete typed executor を runtime adapter が包む。kernel/validation/dtype semantics を再実装しない。
- AD rule なしの runtime extension は forward eager/compiled を正常実行し、微分要求時のみ typed missing-rule error。暗黙 fallback なし。

### 重要な但し書き（DeepSeek の追加要求）

1. **Tier 1→2 は崖**: `ExtensionOp` は 8メソッド + `infer_output_meta`（symbolic shape/dtype 推論、必須）+ effects/aliases 宣言 + erased shim。これは「kernel 1本」ではない。実コスト表をドキュメントに明記すべき。
2. **concrete-only は既存 `TensorBackend` primitive の合成に限定**。新規 leaf kernel は外部実装不能の可能性（backend 内部が `pub(crate)` の場合）。recipe に境界を明記。
3. **stable adapter の最小形を文書で指定**（署名レベル）。未定義のまま fixture B を gate に入れない。

### gate（受け入れ条件）

1. 数値非劣性: `native-session` と `native-context` の二系列それぞれを direct baseline に非劣性。generic fallback は非退行（`prepare()` 試行後の promoted も分離計測）。
2. structural counter: graph artifact zero、session/resource entry 最大一回、`ExtensionCacheStore` 供給と backend context 取得の内訳観測。
3. `Unsupported` を返す `prepare` が allocation しない（capability 判定が planning より先）。
4. 余分な allocation zero（view/must-alias の混同はしない）。
5. 意味論・typed error・AD JVP/VJP の eager/compiled 一致（out-of-tree fixture で両経路通過）。
6. observability: `native-session` / `native-context` / `promoted-graph(reason)` を opt-in / disabled 時ほぼゼロコストで。

### fixture（2本に分離）

- **fixture A（concrete-only）**: stable public API のみ、依存 `tenferro-tensor` のみ、typed executor 1本から forward が通る。semantic/snapshot/module/engine counter ゼロ。
- **fixture B（runtime-upgrade）**: A の executor を再利用して `ExtensionOp + stable adapter` を追加。AD rule なしで unified `apply_eager` + compiled forward。微分要求だけ typed missing-rule error。

## 残課題（次の議論）

- **eager AD のオーバーヘッド**: requires_grad の eager op は AD tape 記録（record → TracedGraph → SemanticProgram）を伴う。forward のみの fast path と AD 記録の関係を設計する必要がある（ユーザー追加フィードバック）。
- stable adapter の具体形。
