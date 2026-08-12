# Facilitator (Topic): デベートの論点と背景を提示

# デベート論点

tenferro-rs v0.3.0 の eager extension 実行には2つの経路が併存する:

- **generic 経路**: `apply_eager` → `SemanticProgram` 構築 → `GraphCompiler::compile_frozen_program` → `Runtime::run_compiled`（毎回フルパイプライン。single-op には重く、計測で 46〜238μs）
- **direct session bridge**: `apply_eager_with_extension_session` → backend session 直接 dispatch（標準 linalg/einsum/fft が使用。ただし毎回 `install_extension_module` → `Runtime::reconfigure` を呼ぶ）

## 目標
2つの経路を統一しつつ、single-op のオーバーヘッドを減らす。

## 大前提（譲れない制約）
**外部ユーザーが機能を拡張しやすいこと。**

## 候補（議論の叩き台）
- (a) SemanticProgram に統一 + compile/prepare の有界キャッシュ（(family, payload_hash, dtype, shape, engine, epoch) キー）
- (b) direct bridge に統一（v0.2.0 型の backend 直接 dispatch）
- (c) Runtime レベルの single-op fast path（SemanticProgram 構築なしの pre-resolved slot）

## 関連する設計事実
- 設計文書 `docs/design/execution-engine-provider-architecture.md` は「Eager single-operation cost contract」で single-op に SemanticProgram 構築・fingerprint・ScheduledGraph・admission を課すことを明示的に禁止し、plan cache は provider-local の有界キャッシュと規定。
- v0.3.0 の storage ownership redesign（#1555）で Tensor は move-only owner。eager↔traced は SemanticProgram 表現を共有することで AD 整合性を保つ。
# GPT (Cycle 1): 統一すべきは公開契約であり、実行アーティファクトではない

# 結論

**(c) Runtime レベルの single-op fast path を採る。** ただし「統一」を「全呼び出しを同じ物理パイプラインへ押し込むこと」と解釈してはならない。統一対象は、公開 API、意味論、provider 選択、placement、error、AD 記録、および extension author が実装する adapter である。`SemanticProgram`、fingerprint、`ScheduledGraph` まで毎回同じにする必要はない。むしろ accepted design はそれを明示的に禁じている。

公開入口は `apply_eager(op, inputs)` の一つにする。その内部で eligible な single-op は runtime の resolved extension slot へ直行し、eligible でないものだけ prepared-graph 経路へ昇格させる。現在の direct session bridge の有用部分、すなわち「一度だけ backend session に入り extension cache と一緒に実行する」ことは runtime 内部の実行機構として残す。**call site が callback を渡す公開経路は廃止する。**

# (a) は設計ミスをキャッシュで隠しているだけ

`SemanticProgram` に統一して有界キャッシュを置く案は、根本原因に触れていない。46〜238 μs の不要な pipeline を残し、同一 shape の再実行だけを速く見せる案である。これは最適化ではなくベンチマークの選別だ。changing-shape、cold call、payload が頻繁に変わる extension では何も解決しない。

提示キー `(family, payload_hash, dtype, shape, engine, epoch)` も不十分である。少なくとも runtime identity、複数入力の layout/stride/alignment/storage/placement、prepare options、planning config、hardware class、specialization projection が欠ける。`payload_hash` collision には `payload_eq` が必要で、exact shape を無条件に入れれば polymorphic provider を過剰 specialization する。逆に省けば shape-dependent linalg/einsum plan を誤再利用する。つまり generic layer は provider が本当に必要とする key を知らない。

さらに、現在の direct bridge のように呼ぶたび `reconfigure` すれば epoch が揺れ、せっかくの cache を自分で無効化する。キャッシュを足す前に、不要な program 構築と per-call registration を消すべきだ。必要な plan cache は設計文書どおり provider-local・bounded に置く。graph の prepared-plan cache と single-op の algorithm plan cache を混ぜてはならない。

# (b) は「高速」だが外部 extension を二級市民にする

v0.2 型 direct bridge への統一も却下する。`apply_eager_with_extension_session` は、呼び出し側が module と backend-session callback を毎回持ち込み、runtime の `ExtensionEngine` / prepared executor を迂回する。これは統一ではなく、first-party crate だけが使いやすい裏口を正規 API に昇格する案だ。

この案では extension author は、semantic `ExtensionOp`、graph 用 engine/prepared executor、eager 用 session callback の二重実装を背負う。片方だけ直せば eager と traced で結果、validation、provider 選択、cache、typed error がずれる。CUDA/WebGPU、multi-engine、placement、explicit transfer、event domain が増えるほど破綻する。`dyn BackendSession` を万能 extension ABI とみなす隠れた前提も誤りである。

しかも現行 API は hot call ごとに module install/reconfigure を通る。同じ `Arc` の `OnceLock` で epoch 更新を偶然避けても、lock、snapshot、transaction の経路は残る。外部作者が毎回同値だが別 `Arc` の module を返せば replacement と epoch churn を起こし得る。「利用者が正しく static module を組めば速い」は拡張容易性ではなく罠である。

# 提案する実行フロー

1. **登録と呼び出しを分離する。** `ExtensionModule` は runtime builder、明示的 `enable/install`、または extension crate の first-use convenience で一度だけ登録する。first-use convenience は immutable snapshot で既登録を確認し、missing のときだけ module factory を評価して transaction を行う。steady state で `Runtime::reconfigure` を呼ばない。
2. `apply_eager` は arity、same-runtime、input metadata、effects/aliases、placement/storage を検証し、現在 epoch の snapshot から `(family, selected engine)` の slot を解決する。slot/handle は runtime 内部の opaque な epoch-bound 値とし、semantic payload に provider handle を埋め込まない。reconfigure 後は cheap epoch check で refresh する。
3. native slot があれば、borrowed single-op request から **同じ `ExtensionEngine`** を使って operation-local preparation を行い、同じ `PreparedOperationExecutor` を一つの session/resource scope で実行する。現在すでにある `execute_in_session` はこの用途に使える。call site supplied callback は不要である。
4. native が `Unsupported` のときだけ pure core lowering を試す。lowering 後に executable core op がちょうど一つで、transfer/barrier/global liveness が不要なら core の resolved slotへ直行する。二つ以上、cross-storage、collective、global buffer planning が必要なら初めて `SemanticProgram` / prepared graph に昇格する。
5. forward の経路にかかわらず、eager AD tape には元の semantic extension op を一度だけ記録する。move-only Tensor ownership と AD 整合性は、forward ごとに `SemanticProgram` を作ることではなく、この semantic identity と共通 rule/engine 契約で守る。

最初の実装は既存 `ExtensionEngine::prepare -> PreparedOperationPlan -> PreparedOperationExecutor` を graph artifact なしで直接呼べばよい。ここで生成される macro-generated `Arc<PreparedOperation>` が測定上支配的、または accepted zero-allocation gate に反するなら、`PrepareCapability` の内部表現に borrowed/stack の immediate single-op variant を加える。**別の `EagerExtensionEngine` trait、別の apply API、full-program cacheを増やしてはならない。**

# 外部ユーザーの拡張容易性を守る条件

外部作者が書く source of truth は次の一組だけであるべきだ。

- pure semantic payload/schema (`ExtensionOp`)
- backend ごとの typed execution function
- optional planning/cache と optional AD rule

既存 `define_extension_runtime!` は、module、engine、prepared operation、executor を一つの `execute_reads` から生成しており、方向性は正しい。この仕組みを外部向けに安定した helper/builder として提供し、session fast path も同じ execution function から生成するべきだ。「internal macro を in-tree crate だけが使えるが、外部ユーザーは関連 trait を全部手書き」は大前提違反である。

extension crate の通常の eager wrapper は、op を構築して `apply_eager` を呼ぶだけにする。低遅延を要求しない外部 extension は core lowering だけでも動ける。低遅延を保証したい extension は native engine または guaranteed one-core-op lowering を提供する。この段階差は明示すべきで、generic lowering が自動的に fast だと偽ってはならない。

missing module は process-global discovery で勝手に探さず、core lowering がなければ typed `MissingExtensionEngine` にする。暗黙 CPU fallback や暗黙 transfer は拡張容易性ではなく、再現不能な挙動である。

# fast-path 判定で落としてはいけない edge case

- source op が一個かではなく、eager-local lowering **後の executable op 数**で判定する。
- native N-ary einsum や operation-local layout+GEMM は一 prepared operation なら eligible。内部 kernel 数を数えて graph に昇格させる必要はない。
- multi-output 自体は失格条件ではない。output arity、fresh/view/must-alias、scratch/output storage が operation-local に記述できるかで決める。
- effects があるだけで一律失格にはしないが、resource identity と順序を operation-local に閉じ込められない場合は昇格する。
- provider の実行 error を `Unsupported` に読み替えて別 provider へ retry しない。fallback は prepare 時の capability 判定だけで行う。
- slot を family だけで cache しない。engine、runtime identity、registration identity/epoch が必要である。実行中は snapshot/slot を `Arc` で保持し、並行 reconfigure で provider lifetime を壊さない。
- view input、non-contiguous input、異なる GPU device、mixed storage、nested execution scope、concurrent first-use をテストする。とくに input ingress validation を module factory より先に行う現行の良い性質は維持する。

# 受け入れ条件

同一 shape の cache-hit 一本で合格させてはならない。最低でも cold registration、first execution、same-shape steady state、毎回 shape が変わる列を分離して測る。eligible path では以下を counter/contract test でゼロと確認する。

- `SemanticProgram` freeze と fingerprint
- `GraphCompiler`
- `ScheduledGraph` / global `BufferPlan`
- run admission/event-slot table
- steady-state module configure/reconfigure
- execution時の string dispatch

さらに eager と compiled が同じ engine selection、numeric result、typed error、AD JVP/VJP を示すこと、provider replacement で epoch-bound slot が更新されること、native engine を外すと one-core-op または prepared graph へ正しく移ること、single-op では backend session/resource lease への entry が最大一回であることを gate にする。

要するに、**(a) は重い道をキャッシュで温存し、(b) は裏口を本線にする。どちらも「外部ユーザーが拡張しやすい」という前提を満たさない。** 正解は、一つの semantic/provider 契約を二つのコスト階層で実行する (c) である。# DeepSeek (Cycle 1): (c) に賛成だが GPT 版は「設計文書の再掲＋未解決メカニズム＋スコープ過大」。(c) の最小版へ切り詰めるべき。

# DeepSeek (Cycle 1) — システムズ批判: GPT の (c) は方向正しいが、実装メカニズムとスコープが甘い

## 結論

**(c) に賛成。ただし GPT の版は「設計文書 `execution-engine-provider-architecture.md` の『Eager single-operation cost contract』節のほぼ再掲」であり、肝心の「どう実装するか」で未解決の矛盾と隠れコストを3つ、そしてスコープ爆弾を1つ抱えている。** 採用すべきは (c) の最小版: 単一公開入口 + 呼び出しごと reconfigure の排除 + native engine の resolved-slot 直行。それ以外（single-core-lowering の fast path、provider-local plan cache の一般化、外部向け macro 安定化）は v0.3.0 から切る。

根拠をコード実読で裏付けた。以下、容赦なく。

---

## 1. 最重要の確認事実: 現行 direct bridge は毎呼び出しで write lock + reconfigure を払っている

`crates/tenferro-ad/src/eager.rs` を実読した:

- `install_extension_module` は **毎回** `self.lock_extension_install()`（mutex）→ `self.runtime.reconfigure(...)`（write lock + epoch bump + snapshot publish）を通る。
- `ensure_extension_module_for_engine`（FFT の targeted 経路）も **毎回** `lock_extension_install()` → `reconfigure(...)`。doc コメントは「既登録なら fresh module を inspect しない」と書くが、reconfigure の publish そのものは避けていない。

つまり GPT の step 1（「steady state で reconfigure を呼ばない」）は、fast-path 機構と**直交**する独立のバグ修正であり、**全 caller（generic も direct も）を同時に救う最小・最高インパクトの変更**だ。これを「(c) の一部」として slot 機構と抱き合わせると、肝心の効果が slot 実装の完成を待つことになる。**切るべきは「install/ensure の steady-state read-only check 化」を単独コミットとして先に実装・計測し、その効果を分離して報告すること。** この 1 点だけで 46〜238μs のうちの reconfigure 分が消える可能性が高く、以降の議論の土台が変わる。

## 2. 矛盾1: 「slot はキャッシュではない」という区別は虚構。(a) のキャッシュを2層に分裂させただけ

GPT は (a) の有界キャッシュを「ベンチマークの選別」と切り捨てるが、自身の提案はキャッシュを **2つの key space に分裂**させている:

1. resolved slot: `(family, selected engine, epoch)`
2. provider-local plan cache: `(payload_hash, dtype, shape, …)` — GPT 自身「必要な plan cache は provider-local・bounded に置く」と認めている

つまりキャッシュは消えていない。**whole-program をキャッシュするか、per-op plan をキャッシュするか、の位置の差であり、総機構数は増えている。** そして2層化すると、**invalidation のコヒーレンスが二重になる**:

- epoch が bump して slot が refresh されたとき、provider-local plan cache が古い engine に対する plan を返し続けたら何が起きるか。slot は新 engine を指し、plan は旧 engine 用。→ stale plan を新 engine で実行。
- つまり provider-local cache も epoch-scoped でなければならず、その key と eviction と invalidation を外部作者が「正しく」実装する責任を負う。これは (a) の「generic layer は provider が必要とする key を知らない」という GPT 自身の批判が、そのまま provider 作者の肩に載ることを意味する。

さらに **「guaranteed one-core-op lowering の fast path」は本質的に plan cache** だ。lowering 結果（＝1個の executable core op）を毎回生成せず再利用するにはキャッシュが必要で、それは (a) の下層版に他ならない。**正直に書け: single-core-lowering の fast path は「native-only fast path」より一段重く、(a) 型キャッシュを必要とする。** v0.3.0 では native-only に限定し、single-core-lowering fast path は明示的に将来課題とせよ。でなければ (a) を批判しつつ (a) を再導入する自己矛盾になる。

## 3. 矛盾2: counter-zero gate は必要条件であって十分条件ではない。数値目標が未固定

GPT の受け入れ条件は「SemanticProgram freeze / fingerprint / GraphCompiler / ScheduledGraph / admission / reconfigure / string dispatch の counter がゼロ」という必要十分条件風のリストだが、**これは「重いものを呼んでいない」証明であり、「速い」証明ではない。**

設計文書自身が警告している（同一節）:

- 「9-11 us fixed cost for small calls」が現状の正体であり、固定費は validation / eager bookkeeping / admission / session entry / provider dispatch / output allocation に分散。
- `CpuBackend::install` は 1-thread でも ~6.9μs。slot lookup の 6.31ns とは桁が違う。
- 「Any future eager fixed-overhead child must first separate validation, eager bookkeeping, admission, session entry, provider dispatch, and output allocation **before selecting a mechanism**」

つまり、**全 graph カウンタをゼロにしても、session entry + output allocation + validation が残れば 46μs → 15μs 程度で止まり、「single-op オーバーヘッド削減」の目標は半分しか達成されない。** しかも counter-zero という客観ゲートがあると、15μs でも「合格」に見える罠がある。

**要求: implementation 前に、数値閾値・統計・repetition policy・noisy-run 処理を固定せよ。** これは設計文書が「child issue fixes a non-inferiority statistic, threshold, repetition policy, and noisy-run handling **before implementation results are known**」と明記している既存の義務である。GPT の受け入れ条件は counter-zero に寄りすぎて、この数値固定義務を無視している。さらに cold registration / first execution / same-shape steady state / 毎回 shape 変化、の4系列を分離測定する件は GPT も正しいが、それに加えて **session entry 回数（設計文書: 「single-op では backend session/resource lease への entry が最大一回」）と output allocation 回数を counter に入れる**べき。そこが残る限り速くならない。

## 4. 矛盾3: `execute_in_session` が外部作者への「第3の負担」であり、しかもサイレントな性能断崖

実読: `PreparedOperationExecutor`（capability.rs）は `execute`（ErasedExecutionContext 版）と `execute_in_session`（borrowed session 版、**default は Unsupported**、`supports_session` default は `false`）の **2つの実行入口**を持つ。macro `define_extension_runtime!` の `execute_in_session` フィールドは **optional** で、現状それを供給しているのは first-party（linalg/einsum/fft）だけ。

GPT の「外部作者の source of truth は一組（semantic payload + typed execution function + optional planning + optional AD）」は、この実態に対して **楽観的すぎる**。現実には:

- 外部作者が `execute_in_session` を書かなければ、native engine を提供しても **fast path に入れず、黙って generic 経路（46〜238μs）に落ちる**。エラーも警告も出ない。作者は自分の native engine が fast だと思い込む。
- `supports_session()` は runtime bool で、CPU では動く session 実行が CUDA では `Unsupported` を返し得る（設計文書: 「never silently retried from inside a session」なので、その場合 **実行時エラー** になる）。このバックエンド依存 failure mode を、GPU を持たない外部作者はテストできない。
- つまり「一つの実行関数」は実質 **2 つ** で、片方を欠いてもコンパイルは通るが、性能と挙動がバックエンドごとに非再現的に変わる。これは「大前提: 拡張容易」に対する最も深刻な違反候補。

**要求:** fast path への参加は **opt-in かつ loud** にせよ。たとえば (i) `supports_session` を宣言したら `execute_in_session` の実装を macro が強制する（欠けはコンパイルエラー）、(ii) fast-path eligible でない外部 op は診断/ログで「promoted to prepared-graph（session execution 未提供）」を一度は表示する、のいずれか。サイレントな性能断崖は拡張容易性ではなく罠である（GPT 自身が「generic lowering が自動的に fast だと偽るな」と言った。同じ批判が native engine の session 実行にも当てはまる）。

## 5. 矛盾4（健全性）: borrowed/stack PreparedOperation と reconfigure 安全のための Arc 保持は両立しない

GPT は「borrowed single-op request」「必要なら PrepareCapability に borrowed/stack の immediate single-op variant を加える」と書く一方、「実行中は snapshot/slot を Arc で保持し、並行 reconfigure で provider lifetime を壊さない」と書く。**この2つは直接矛盾する:**

- stack/borrowed な executor は、実行中の並行 `reconfigure`（＝engine/executor の retirement）に対して lifetime を守れない。move-only Tensor ownership（#1555）と、executor 内部での再入（nested eager op、実行中の cache insert）が絡むと、borrowed 値を間接的に保持する経路が生まれ得る。
- 正しい解は **Arc snapshot から borrow** すること。snapshot の Arc が call の間 outlive するのだから、borrowed なリクエストは snapshot から借りれば lifetime が足り、stack variant は不要。

**つまり stack/borrowed variant は YAGNI。** Arc snapshot + borrow が最も単純で、reconfigure 安全と borrowed リクエスト（設計文書の 7ns プロトタイプ）を同時に満たす。GPT の「別の EagerExtensionEngine trait、別の apply API、full-program cache を増やすな」という自制は正しいが、同じ自制を「stack immediate variant」にも適用せよ。

## 6. 矛盾5（探針コスト）: eligibility 判定に lowering を走らせるのは間違い

GPT step 4「native が Unsupported のときだけ pure core lowering を試す。lowering 後に executable core op がちょうど一つで…」は、**eligibility 判定のたびに lowering を実行する**ように読める。これには二つの致命的問題:

- 非 eligible な op（＝graph へ昇格する op）は、**probe の lowering を払った上で** full pipeline を払う。つまり generic 経路がむしろ**遅くなる**。設計文書の受け入れ条件「no new microsecond-scale orchestration step」に違反。
- single-core-lowering が eligible な op は、毎呼び出し lowering を払う → 決して fast にならない（lowering 自体が allocation を伴い得る）。

**eligibility は「宣言された静的 capability bit」で判定せよ。** すなわち `is_native_single_op` / `guaranteed_single_core_lowering` を作者が宣言し、判定はビット参照のみ。lowering を「数えるため」に走らせるのは禁止。設計文書の「guaranteed single-core-operation lowering」という語はこの宣言を意味しており、GPT の step 4 の書き方はそれと食い違う。

## 7. スコープ爆弾: 外部向け `define_extension_runtime!` 安定化は v0.3.0 に載せてはならない

GPT は「この仕組みを外部向けに安定した helper/builder として提供し」と書く。だが実態は:

- crate 名が `tenferro-internal-extension-macros`。**文字通り internal** の proc-macro を外部向けに安定化するのは、semver コミット + hygiene + エラーメッセージ + GPU/CUDA カバレッジ + ドキュメント doctest の一大作業であり、小さなリファクタではない。
- 「大前提: 外部ユーザーが拡張しやすい」の成果物は **trait 面 + 非マクロ helper + 動く一例 + ドキュメント**であって、**macro の公開ではない**。外部作者は `ExtensionEngine` + `PreparedOperationExecutor`（+ optional `execute_in_session`）を手実装できれば拡張できる。macro はそれを楽にするだけで、正しさの前提ではない。

**要求:** v0.3.0 では trait 面（と `define_extension_runtime!` の session variant の強制化）を整え、macro の外部公開は issue に切り出して defer。拡張容易性を「macro 公開」と同一視しないこと。

## 8. 受け入れ条件の追加: 経路選択の observability

設計文書は placement 選択の observability（「chosen domain and reason are observable」）を要求しているが、**fast vs promoted の経路選択の observability は明示要求されていない**。外部 author が「自分の op がどの経路を通ったか」を診断できないと、第4節のサイレント性能断崖が不可視のまま残る。**受け入れ条件に「各 eager 呼び出しの経路（native-slot / promoted-graph）とその理由が diagnostics で観測可能」を追加せよ。**

## まとめ: 採用すべき (c) の最小版

v0.3.0 のスコープを以下に切り詰める:

1. **install/ensure の steady-state read-only no-op 化**（独立コミット、独立計測）— 全 caller を救う最高インパクト。
2. `apply_eager` 単一公開入口。call site が callback を渡す公開経路（`apply_eager_with_extension_session` 系）は廃止。
3. native engine の resolved slot 直行。Arc snapshot から borrow、新 trait なし、stack variant なし。
4. eligibility は宣言ビット。**single-core-lowering fast path と provider-local plan cache の一般化は defer**（＝(a) 型キャッシュの再導入を避ける）。
5. `execute_in_session` は opt-in かつ loud（macro 強制 or 診断）。
6. 数値閾値を implementation 前に固定し、session entry / output allocation の counter も gate に含める。

これが (c) の ponytail 版。GPT の版は方向は正しいが、設計文書の再掲に留まり、未解決の健全性矛盾（第5節）、サイレント性能断崖（第4節）、そして「(a) を批判しつつ (a) を再導入する」自己矛盾（第2節）を含む。スコープを切れば、矛盾は消える。
# GPT (Cycle 2): DeepSeek の縮小案は半分正しいが、静的 eligibility と trait 手実装は大前提を壊す

# GPT (Cycle 2): DeepSeek の縮小案は半分正しいが、静的 eligibility と trait 手実装は大前提を壊す

## 結論

DeepSeek の批判のうち、**変更順序の分離、v0.3.0 を native-only に絞ること、stack variant を入れないこと、数値ゲートを実装前に固定すること**には譲歩する。Cycle 1 の私案は、将来の one-core lowering と macro 公開まで同じ束に入れた点で広すぎた。

しかし DeepSeek の「最小版」をそのまま採るのは危険だ。とくに次の三点は退ける。

1. `install_extension_module` 全般を steady-state no-op にできるという前提は、**replace の公開意味論を無視している**。
2. eligibility を作者の静的ビットにする案は、**payload・dtype・layout・placement・backend で変わる capability を嘘の一ビットへ潰す**。
3. `execute_in_session` の有無を fast path 参加条件にする案は、**外部作者に二本目の executor を強制し、大前提に逆行する**。

採るべき v0.3.0 は、キャッシュ付き slot ではなく、**現在の immutable snapshot から engine handle を直接解決し、既存 `ExtensionEngine::prepare` の結果を graph artifact なしで即時実行する native-only path**である。`execute_in_session` は最適化であって参加資格ではない。

## 1. install/reconfigure の先行修正には賛成。ただし DeepSeek は API 意味論を雑に扱っている

独立コミットを先に置く順序は正しい。だが「install/ensure の steady-state read-only no-op 化」と一括りにするのは誤りである。

実コードの `EagerRuntime::install_extension_module` は名前どおり **install or replace** で、内部でも `replace_extension_module` を呼ぶ。同じ module ID の新しい `Arc` が来たとき、それが単なる再生成なのか、意図した provider replacement なのかを runtime は判別できない。同一 ID だから黙って捨てれば、reconfigure と epoch の契約を壊す。DeepSeek はここを「独立のバグ修正」と断定したが、実際には無条件で直すと意味論変更である。

一方、`ensure_extension_module_for_engine` は文書上も「同じ module ID・family・engine の既登録を再利用する」契約なので、こちらは安全に二段階化できる。

- current snapshot で exact registration を read-only check
- hit なら現在 epoch を返す
- miss なら install mutex を取り、snapshot を再確認してから一度だけ reconfigure

したがって先行コミットは **targeted ensure の double-checked no-op 化**と、first-party eager wrapper の hot path を `install` から `ensure` へ移す変更に限るべきだ。replacement API 自体は write path のまま残す。

また「全 caller（generic も direct も）を同時に救う」は事実ではない。generic `apply_eager` は `exec_outputs_read` を呼び、module を引数に取らない。per-call install を直接払っているのは module を持ち込む bridge 側である。46〜238μs の大半が reconfigure かもしれない、という推測も計測前には断定できない。独立計測に賛成する理由は、まさにその憶測を排除するためだ。

## 2. 「slot はキャッシュの分裂」は category error。ただし別 slot cache は捨てる

DeepSeek は resolved slot と provider-local plan cache を二層キャッシュと呼ぶが、engine handle と algorithm plan は同じ種類のものではない。

- engine handle は現在 snapshot に既に存在する dispatch state
- plan cache は payload/shape 等から得る高価な計算結果の再利用

前者を snapshot から一度読むことは、後者の cache key・eviction・specialization を増やさない。両者を「どちらも何かを保持するからキャッシュ」と呼ぶなら、vtable も registry map も全部キャッシュになり、議論の区別が消える。

ただし DeepSeek の批判から、Cycle 1 の **EagerRuntime 内に別の epoch-bound slot cache を持つ案は撤回する**。snapshot lookup が既報どおり ns 級なら、さらに cache する理由はない。各 call は `Arc<RuntimeSnapshot>` を一つ保持し、その snapshot 内の selected engine handle を解決すればよい。これなら slot refresh/invalidation という第二機構そのものがない。

provider-local plan cache の一般化も v0.3.0 では追加しない。既存 provider が既存 cache contract を使うことまで禁止する必要はないが、新 fast path の成立条件にはしない。実行 plan/executor は、それを作った engine を保持する同じ snapshot の lifetime 内だけで使う。旧 engine の plan を新 engine に渡す設計ではない。DeepSeek の stale-plan 例は、plan と executor の所有者を切り離すという未提案の実装を勝手に仮定した反論である。

## 3. one-core lowering は defer する。しかし「本質的に (a) 型 cache」は誤り

v0.3.0 を native engine のみに絞り、single-core-lowering fast path を後続 issue に出す点は受け入れる。unsupported op に lowering probe と full graph の両方を払わせないためである。

ただし「one-core lowering は本質的に plan cache」という DeepSeek の断言は成り立たない。安価な declarative lowering が core op を一個構築して即時 dispatch するなら cache は不要である。必要なのは lowering cost の計測と上限であって、cache の存在ではない。今回はスコープから切るが、それは (a) と同一だからではなく、v0.3.0 で未計測の probe を増やさないためだ。

## 4. eligibility を静的 capability bit にする案は却下する

ここが DeepSeek 案の最大の欠陥である。single-op eligibility は family の定数ではない。

- 同じ linalg family でも op variant で異なる
- 同じ op でも CPU/CUDA/WebGPU で異なる
- dtype、shape、stride、storage、device、alias/effect 条件で異なる
- provider replacement 後には同じ family でも能力が変わる

実際、現行 `linalg_session_supported` 自体が op と backend type を見ている。これを作者申告の `is_native_single_op` 一ビットに潰せば、false は利用可能な fast path を失い、true は実行時 failure を作る。さらに bit と `prepare` の結果が二つの source of truth になり、外部作者が同期させる責任だけが増える。

native-only なら lowering を eligibility probe に使う必要はない。手順は単純である。

1. snapshot から family/selected engine を解決する。
2. borrowed `ExtensionPrepareRequest` で既存 `ExtensionEngine::prepare` を一度呼ぶ。
3. `Prepared` なら、その executor を graph artifact なしで即時実行する。
4. prepare-time `Unsupported` なら prepared graph へ昇格する。
5. provider contract error や execution errorは fallback に読み替えない。

`prepare` はどうせ executable を得るために必要であり、静的ビットを先に読むことで supported path の仕事は一つも減らない。必要なら registration 側に「immediate prepare を提供し得る」という粗い prefilter を置けるが、それは候補絞り込みにすぎず、eligibility の最終判定ではない。

## 5. `execute_in_session` を fast path の参加資格にしてはならない

DeepSeek は「外部作者が `execute_in_session` を書かなければ generic 46〜238μs に落ちる」と主張した。これは現行 bridge の観察を新 fast path へそのまま投影しただけで、設計上の必然ではない。

`PreparedOperationExecutor` には既に通常の `execute` が必須である。native prepare が成功したら、runtime immediate path は次のように実行できる。

- `supports_session() == true`: runtime-owned session を一度だけ借り、`execute_in_session`
- `supports_session() == false`: 同じ prepared executor の必須 `execute` を runtime-owned backend context で直接実行

後者も `SemanticProgram`、fingerprint、compiler、schedule、admission を通らない。session-aware override より固定費が高い可能性はあるが、「full generic path へのサイレント転落」ではない。これにより外部作者は **必須の typed execution を一本実装すれば native immediate path に参加できる**。最低遅延が必要な作者だけ session override を追加する。

しかも現行 macro は `session_supported` と `execute_in_session` を片方だけ指定すると既に compile error にしている。DeepSeek の「macro が強制せよ」は半分実装済みの要求である。手書き trait 実装が `supports_session=true` と default `execute_in_session` を矛盾させる余地は contract test で塞ぐべきだが、そのために全作者へ二本目を強制するのは筋が悪い。

backend-specific `Unsupported` を session 内で返した場合は retry しない、という Cycle 1 の条件は維持する。`supports_session` は実際の op/backend に対して保守的でなければならない。

## 6. stack variant は撤回する。ただし「健全性矛盾」という攻撃は誇張である

v0.3.0 では stack/immediate variant を加えない。`Arc<RuntimeSnapshot>` を call 中保持し、そこから engine/executor を借りる。allocation が計測上の支配項になった場合だけ後続 issue で扱う。ここは DeepSeek の YAGNI 指摘を受け入れる。

ただし「borrowed request と reconfigure safety は両立しない」は間違いである。snapshot の `Arc` が call を outlive し、borrow が call から escape しない API なら Rust の lifetime が保証する。DeepSeek は「nested eager が間接保持し得る」と言うが、その escape を許す具体的シグネチャを示していない。撤回理由は soundness 不可能だからではなく、**まだ必要性が測定されていないから**である。

## 7. 数値ゲートは追加する。ただし DeepSeek の 15μs 論は作り話である

counter-zero は構造的必要条件にすぎず、数値性能条件も必要、という批判は正しい。実装順は次にする。

1. 現行 generic/direct を cold、first、steady same-shape、changing-shape に分けて測る。
2. targeted ensure no-op の独立コミットを測る。
3. その結果を公開したうえで、fast-path 実装前に post-ensure direct bridge を non-inferiority baseline とし、統計量・許容差・反復数・noisy-run 除外規則を固定する。
4. native `apply_eager` は direct baseline に非劣性、generic fallback は非退行、という二つを gate にする。

同時に structural counter で graph artifact がゼロ、session/resource entry が最大一回であることを確認する。output allocation も回数と bytes を観測するが、目標を無条件にゼロにはしない。fresh output を返す op で必要な allocation まで消せというのは不可能であり、view/must-alias/lazy output と混同してはならない。目標は operation contract が要求する以上の allocation をしないことだ。

DeepSeek の「46μs が15μsで止まり、目標の半分」という数字には根拠がない。session entry と allocation の内訳を分離計測せよという主張には賛成するが、未計測の 15μs を既成事実として設計判断に使うのは、自ら要求した measurement-first に反する。

## 8. macro 公開は defer してよい。しかし trait 手実装を「拡張容易」と呼ぶな

internal proc-macro の semver 安定化を v0.3.0 に抱き合わせない点は譲歩する。ただし DeepSeek の「外部作者は `ExtensionEngine` + `PreparedOperationExecutor` を手実装できれば拡張できる」は、大前提を最低水準へすり替えている。

現行コード自身が `PreparedOperationExecutor` を `#[doc(hidden)]` 相当の internal bridge と説明している。外部作者に module、planning config、engine、prepared metadata、executor、registration identity を手で配線させ、「理論上可能だから easy」と言うのは通らない。

v0.3.0 の受け入れ条件には、proc-macro 公開の代わりに次を入れる。

- stable な小さい非 macro adapter/helper
- semantic op と typed execution function 一本から module/engine/prepared executor を組めること
- out-of-tree fixture crate が stable public API のみで eager immediate と compiled の両方を通ること
- 同じ fixture で typed error と AD rule の一致を確認すること

helper の具体形を巨大な framework にする必要はない。しかし「五つの trait を手実装してください」は外部拡張容易性ではない。macro 安定化は defer できても、**ergonomic な stable path は defer できない**。

## 9. observability は賛成。ただし hot call の常時ログは拒否する

`native-session`、`native-context`、`promoted-graph(reason)` の区別を test counter または opt-in tracing で観測可能にする。外部作者向け診断は必要である。

一方、各 eager call で無条件に文字列診断を生成すれば、削ろうとしている fixed overhead を自分で戻す。steady-state string dispatch zero の条件は維持し、observability は disabled 時ほぼゼロコストの hook にする。

## v0.3.0 の修正版スコープ

コミット境界も含め、以下で決着させるべきだ。

1. **計測基盤と現行 baseline**を先に固定。
2. **targeted ensure の read-only steady-state化**を独立コミット。generic replace semantics は変えない。
3. `apply_eager` 内に **snapshot-resolved native immediate path**を追加。別 slot cache、新 trait、static eligibility bit、one-core lowering、general plan cache は追加しない。
4. `Prepared` は `execute_in_session` または必須 `execute` で即時実行し、`Unsupported` のみ graph へ昇格。
5. first-party wrapper を単一 `apply_eager` へ移し、call-site callback bridge を公開経路から外す。
6. stable adapter + out-of-tree fixture を提供。internal macro の公開安定化は別 issue。
7. 数値非劣性、graph counter zero、session entry 最大一回、余分な allocation zero、意味論・typed error・AD 一致を gate にする。

要するに、DeepSeek のスコープ削減は有益だが、**静的宣言ビットと session executor 必須化は「作者に責任を押しつけて runtime を簡単に見せる」だけ**である。それは ponytail ではない。外部作者が一本の execution source of truth で参加でき、runtime が動的 capability を正しく判定する最小設計こそ、譲れない前提を満たす。# DeepSeek (Cycle 2): コード実読で自分のCycle 1の誤りを3点認め、GPT案に収束。残る論点は「native-context/session の分離計測」と「promoted 経路の prepare() 早出し bound」の2つの gate だけ。

# DeepSeek (Cycle 2) — コード実読で自分の誤りを認め、GPT 案に収束する。残る論点は gate 2 本のみ。

## 結論

Cycle 1 の私の攻撃のうち、**コード実読で誤りが判明したもの**を先に撤回する。撤回後に残る実質的な不一致は、(i) promoted 経路の `prepare()` コスト上限、(ii) `native-context` と `native-session` の分離計測、の 2 点である。これらは「反対」ではなく「gate の追加要求」であり、GPT の Cycle 2 スコープにそのまま載る。**デベートは実質収束した。**

以下、撤回 → 収束の確認 → 残る gate 2 本、の順。

---

## 1. 撤回その1: 「install/ensure は毎回 epoch bump + publish を払う」は誤り。実際のコストは `CandidateConfig::from_snapshot` の全クローン

`crates/tenferro-runtime/src/runtime/snapshot.rs` を実読した。`Runtime::reconfigure` は **既に** `changed == false` のとき `return Ok(base.epoch())` で write lock・epoch bump・snapshot publish を**全部スキップ**する（snapshot.rs:1040 付近）。つまり私の Cycle 1 第1節の「write lock + epoch bump + snapshot publish」は**事実誤認**だった。GPT の「DeepSeek は API 意味論を雑に扱っている」は正しい。

しかし実読で、私の「毎呼び出しに無駄がある」という**直感自体は別の場所に正しく着地した**。真の残コストは:

1. `EagerRuntime::lock_extension_install()` — 毎回の mutex（eager.rs:1024）
2. `reconfigure` 冒頭の `snapshot()`（read lock）
3. **`CandidateConfig::from_snapshot(&base)` — engines / modules / transfers の BTreeMap を全部クローン**（snapshot.rs:99 の `from_snapshot` は `engines.iter().map(...)` で全レコードを再構築）
4. edit（`ensure_extension_module_for_engine` の no-op check）→ `changed == false` → early return

つまり GPT の「targeted ensure の read-only no-op 化」の正体は、**epoch bump の回避ではなく、`CandidateConfig::from_snapshot` の全クローンと install mutex の回避**である。これは GPT の提案を**より強く動機づける**事実だ。GPT の二段階化（current snapshot で `(module_id, family_id, engine_id)` を read-only check → hit なら即 return）は、公開済み snapshot が既に `Arc` なのでクローン不要で読める。**このコミットに賛成。install-or-replace の `install_extension_module` を無条件 no-op 化しない、という GPT の制限も正しい**（eager.rs:1391 は確かに `edit.replace_extension_module(module)` を呼ぶ replace 意味論）。

唯一の付言: generic `apply_eager` は module を持ち込まない（`exec_outputs_read` 経由）という GPT の観察も正しい。したがってこのコミットは「generic も direct も同時に救う」のではなく、**bridge 側 hot path を救う**コミットである。私の Cycle 1 の「全 caller を同時に救う」は撤回する。

---

## 2. 撤回その2: 数値「15μs」は作り話。撤回する。ただし measurement-first の原則は残す

未計測の 15μs を設計判断に使ったのは、私自身が要求した measurement-first に反する。**撤回。** ただし「counter-zero は必要条件であって十分条件ではない」という構造的主張は GPT も採用した（Cycle 2 第7節）。ここは収束済み。

---

## 3. 撤回その3: `execute_in_session` 欠如 =「generic 46〜238μs へのサイレント転落」は誤り。ただし gate は分離せよ

GPT の指摘は正しい。`PreparedOperationExecutor`（capability.rs:285）には**必須の `execute(&mut ErasedExecutionContext, &mut ExtensionCacheStore, &[TensorRead])`** があり、`execute_in_session` / `supports_session` は default `false` / `Unsupported` の**optional**である。native prepare 成功後の immediate path は:

- `supports_session() == true` → `execute_in_session`
- `false` → 必須 `execute` を runtime-owned erased context で直接実行

**どちらも `SemanticProgram` / fingerprint / `GraphCompiler` / `ScheduledGraph` / admission を通らない。** したがって「session override を書かない外部作者は generic 経路へ落ちる」は誤りで、外部作者は**必須の typed execution 一本で native immediate path に参加できる**。macro が `execute_in_session` と `session_supported` を片方だけ指定すると compile error にする件も実読で確認した（`tenferro-internal-extension-macros/src/lib.rs:216-220`「must be supplied together」）。GPT の「半分実装済みの要求」も正しい。

**ここは収束。ただし gate を一つ追加要求する（後述の残論点2）。**

---

## 4. 残論点1: promoted 経路の `prepare()` コスト上限を gate にせよ

GPT の step「snapshot から family/engine を解決 → `ExtensionEngine::prepare` を一度呼ぶ → `Prepared` なら即時実行、`Unsupported` なら graph 昇格」は、**eligibility 判定と仕事が同じ呼び出しで一体化している**点で、私の「static bit」案より良い。私の Cycle 1 の「lowering を probe に走らせるな」という批判は、native-only スコープでは**空振り**だった（lowering を走らせる案は GPT が既に defer した）。static bit 案は撤回する。

ただし構造的な懸念が**一つだけ残る**: `prepare()` が `Unsupported` を返す**までのコスト**だ。もし provider の `prepare` が「capability 判定より前に高価な planning 仕事（allocation、plan 構築）をしてから `Unsupported` を返す」実装なら、promoted 経路（＝非 native op）は **`prepare()` の分 + full graph pipeline の分**を払い、generic が**むしろ遅くなる**。これは GPT 自身の Cycle 1 受け入れ条件「no new microsecond-scale orchestration step」に違反するリスク。

**要求:** gate に以下を追加する。

- `prepare()` の `Unsupported` は planning 仕事より**前**に（capability 判定だけで）返されること。契約テストで、`Unsupported` を返す provider の `prepare` が allocation をしないことを確認する。
- promoted 経路（`Unsupported` → graph 昇格）の**非退行**を、native 経路とは**別系列**で計測する。GPT の gate「generic fallback は非退行」をそのまま使うが、「`prepare()` を試した後の fallback」と「最初から generic の fallback」を同一視しないこと。

GPT が述べた「registration 側の粗い prefilter」は、このコストが測定で顕在化した場合にのみ追加すればよい。**デフォルトでは入れない**（YAGNI）。入れても「候補絞り込みの hint」であって決定はしない、という GPT の位置づけに賛成する。これで「bit と prepare の二重 source of truth」問題も消える（hint は決定しないから）。

---

## 5. 残論点2: `native-context`（必須 `execute`）と `native-session` は分離して計測せよ

GPT の「外部作者は必須 executor 一本で参加できる」は**構造的には正しい**が、**性能的には未計測**である。GPT 自身「session-aware override より固定費が高い可能性はある」と認めている。ここに gate の穴がある:

GPT の gate は「native `apply_eager` は direct baseline に非劣性」と書くが、**first-party の linalg/einsum/fft は `supports_session == true`** なので、この gate は**事実上 `native-session` 経路だけを測って合格し得る**。外部作者が書く「`execute` だけ実装、session override なし」の op は `native-context` 経路を通り、その固定費は別物である。**first-party が session で合格し、外部 op が context 経路で黙って遅い**、という私が Cycle 1 で警告した断崖が、gate の作りによっては温存される。

**要求:** gate を二系列に分けよ。

1. `native-session`（`supports_session == true`）: first-party が通る経路。
2. `native-context`（必須 `execute` のみ、`supports_session == false`）: session override を書かない外部 op が通る経路。**これも独立に direct baseline に対して非劣性を確認する。**

さらに `execute` のシグネチャが要求する固定費を内訳で測ること。`execute` は `&mut ErasedExecutionContext` **と** `&mut ExtensionCacheStore` の両方を要求する（capability.rs:293-298）。つまり immediate path は (a) backend context の取得（既存 `exec_outputs_with_runtime` の "lock_backend" 相当）、(b) `ExtensionCacheStore` の供給、(c) output allocation、の三つを必ず払う。`ErasedExecutionContext::new` 自体は型消去ラッパで安い（capability.rs:590）が、**その中身である backend context の取得コスト**が floor を決める。これが「one executor is enough」の性能面の正体であり、計測前には何μs か言えない。**計測し、non-inferiority を `native-context` でも取るまで「外部作者は一本で fast」と謳ってはならない。**

---

## 6. 収束の確認（合意スコープ）

以下は合意とみなす:

1. 計測基盤と現行 baseline を先に固定（cold / first / steady same-shape / changing-shape の4系列）。
2. **targeted ensure の read-only steady-state 化**を独立コミット（install-or-replace の意味論は不変）。真の対象は `CandidateConfig::from_snapshot` 全クローン + install mutex の回避。
3. `apply_eager` に snapshot-resolved native immediate path。別 slot cache・新 trait・static eligibility bit・one-core lowering・general plan cache は**入れない**。
4. `Prepared` は `execute_in_session` または必須 `execute` で即時実行、`Unsupported` のみ graph 昇格。
5. first-party wrapper を単一 `apply_eager` へ移し、call-site callback bridge を公開経路から外す。
6. **stable adapter + out-of-tree fixture** を提供。internal macro の公開安定化は別 issue。

第6項の「trait 手実装で拡張容易」という私の Cycle 1 の書き方は**撤回する**。実コードも `PreparedOperationExecutor` を「public trait ではあるが internal bridge（`#[doc(hidden)]`、capability.rs:280-284）」と明記しており、外部作者に module / planning config / engine / prepared metadata / executor / registration identity を手配線させて「理論上可能だから easy」と言うのは大前提違反という GPT の指摘が正しい。**out-of-tree fixture crate が stable public API のみで eager immediate と compiled の両方を通る**ことを受け入れ条件にする方が、私の「拡張しやすい」より強く、かつ検証可能である。賛成する。adapter は「小さく非 macro」に留める（ponytail: framework 化しない）。

---

## 7. 最終 gate リスト（私の追加を太字）

1. 数値非劣性: **`native-session` と `native-context` の二系列それぞれ**を direct baseline に非劣性。generic fallback は非退行（**`prepare()` 試行後の promoted も分離**）。
2. structural counter: graph artifact zero、session/resource entry 最大一回、**`ExtensionCacheStore` 供給と backend context 取得の回数/コストを内訳で観測**。
3. **`Unsupported` を返す `prepare` が allocation をしない（capability 判定が planning より先）**ことを契約テスト。
4. 余分な allocation zero（view/must-alias の混同はしない、という GPT の条件を維持）。
5. 意味論・typed error・AD JVP/VJP の eager/compiled 一致、out-of-tree fixture で両経路通過。
6. observability: `native-session` / `native-context` / `promoted-graph(reason)` を opt-in / disabled 時ほぼゼロコストで。

---

## まとめ

GPT の Cycle 2 は私の誤認3点（epoch bump の実態、session 欠如＝generic 転落、trait 手実装＝拡張容易）をコードで正した。私はそれを実読で確認し、撤回する。**方向と機構は GPT 案に収束。** 残るのは「gate の作り」だけであり、それは反論ではなく追加要求である: (1) promoted 経路の `prepare()` 早出し bound、(2) `native-context` と `native-session` の分離非劣性。この2本を gate に入れれば、v0.3.0 の設計は私の側から異論がない。
# User (Feedback): AD不要・concrete計算のみの外部ユーザーへの実装容易性を問う

## User Feedback

外部利用者は AD は不要で、eager による TypedTensor / Tensor による計算だけがしたいとする。そのときにどのように実装しやすくするのか？

つまり、semantic payload（`ExtensionOp`）や AD rule が不要な、concrete 計算のみの外部ユーザーにとって、extension 追加の最小の実装経路はどうあるべきか。これを「大前提: 外部ユーザーが機能を拡張しやすいこと」の具体ケースとして議論せよ。
# GPT (Cycle 3): concrete-only を runtime extension に押し込むな――最小経路と昇格経路を分離せよ

# GPT (Cycle 3): concrete-only を runtime extension に押し込むな――最小経路と昇格経路を分離せよ

## 結論

新しいフィードバックへの答えは明快である。

- **AD rule は concrete forward 計算には一切必要ない。**
- **`ExtensionOp` も `Tensor` / `TypedTensor` を caller-owned backend で直接計算するだけなら必要ない。**
- **`ExtensionEngine` と `PreparedOperationExecutor` も不要である。**
- したがって、収束した `ExtensionOp + ExtensionEngine + PreparedOperationExecutor` を全外部作者の最小入口に据えるなら、それは明白に過剰であり、「外部ユーザーが拡張しやすい」という大前提への違反である。

ただし、これは Cycle 2 の収束を覆さない。収束した設計は **runtime に統合される extension**、すなわち `EagerTensor`、traced/compiled、provider 選択、runtime-owned cache、reconfigure を必要とする層の正解である。間違いは、それを concrete-only 層まで強制することだ。

要するに、**最小 concrete 経路と runtime 統合経路は能力レベルが違う。前者を後者の劣化版として実装してはならない。**

## 1. まず「eager」の語を混同するな

ここで最も危険な論理的すり替えは、次の二つを同一視することだ。

1. `Tensor` / `TypedTensor` を受け取り、その場で backend を呼ぶ concrete immediate execution
2. `tenferro-ad` の `EagerTensor` を受け取り、runtime identity と将来の tape/AD 変換可能性を保持する eager execution

前者は普通の backend-explicit Rust API である。後者は semantic/runtime object の API である。`apply_eager` の二経路統一が対象にしているのは後者だ。前者まで `apply_eager` に通せば「統一」ではなく、**registry、snapshot、型消去、provider 準備という不要な税を concrete 利用者へ転嫁しただけ**になる。

「single `apply_eager` が公開入口だから concrete もそこへ統一すべきだ」という反論は category error である。単一化すべきなのは、同じ `EagerTensor` 意味論に対する generic path と session bridge であって、異なる所有・能力契約を持つ `Tensor` API までではない。

## 2. concrete-only の最小実装経路

外部作者が既存 backend primitive を組み合わせて新しい concrete 演算を作るだけなら、必要なのは次だけでよい。

1. 普通の config/引数型。軸や係数を Rust の引数として受ける。payload hash、`family_id`、`clone_arc`、symbolic shape inference は不要。
2. 一つの typed execution function。概念的には `execute_foo<T, B: TensorBackend>(..., backend: &mut B) -> Result<TypedTensor<T>>` でよい。
3. 必要なら `Tensor` / `TypedTensor` 上の小さい extension trait。dtype-erased `Tensor` 側は外側で一度だけ dispatch し、同じ typed body を呼ぶ。
4. application が caller-owned backend を渡す。暗黙 CPU fallback、暗黙 transfer、process-global registration はしない。

依存は原則 `tenferro-tensor` のみでよい。実行例・テスト用に `tenferro-cpu` を dev-dependency に置けば足りる。`tenferro-runtime`、`tenferro-ad`、`tenferro-ops`、internal macro を要求する理由はない。

演算が既存 `TensorBackend` primitive の合成ではなく、新しい leaf kernel を要求するなら、作者自身の crate に `FooBackend` のような**狭い operation-specific capability trait**を置き、対応 backend にだけ実装すればよい。巨大な `TensorBackend` 一式を実装させてはならない。既存の `LinalgBackend` や `FftBackend` がこの形の先例である。GPU stream や private buffer への外部アクセスが足りないなら、それは backend leaf extension API の不足であって、`ExtensionOp` を書かせても解決しない。semantic registration と kernel extensibility を混同するな。

高価な planning が本当に必要な演算だけ、caller-owned `ConcreteFooPlan` や bounded cache を追加すればよい。単発演算に最初から module、epoch、`ExtensionCacheStore` を持ち込むのは YAGNI である。

## 3. 三段階の能力契約を明示せよ

| レベル | 作者が実装するもの | 得られるもの | 不要なもの |
|---|---|---|---|
| Concrete-only | typed executor、任意の `Tensor`/`TypedTensor` wrapper、必要なら狭い backend trait | backend-explicit immediate 計算 | `ExtensionOp`、module、engine、prepared bridge、AD |
| Runtime-integrated / no AD | 上記に semantic `ExtensionOp` と stable runtime adapter を追加 | unified `apply_eager`、traced/compiled、provider selection、runtime cache/reconfigure | AD rule |
| Runtime-integrated + AD | 上記に optional semantic AD rule を追加 | JVP/VJP/linearize/transpose | forward executor の再実装 |

AD を第三段へ独立させることが重要である。`ExtensionOp` は AD の同義語ではない。runtime では payload identity、symbolic output metadata、alias/effect、provider dispatch のために必要だが、AD rule は forward execution と独立である。AD rule がない runtime extension は forward eager/compiled を正常に実行し、実際に微分を要求された時だけ typed `Unsupported` / missing-rule error を返すべきだ。ゼロ勾配や数値微分への暗黙 fallback は論外である。

逆に concrete-only 呼び出しでは config は call stack にあり、入力 shape/dtype は既に concrete、backend も caller が指定している。そこへ semantic identity を要求するのは、使わない graph 能力の前払いにすぎない。

## 4. 二層の関係は「同じ kernel source of truth」で結べ

二層を分けると実装が二重化する、という反論も弱い。二重化してよいと言っているのではない。関係は一方向にする。

```text
Tensor / TypedTensor public wrapper ───────────────┐
                                                   ├─> one concrete typed executor
EagerTensor apply_eager -> ExtensionOp -> engine ─┘
Traced/compiled -> same engine/prepared adapter ──┘
```

runtime adapter は concrete executor を包む。kernel、validation、dtype semantics を再実装しない。`PreparedOperationExecutor` は runtime/provider 間の owner-scoped bridge のまま隠し、Cycle 2 で合意した stable adapter が semantic op と concrete execution function から生成・配線する。外部作者へ bridge trait の手実装を要求してはならない。

この形なら concrete-only から runtime 統合への昇格は追加であり、書き直しではない。

- 最初は `execute_foo` だけを書く。
- traced または `EagerTensor` が必要になった時だけ `FooOp { config }` を semantic payload にする。
- engine/prepared adapter は同じ `execute_foo` を呼ぶ。
- 微分が必要になった時だけ AD feature と rule を足す。

「将来 AD が欲しくなるかもしれないから最初から `ExtensionOp` を書け」は、典型的な speculative abstraction である。将来は必要になった時に昇格すればよい。

## 5. 現行コード自身がこの分離を既に証明している

これは机上の新設計ではない。`docs/design/einsum.md` は concrete non-AD と autodiff eager を明示的に分け、`crates/tenferro-einsum/src/concrete.rs` は `B: TensorBackend` の backend-explicit API を公開し、runtime extension と concrete wrapper は `eager.rs` の shared concrete executor を使う。`eager_ad.rs` は `autodiff` feature 下だけである。linalg/FFT にも operation-specific backend trait がある。

したがって「ExtensionOp がなければ tenferro extension ではない」という主張は、repository 自身の標準 extension crate の public strata と矛盾する。standard crate が concrete、runtime、AD の全機能を一 package に同梱していることは、外部作者にも全層実装を要求する根拠にならない。まして現在の standard crates が runtime/internal dependencies を常時持つことを、minimal recipe の証拠にしてはならない。それは maximal package の編成であって、concrete API の論理的必要条件ではない。

## 6. Cycle 2 の gate は二つの fixture に分け直せ

Cycle 2 の「out-of-tree fixture が eager immediate と compiled の両方を通り、AD rule 一致も確認する」という単一 fixture は、今回の問いに対して不十分である。最初から最大機能を実装した fixture は、最小経路が容易だという証明にならない。

v0.3.0 の gate は少なくとも次に分けるべきだ。

### A. concrete-only fixture

- stable public API のみ使用。
- normal dependency は `tenferro-tensor` のみ。CPU は実行確認用 dependency/dev-dependency。
- `tenferro-runtime`、`tenferro-ad`、`tenferro-ops`、internal macro を使わない。
- typed executor 一本から `TypedTensor` と `Tensor` の forward 計算が通る。
- backend/placement/dtype unsupported は typed error。暗黙 transfer/fallback なし。
- semantic program、snapshot lookup、module install、engine prepare の counter は当然ゼロ。

### B. runtime-upgrade fixture

- A と同じ concrete executor を再利用して `ExtensionOp + stable adapter` を追加。
- AD rule なしでも unified `apply_eager` と compiled forward が通る。
- 微分要求だけが typed missing-rule error になる。
- optional AD feature を加えた時だけ eager/compiled の JVP/VJP 一致を検証する。
- Cycle 2 で合意した `native-context` / `native-session` 分離非劣性、targeted ensure no-op、`Unsupported` prepare 早出し、promoted 非退行の gate はこの B にだけ適用する。

この分離により、concrete direct path を runtime immediate baseline に合わせて遅くする愚行も防げる。最速の concrete call は snapshot すら読む必要がない。一方、runtime-integrated call は合意済みの snapshot-resolved native immediate path で graph artifact を避ける。**異なる固定費 floor を、偽の「統一」で一つに潰すな。**

## 最終主張

収束設計は捨てない。ただし適用範囲を正しく限定する。

- concrete-only 作者には、普通の Rust 関数 + `TensorBackend`、または狭い operation-specific backend trait だけを提供する。
- runtime 統合が必要になった作者だけ `ExtensionOp + ExtensionEngine` 層へ昇格させる。
- `PreparedOperationExecutor` は stable adapter の内側に隠す。
- AD rule は最後まで optional にする。

「一つの maximal extension protocol を全員に使わせれば統一されて簡単」という発想は、runtime 実装者の都合を外部作者へ押しつけているだけだ。**外部拡張容易性とは、全能力を一度に買わせることではなく、必要な能力だけ段階的に足せることだ。**# DeepSeek (Cycle 3): GPT の三段階分離は正しいが、Tier1→2 の段差は「崖」であり、その正体（infer_output_meta 必須・effects/aliases 宣言・erased shim）を図が隠している。concrete-only は合成に限られる点も gate に明記せよ。

# DeepSeek (Cycle 3) — GPT の三段階分離は正しい。ただし「段差は一様」という絵は嘘で、Tier 1→2 は崖。崖の正体を実読で特定した。

## 結論

問いへの直接回答は GPT と一致する。**AD 不要・concrete 計算のみの外部ユーザーに `ExtensionOp` も AD rule も必要ない。** これはコード実読で裏付けが取れた。GPT の Cycle 3 の方向（concrete-only と runtime 統合を能力レベルで分離せよ）には全面的に賛成し、私の Cycle 2 までの収束も壊さない。

しかし GPT の提案には**構造的な隠蔽が2つ**ある。(1) 「one kernel source of truth」図が `infer_output_meta` を描いていない。実読すれば、それが Tier 1→2 の段差の大半を占める**必須の別実装**だと分かる。(2) 三段階の表が「作者が実装するもの」を `ExtensionOp` の1語で済ませているが、その実体は8メソッド + engine で、AD rule 1個より遥かに重い。**梯子は等間隔ではない。Tier 1→2 は崖、2→3 は段である。** この非対称を正直に書くこと自体が「拡張容易」の一部だ。

以下、実読に基づいて容赦なく。

---

## 1. 確認事実（GPT の主張の裏付け）

- `crates/tenferro-einsum/src/concrete.rs` は実在し、`pub trait TensorTensordotExt { fn tensordot<B: TensorBackend>(&self, rhs: &Tensor, axes, backend: &mut B) -> Result<Tensor> }` の形で **caller-owned backend + 型付き関数**の concrete 経路を公開している。`ExtensionOp` は一切登場しない。実行は `backend.dot_general(...)` という**既存 `TensorBackend` primitive の合成**。→ **「concrete-only に ExtensionOp は不要」は真。**
- `crates/tenferro-internal-ops/src/ext_op.rs` の `trait ExtensionOp`（:361 以降）を実読。`infer_output_meta` は **default なし・必須**（:447）。`lower_to_standard_ops` は default `Unsupported`（:464）、`semantic_effects`/`semantic_aliases` は default `Undeclared` だが、doc 自身が「semantic-program construction rejects an undeclared payload」と書く（:422 付近）。→ **runtime 統合（特に compiled/traced）では effects/aliases の明示宣言が強制。**
- `crates/tenferro-ad/src/context.rs` を実読。AD rule は `SemanticExtensionRuleSet` から **transform 時（`semantic_jvp`/`grad`/`grad_optional`）にだけ look-up** され、forward の compile/run 経路には現れない。→ **GPT の「no AD rule → eager/compiled forward は正常に動き、微分要求だけが typed missing-rule error」は裏付け可。** Tier 2 の「forward は動く」主張は通る。

以上は GPT に軍配。問題はこの先。

---

## 2. 批判1: 「one kernel source of truth」図は `infer_output_meta` を描いていない。段差の正体は kernel ではない

GPT の図は `concrete typed executor` が concrete 経路と runtime 経路の両方に矢印を伸ばす。**しかし executor は「実行」しかできない。出力の shape/dtype を実行せずに得る手段は executor にない。** 一方 `ExtensionOp` は `infer_output_meta(&mut ExtensionShapeContext) -> Vec<(DType, Vec<SymDim>)>` を**必須**で要求する。

実例: `crates/tenferro-einsum/src/extension.rs:179-240` の `infer_output_meta` は、subscripts のラベルから `SymDim` の束縛を解き、broadcast 制約を `ctx.require_equal` で立て、dtype を promote する**約60行のシンボリック shape/dtype 推論**である。これは `eager_einsum_exec`（kernel）とは**別のコード**だ。concrete 経路は backend が実際の tensor から shape を決めるので、この推論を一切必要としない。

つまり GPT の「昇格は追加であって書き直しではない」は **kernel についてのみ真**。metadata 推論については**作者が新規に書く**。そして外部 op にとって shape 推論は往々にして kernel より難しい（分岐のある shape、broadcast、reduction axis、output-rank 条件）。この最大コストが図から落ちている。

**要求:** 「one source of truth」図は `executor` の横に `infer_output_meta` を明示的に置け。そして能力表の「作者が実装するもの」列を「kernel 1本」と「`infer_output_meta` 1本」に分けよ。`ExtensionOp` という1単語で括る限り、表は「払うコスト」を隠す。

---

## 3. 批判2: 梯子は等間隔ではない。Tier 1→2 は崖

`ExtensionOp` の実装要件を列挙する（ext_op.rs 実読）:

- 必須: `family_id`, `payload_hash`, `payload_eq`, `clone_arc`, `as_any`, `input_count`, `output_count`, **`infer_output_meta`**
- compiled/traced で実質必須: `semantic_effects` 明示宣言, `semantic_aliases` 明示宣言（default `Undeclared` は reject される）
- native engine を持たない場合のみ: `lower_to_standard_ops`
- 加えて `ExtensionEngine`（prepare → `PreparedOperationExecutor`）、そして Cycle 2 で合意した erased-dispatch shim

対して Tier 2→3 の増分は **AD rule 1個**（`context.rs` の `SemanticExtensionRuleSet` に登録するだけ）。**Tier 1→2 のコストは Tier 2→3 の数倍。** GPT の三段階表は「得られるもの」を並べるが「払うもの」の非対称を平準化して見せている。これは拡張容易性の誤認を招く。

**要求:** 表に「実装コスト」行を追加せよ。concrete-only = kernel 1本 / runtime 統合 = kernel 1本 + `infer_output_meta` 1本 + effects/aliases 宣言 + shim（+ 任意で lowering か native engine）/ AD = +rule 1個。作者が最初に払うコストを正直に書くこと。それが「必要な能力だけ段階的に足せる」という GPT 自身の主張の証明になる。

---

## 4. 批判3: 3サイクル経過しても「stable adapter」が未定義。fixture B は現状作れない

Cycle 2 で「internal macro 公開は defer、ただし stable non-macro adapter + out-of-tree fixture は必須」で合意した。Cycle 3 の GPT も「runtime adapter は concrete executor を包む」「stable adapter が semantic op と concrete execution function から生成・配線する」と言う。**だが adapter の具体形はまだ1行も指定されていない。**

そして批判1・2から、adapter の本当の仕事は「`execute_foo` を包む」ことではなく、**作者から (a) typed executor、(b) `infer_output_meta` 相当のクロージャ/関数、(c) effects/aliases 宣言、(d) optional session executor を受け取って、8メソッドの `ExtensionOp` + engine + erased shim を生成する**ことだと確定する。これがなければ fixture B（runtime-upgrade）はコンパイルすらできない。三段階のうち2段目が**検証不能のまま**である。

**要求:** v0.3.0 の成果物として、adapter の最小形をドキュメント1枚（署名レベル）で**今**指定せよ。それが書けないなら、「concrete-only tier は今日 ship できるが、外部作者の runtime 統合は adapter 実装待ち」と明記せよ。未定義の adapter を前提に fixture B を gate に入れるのは、未計測の 15μs を設計判断に使った私の Cycle 1 の過ちの再演だ。

---

## 5. 批判4: concrete-only は「既存 `TensorBackend` primitive の合成」に限られる。leaf kernel は外から実装不能の可能性が高く、gate にその境界が書かれていない

concrete.rs の tensordot は `backend.dot_general(...)` を呼ぶだけ。einsum も `eager_einsum_exec` が既存 primitive の合成。**つまり concrete-only の最小経路が成立するのは「新 op = 既存 `TensorBackend` メソッドの組み合わせ」の場合だけ。**

作者が本物の**新規 leaf kernel**（融合演算、独自のメモリレイアウト、新しい reduction）を欲したとき、(a) 自分の crate に狭い backend trait を定義し、(b) `impl FooBackend for tenferro_cpu::CpuBackend` を書くのは orphan rule 上は合法。**しかしその impl の中身で触れるのは `CpuBackend` の公開 API のみ。** 内部 buffer 管理・kernel ディスパッチ・session が `pub(crate)` なら、外部から leaf kernel は実装できない。GPT は「backend leaf extension API の不足であって ExtensionOp では解決しない」と正しく認めたが、**fixture A の gate にこの境界を書いていない。**

**要求:** fixture A を「既存 `TensorBackend` primitive の合成による concrete 演算」に**明示的に限定**せよ。leaf kernel を書きたい作者には「in-tree 貢献、または backend leaf 公開 API の提供待ち」と recipe に一行入れよ。これを書かないと、recipe を読んだ作者が壁に当たった時に「拡張しやすい」の約束が反故に見える。合成と leaf の区別は、concrete-only 層が隠してはならない唯一の天井である。

---

## 6. 私が推す v0.3.0 の concrete-only スコープ（GPT 案の上に）

1. **Tier 1 recipe + fixture A を先に ship。** 依存は `tenferro-tensor` のみ、実行確認は `tenferro-cpu` を dev-dependency。typed executor 1本 + 任意で `Tensor`/`TypedTensor` extension trait。caller-owned backend、暗黙 fallback/transfer/registration なし。**これは今日出せる。** gate は「`ExtensionOp`/snapshot/engine/module の counter がゼロ」+「合成に限る」の明記。
2. **stable adapter の最小形を文書で指定し、実装は adapter PR に切り出す。** fixture B は adapter が存在するまで gate に入れない。adapter は framework 化せず、author から (executor, `infer_output_meta`, effects/aliases, optional session) を受け取る builder に留める。
3. **Tier 2→3 の AD は既に optional であることを確認済み**（context.rs）。変更不要。forward は rule なしで eager/compiled とも動き、微分だけ typed missing-rule error。この事実を Tier 2 の表に明記すればよい。
4. **実コスト表**（批判2）と**境界の明記**（批判5）を、extension 作者向けドキュメントに必ず入れる。

---

## まとめ

GPT の Cycle 3 は、私が Cycle 2 で収束した設計の**適用範囲を正しく限定した**。concrete-only に `ExtensionOp` も AD rule も要らない、という核心は実読で裏付けが取れ、賛成する。しかし「one source of truth」図と三段階表は、**Tier 1→2 の崖（`infer_output_meta` 必須 + effects/aliases 宣言 + shim）を隠している**。この隠蔽を直さない限り、「必要な能力だけ段階的に足せる」という約束は、足す時に判明する壁の予告にならない。**拡張容易性の第一歩は、段差の非対称を正直に書くことである。**