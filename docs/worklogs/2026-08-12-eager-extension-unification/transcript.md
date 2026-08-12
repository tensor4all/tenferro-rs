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

GPT の Cycle 3 は、私が Cycle 2 で収束した設計の**適用範囲を正しく限定した**。concrete-only に `ExtensionOp` も AD rule も要らない、という核心は実読で裏付けが取れ、賛成する。しかし「one source of truth」図と三段階表は、**Tier 1→2 の崖（`infer_output_meta` 必須 + effects/aliases 宣言 + shim）を隠している**。この隠蔽を直さない限り、「必要な能力だけ段階的に足せる」という約束は、足す時に判明する壁の予告にならない。**拡張容易性の第一歩は、段差の非対称を正直に書くことである。**# User (Feedback): eager AD（requires_grad）でも同様のオーバーヘッドがあるはず。AD記録と forward fast path の両立を追加議論

## User Feedback（追加議論）

eager AD がある場合（`requires_grad` の eager op）、同様のオーバーヘッドがあることが期待される。

つまり、requires_grad 付きの eager op は AD tape 記録（`record_eager_outputs` → `TracedGraph` → `SemanticProgram`）を伴う。Cycle 2-3 で合意した「forward のみの native immediate path（SemanticProgram 構築なし）」では、AD 記録をどう扱うかが未解決である。

論点:
- requires_grad の forward も fast path（SemanticProgram 構築なし）を通れるか、それとも AD 記録のために SemanticProgram 構築が必須か
- AD 記録のオーバーヘッドを forward fast path から分離できるか（遅延記録、または tape 記録の最適化）
- eager AD と traced AD の整合性（同じ SemanticProgram 共有）を保ちつつ、eager forward の fast path を両立できるか
# GPT (Cycle 4): requires_grad は forward fast path を失わない――capture now, materialize later

# GPT (Cycle 4): requires_grad は forward fast path を失わない――capture now, materialize later

## 結論

追加フィードバックは正しい。**eager AD の end-to-end 呼び出しには forward 実行とは別の記録税があり、no-grad の計測だけで fast path を合格させてはならない。** ただし、そこから「`requires_grad` なら `SemanticProgram` を forward で構築するしかない」と結論するのは完全な飛躍である。

私の結論は次の5点だ。

1. **`requires_grad` は native immediate path の eligibility 条件に入れてはならない。** placement、storage、effects、aliases、provider capability が同じなら、forward は no-grad と同じ snapshot-resolved `prepare -> execute` を通る。
2. **forward 時の `SemanticProgram` 構築は不要である。** 必要なのは semantic identity と依存関係を失わない最小記録であって、freeze、fingerprint、compiler、schedule、admission ではない。
3. 遅延すべきなのは「記録」そのものではなく、**記録から完全な `SemanticProgram` を materialize する仕事**である。正確には **capture now, materialize later** だ。完全な非同期・後付け記録は correctness を壊す。
4. eager AD と traced AD の統一点は、forward の物理経路ではなく、**AD transform に渡る canonical `SemanticProgram` と semantic rule**である。eager 専用 AD callback や provider 固有 tape を増やしてはならない。
5. v0.3.0 ではまず immediate forward と現行 recorder を分離して計測する。現行 recorder が支配項だと確認してから、graph analysis の遅延または内部 compact record を別コミットで入れる。**計測前から第二の公開 IR や外部 trait を発明するのは論外**である。

DeepSeek の Cycle 3 は Tier 1→2 の authoring cost の崖を正しく指摘したが、「Tier 2→3 は小さな段」と source code の実装量だけで片付けたのは浅い。**作者の実装コストが小さくても、実行時の per-op 記録・retention・metadata 登録コストは大きくなり得る。** authoring cost と runtime cost は別軸であり、今回のフィードバックはまさに後者の見落としを突いている。

## 1. まず事実を正せ: 現行コードでも AD 記録と `SemanticProgram` 構築は同時ではない

現在の `finish_eager_extension_outputs` は、forward outputs を得た**後**で `record_eager_outputs` を呼ぶ。`record_eager_outputs` → `record_semantic_eager_outputs` は extension なら `tenferro_runtime::extension::apply` を使い、`Graph<StdTensorOp>` と `TracedTensor` を作る。そこで `register_scoped_graph_analysis` まで走るため安いとは限らないが、これはまだ `SemanticProgram` の freeze/compile ではない。

`SemanticProgram` 側へ進むのは後の `semantic_eager_vjp_optional` / JVP で `compile_ad_source` を呼ぶ時である。つまり「AD 記録には forward 時の `SemanticProgram` が必須」という前提は、**TracedGraph recording と SemanticProgram materialization を混同している。** コード自身がすでに両者を時間的に分離している。

したがって最小の正しい変更は明快だ。

```text
apply_eager
  -> shared validation
  -> snapshot-resolved native prepare
  -> native-session または native-context execute
  -> output contract validation
  -> 必要なら semantic recording
  -> EagerTensor を返す
```

forward の実行部分は `requires_grad` の有無で変えない。記録が現行 `TracedTensor` 経路のままでも、**generic forward 用 `SemanticProgram -> compile -> run` をもう一度通す理由はゼロ**である。

「AD があるから generic forward に戻す」は最悪の案だ。forward のための graph pipeline と、将来の AD transform のための semantic record を一緒に払い、同じ semantic op を二度 orchestration するだけである。それは整合性ではなく重複だ。

## 2. ただし「forward fast path に入った」で勝利宣言するな

ここは追加フィードバックに全面的に同意する。execution counters がゼロでも、呼び出しの後半で `extension::apply`、GraphBuilder、graph analysis、metadata scope 登録を払えば、利用者が見る latency は高いままである。

しかも現行コードでは、正確な分岐は単純な `requires_grad` ではない。

- `eager_grad_recording_enabled()` は `no_grad` depth が 0 なら true。
- `finish_eager_extension_outputs` は recording enabled なら、入力がすべて untracked でも `record_eager_outputs` を呼ぶ。
- `requires_grad` の scan は出力の tracking と grad-slot/value-record 登録を決めるが、semantic trace 自体の記録税は recording-enabled/untracked でも発生し得る。

したがって「requires_grad overhead」を測る時に、次の三つを混ぜるな。

1. `no_grad` scope: semantic recording なし。
2. recording enabled だが全入力 untracked: semantic trace は保持、grad slot なし。
3. 一つ以上が `requires_grad`: semantic trace + tracked output 登録。

この三系列を分けず、(1) と (3) の差を丸ごと「requires_grad の税」と呼ぶベンチマークは雑である。逆に、(2) を消すため「全入力 untracked なら何も記録しない」と短絡するのも危険だ。functional JVP/VJP が untracked intermediate を `wrt` に取れる現在の意味論を変え得る。**性能修正のふりをした AD API 意味論変更を紛れ込ませてはならない。** まず契約を固定し、三系列を測れ。

## 3. 正しい分離は「同期 capture、遅延 materialization」

AD のために forward で必ず捕捉しなければならないものはある。

- 元の `StdTensorOp::Extension(Arc<dyn ExtensionOp>)`、すなわち exact semantic payload
- ordered input record IDs と output slot IDs
- multi-output を一つの op node として保つ identity
- concrete output arity/dtype/shape と、forward validity に必要な operation-local semantic metadata
- alias/effect contract
- forward 値を residual として必要とする既存 AD contract があるなら、その owner-safe handle

これを forward 成功後に同期的に commit しない案は不正である。background thread に投げれば、直後の `backward()` との race、record failure の遅延、context drop、reconfigure、multi-thread eager ordering、effectful op の順序が壊れる。**「非同期 tape なら forward が無料」は、失敗と lifetime を別スレッドへ隠しただけのインチキ最適化だ。**

一方、次は forward で行う必要がない。

- reachable DAG 全体の再走査
- 完全な `SemanticProgram` の構築・freeze
- semantic fingerprint
- `GraphCompiler`
- AD transform
- `ScheduledGraph` / global `BufferPlan`
- run admission と event-slot table

つまり capture は一 op 当たり O(input arity + output arity) で、過去の graph depth に比例してはならない。最初の `grad` / `vjp` / `jvp` / compile 要求時に、対象 output から reachable record を input order に沿って決定的に巡回し、**既存の canonical `SemanticProgramBuilder` へ一度だけ materialize**する。その後は既存 `AdContext`、`SemanticExtensionRuleSet`、transform cache、prepared derivative cache をそのまま使う。

ここで重要なのは、lazy にするのは**完全 program の組み立て**であって、forward の semantic validity まで全部後回しにすることではない。`infer_output_meta`、effects、aliases の不正を backward 時まで隠せば、traced なら graph build で失敗する op が eager forward だけ成功する。これは Cycle 3 で守ろうとした eager/traced 整合性を自分で破壊する。

したがって operation-local canonical inference が forward-time contract の一部なら、それは lightweight recorder/validator で一度だけ行い、symbolic output metadata と constraints を記録側へ渡すべきだ。**full graph analysis を避けることと、semantic validation を省略することを混同するな。** 後の materializer はその canonical metadata を import し、別の eager 専用推論を実装しない。

## 4. compact record は必要なら内部実装に留めろ

現行 `record_semantic_eager_outputs` の何が高いかはまだ分解計測が必要である。単一 carrier `Graph` の生成自体が十分安く、`register_scoped_graph_analysis` や metadata registry が支配しているなら、最小解は既存 `Graph<StdTensorOp>` を保持したまま analysis/freeze を遅延することだ。最初から `EagerTapeIR`、serializer、new arena、new public trait を作る必要はない。

それでも current graph wrapper の object count や scope merge が支配するなら、内部 record は次の最小形で足りる。

```text
SemanticRecordNode {
  op: StdTensorOp,             // exact semantic source of truth
  inputs: [SemanticValueId],   // input order preserved
  outputs: output arity/index,
  inferred metadata/constraints,
  required retained bindings,
}
```

これは第二の演算語彙ではない。`StdTensorOp` を運ぶ deferred carrier にすぎない。新しい eager AD rule、eager lowering、provider handle、prepared executor を記録してはならない。multi-output は output ごとに payload を複製せず、一つの node + output index にする。common subgraph は stable record ID で deduplicate し、HashMap の偶然の iteration order で materialize してはならない。shape churn で fingerprint が揺れれば失格である。

この内部最適化を外部 extension author に露出させる案は即却下する。外部作者が提供するのは Cycle 3 の契約どおり、runtime-integrated 層では semantic `ExtensionOp`、typed forward executor、任意の semantic AD rule である。**`record_eager`、`save_for_backward_eager`、`EagerTapeOp` のような第二 API を書かせた瞬間に、大前提を破る。** recorder は全 extension に対して runtime が共通実装する。

## 5. eager AD と traced AD の一致は「同じ物体を毎回作ること」ではない

「同じ `SemanticProgram` を共有する」を、forward の各 op で完成済み program object を作ることだと読むのは馬鹿げている。eager graph は呼び出しごとに伸びるので、将来の output root がまだ存在しない。共有すべきものは次である。

1. eager record は forward provider や native/session 選択ではなく、**元の semantic extension payload**を保持する。
2. materialization は traced と同じ canonical metadata inference、effects、aliases、shape constraints、provenance を使う。
3. eager と traced は同じ `FrozenProgram` transform と同じ extension AD rule registry に入る。
4. native forward が CPU、CUDA、別 provider のどれを選んでも source semantic fingerprint は変わらない。
5. AD rule の missing/unsupported は forward executor の失敗に偽装せず、実際の transform 要求時に typed error になる。Cycle 3 の「AD rule は optional」を維持する。

これなら eager materialization 結果と直接 traced で作った結果に `semantic_eq` と同一 fingerprint を要求できる。現在の shape-churn test が守る「長さ 2 と 3 でも symbolic program は同一」という性質も維持できる。

逆に、次の案はすべて不合格だ。

- forward で選ばれた `PreparedOperation` や provider slot を tape に保存する: reconfigure で stale、traced AD と非同一。
- eager 専用 VJP closure を executor に返させる: 外部作者の二重実装、高階 AD と transform cache が分裂。
- extension を forward で core lowering した結果だけ記録する: semantic extension rule を迂回し、native と promoted で AD が変わる。
- per-op `SemanticProgram` を作って cache する: Cycle 1 の (a) を AD 名義で復活させただけ。
- inference/effects/aliases を全部 backward まで遅らせる: forward error contract が traced とずれる。

なお effectful・nondeterministic op を「後で primal replay すればよい」と仮定してはならない。forward 値が residual として必要なら owner-safe に捕捉する既存 semantic AD contract が必要であり、それがなければ typed unsupported で止めるべきだ。遅延 materialization は遅延**再実行**ではない。

## 6. `requires_grad` fast path の edge cases

最低でも以下を gate に含める。

- mixed tracked/untracked inputs: untracked 値を消さず、semantic constant/binding として正しく保持。
- `no_grad` と detach: tape cut の意味論を変えない。
- multi-output: record node は一つ、各 output の active mask と output index は正しい。
- 同じ intermediate を二入力に使う場合: node を二重 materialize しない。effectful op なら特に必須。
- view/must-alias outputs: record のために勝手な contiguous copy を追加しない。
- non-contiguous、mixed placement rejection、GPU device identity: forward immediate と recorded program の input contract が一致。
- provider replacement: forward-time provider lifetime は snapshot が守り、後の AD source semantics は provider identity に汚染されない。
- AD rule replacement/missing、integer/Bool/complex、higher-order JVP-of-grad: eager 専用 shortcut を入れず同じ typed semantic rule を通す。
- forward 成功・record commit 失敗・multi-output construction 失敗の failure atomicity: 半端な node や grad slot を context に残さない。

## 7. 受け入れ計測を AD 用に作り直せ

Cycle 2 の `native-session` / `native-context` 二系列だけでは不足である。それぞれについて少なくとも次を測る。

| 系列 | 何を測るか |
|---|---|
| recording disabled (`no_grad`) | pure immediate forward floor |
| recording enabled / all untracked | semantic capture tax |
| one or more `requires_grad` | capture + tracked value/grad-slot registration tax |
| first VJP/JVP/backward | deferred materialization + transform + compile + execution |
| repeated VJP/backward | transform/prepared cache の実効性 |
| N forwards + one backward | 単なる cost shifting でない total-work 改善 |

promoted-graph についても同じ三つの recording mode を測り、Cycle 2 の `prepare(Unsupported)` early bound を維持する。

counter は phase を分ける。

### Forward execution phase

- `SemanticProgram` construction/freeze = 0
- fingerprint = 0
- `GraphCompiler` = 0
- `ScheduledGraph` / global `BufferPlan` = 0
- admission/event-slot table = 0
- native prepare/execute = 1 回
- session/resource entry = 最大 1 回

### AD capture phase

- semantic op record = 1 node（multi-output でも1）
- work = O(arity)、graph depth に非依存
- provider/prepared handle capture = 0
- full-program materialization = 0
- allocation/retained bytes は no-grad と同じとは要求しないが、既存 recorder 以下かつ明示的に計測

### First AD request

- reachable DAG materialization = 最大 1 回/root
- eager materialized program と traced programの `semantic_eq` / fingerprint 一致
- numeric VJP/JVP/HVP、typed errors、shape-churn 一致

ここで forward だけ速くして backward に同じ仕事を丸ごと移し、「forward fast path 達成」と宣伝してはならない。forward-only workload では遅延は実益だが、forward+backward の総計も別に公開する。**遅延は自動的な削減ではない。毎 op の full analysis を O(1) capture に変え、root ごとに一度だけ materialize して初めて削減である。**

## 8. v0.3.0 の実装順序

スコープを爆発させない順序はこれしかない。

1. direct bridge と generic の両方で、上記 AD 三系列の baseline と phase counter を追加する。
2. unified `apply_eager` の snapshot-resolved immediate forward を、`requires_grad` に関係なく通す。recording はまず現行 `record_eager_outputs` のままにして、forward 経路統一の correctness と効果を単独計測する。
3. recording section が predeclared threshold の支配項なら、まず既存 graph carrier の full analysis 遅延を試す。それで足りなければ内部 compact record を導入する。外部 API・外部 trait・eager 専用 AD ruleは増やさない。
4. eager materialization と traced の semantic equality、shape churn、VJP/JVP/HVP、multi-output、reconfigure gate を通してから call-site session bridge を公開経路から外す。

要するに、**`requires_grad` は forward fast path を無効化する理由ではない。しかし AD capture をゼロコストと偽る理由にもならない。** forward は同じ native immediate path、記録は最小かつ同期、完全 `SemanticProgram` は AD 境界で遅延 materialize、rule は traced と共有する。この分離なら single-op latency、AD 整合性、外部拡張容易性の三つを同時に守れる。# DeepSeek (Cycle 4): GPT の「capture now, materialize later」は方向正しいが、現行 recorder は毎 op `extension::apply`（graph build＋analysis＋infer_output_meta＋global registry）を再入しており、「analysis 遅延」は incremental metadata 伝播を壊すので順序が逆。eager 側 scope Vec 具現化は O(depth²)/op の別バグ。

# DeepSeek (Cycle 4) — GPT の「capture now, materialize later」は方向正しいが、現行 recorder の正体を読み違えている。記録経路は `extension::apply` の毎 op 再入であり、「analysis 遅延」は incremental metadata 伝播を壊すので順序が逆。加えて eager 側 scope Vec 具現化が O(depth²)/op の別バグ。

## 結論（先に）

GPT の Cycle 4 の 5 命題のうち、**1（requires_grad を eligibility に入れるな）、2（forward に freeze/fingerprint/compile/schedule/admission は不要）、4（eager/traced の統一点は canonical SemanticProgram と semantic rule）、5（計測前に第二公開 IR・外部 trait を発明するな）に同意。** 非同期 tape を「インチキ」と断じた判定、provider/prepared handle を tape に保存するな、という制約も正しい。

しかし**命題 3 の「capture now, materialize later」の前半「capture」の正体を GPT は間違えている。** ユーザーの「requires_grad でも同様のオーバーヘッド」の正体は `SemanticProgram` ではない。記録経路そのものが、毎 op、generic 層の graph 構築パイプラインを**再入**している。GPT は「現行コードでも AD 記録と SemanticProgram 構築は同時ではない（`record_semantic_eager_outputs` は Graph と TracedTensor を作るだけで、freeze/compile は後）」と正しく観察したが、そこから「forward に SemanticProgram は不要」と片付けたのは、問題を 1 層だけずらしたにすぎない。**記録が払っている per-op コストこそが削るべき対象であり、それは「SemanticProgram 構築」という名前ではない。**

以下、コード実読に基づき容赦なく。

---

## 1. 事実: 記録経路は毎 op で `extension::apply` を再入している。これは generic 層の graph build + analysis + canonical 推論 + global registry である

`crates/tenferro-ad/src/eager.rs` の `record_semantic_eager_outputs`（:3993 付近）は、extension op の場合こう呼ぶ:

```rust
StdTensorOp::Extension(ext) => {
    tenferro_runtime::extension::apply(Arc::clone(ext), &semantic_inputs)?
}
```

そして `crates/tenferro-runtime/src/extension.rs` の `apply`（:110 付近）は、毎回:

1. `GraphBuilder::new()` → `add_parent(input.graph.clone())`（各入力）→ `add_operation` → `Arc::new(builder.build())` — **新しい carrier Graph を毎 op 構築**
2. `register_scoped_graph_analysis(graph, [])` — **graph analysis walk**
3. `traced_outputs_from_analysis` — 各出力に `TracedTensor` を生成し、`merge_traced_inputs_map`・`MetadataScopeChain`・`ConstraintScopeChain`・`extra_roots`・`checkpoint_chain` を入力から継承クローン

この 2 の analysis walk（`crates/tenferro-runtime/src/metadata.rs` の `graph_analysis_registrations` → `append_graph_metadata_registrations` → `infer_output_metas`）は、extension op に対して **`infer_extension_output_meta_with_constraints` すなわち canonical `infer_output_meta` を毎 op 呼ぶ**。さらに各入力 key について `lookup_global_metadata`（global registry の read）を行い、最後に `register_scoped_global_metadata_batch` で symbolic 出力 meta を global registry に書く。

つまり**ユーザーの直感は文字通り正しい**。requires_grad の eager op は、forward 実行とは別に、generic 層が traced 経路で払うのと**同種の** graph build + analysis + シンボリック推論 + global registry lock を、記録として毎 op 払っている。GPT の「SemanticProgram 構築とは別物」は技術的には真だが、**コストの性質は「別物」ではなく「一段軽いだけで同じ族」** である。

---

## 2. 構造的欠陥その1: 「analysis 遅延」は incremental metadata 伝播を壊す。GPT の step 3 の順序は逆

GPT は実装順序 step 3 で「recording が支配項なら、まず既存 graph carrier の full analysis 遅延を試す。それで足りなければ内部 compact record を導入」と書く。**この順序は成立しない。**

`register_scoped_graph_analysis` は単なる「余分な解析」ではない。`apply` 内で出力 shape を `registered_meta(&graph.values()[output].key)` で**global registry から読み戻す**ことで TracedTensor の `shape_hint`/metadata を組み立て、そして**次の op の記録がその key を `lookup_global_metadata` で解決する**。すなわち per-op の analysis + global registry 登録は、**incremental に伸びる eager graph の input→output metadata 伝播機構そのもの**であり、負荷というより配線である。

これを単純に「遅延」すれば、下流 op の記録時に入力 meta が registry に存在せず、`missing input metadata` エラーになる（`append_graph_metadata_registrations` は見つからない場合、compatibility path で親 graph を再帰走査して拾いに行くが、それは analysis をむしろ深く再実行する）。したがって **analysis 遅延は、それを可能にする別の metadata 伝播手段（＝compact record が meta をローカルに保持する設計、または materialize 時の一括再解析）を先に要求する。**

**正しい順序は逆: まず record の担体を決め（Graph を保つなら per-op analysis を別の明示的 meta 伝播に置換、compact record に置換するならその変換 glue）、その上で analysis が集約できる。** GPT の「analysis 遅延を先に試す」は、後で compact record を前提としていた自分の結論と噛み合わない。

---

## 3. 構造的欠陥その2: counter 表の「work = O(arity)、graph depth 非依存」は現行コードと矛盾。eager 側 scope Vec 具現化が O(depth²)/op

GPT の「AD capture phase」counter 表は `work = O(arity)、graph depth に非依存` と書く。現行 recorder はこれを満たさない。しかも原因は analysis ではなく**別の場所**にある。

`crates/tenferro-ad/src/extension.rs` の `finish_eager_extension_outputs`（:420 付近）は、各出力 EagerTensor に対し materialized された `metadata_scopes: Vec<Arc<GlobalMetadataScope>>` を毎 op 再構築する:

```rust
let mut metadata_scopes = vec![Arc::clone(&recorded.metadata_scope)];
for input in inputs {
    for scope in &input.metadata_scopes {
        push_metadata_scope(&mut metadata_scopes, Arc::clone(scope));  // 線形スキャン dedup
    }
}
```

`EagerTensor.metadata_scopes` は `Vec<Arc<GlobalMetadataScope>>`（eager.rs:2943）であり、`MetadataScopeChain` の lazy な `OnceLock` 構造（metadata.rs:25-30 は遅延 materialize）とは**別に、eager 側で毎 op 具現化**される。`push_metadata_scope` は `scopes.iter().all(|e| !Arc::ptr_eq(e, &scope))` の**線形スキャン**。単項 chain では op n の入力が持つ scope 数は n-1 なので、op n の具現化は O(n²)、**n op の chain 全体で O(n³) の Arc 比較と Vec 再確保**になる。

これは analysis 遅延では消えない。GPT の compact record（`inputs: [SemanticValueId]` の直接辺）がこれを直すのは、**record が `Vec<Arc<GlobalMetadataScope>>` の持ち回りそのものを廃止する場合だけ** である。record が「op + 入力 id」を保ち、scope 具現化を materialize 時の一度の walk に移して初めて O(arity)/op になる。GPT の counter 表は、この設計変更を前提とした**到達目標**であって現行の観測値ではない。計測 gate では「chain depth で latency が線形を超えて増えないこと」を、現行 recorder の測定値と対比して確認する形にせよ。

---

## 4. 構造的欠陥その3: 記録 floor に `infer_output_meta` と global registry lock が入る。shape-churn 安定性が concrete 化を禁じ、二重登録も走る

GPT は「operation-local canonical inference は lightweight recorder/validator で一度だけ行い、symbolic output metadata を記録側へ渡す」と言うが、この「lightweight」は事実に反する。三つの確定事実がある。

**（a）`infer_output_meta` は記録時に必ず走る。** 上記 1 のとおり、analysis walk が extension op に canonical `infer_extension_output_meta_with_constraints` を呼ぶ。これは Cycle 3 で私が「崖」と呼んだ約 60 行（einsum の場合）のシンボリック shape/dtype 推論であり、軽量ではない。**AD 記録の floor = forward 実行 + `infer_output_meta` + registry 操作** であって、GPT の「capture = O(arity)」はこの項を数えていない。

**（b）shape-churn 安定性が「concrete 記録 + 遅延 symbolic」への逃げ道を塞ぐ。** `new_leaf`（eager.rs:3068 付近）は leaf の semantic_trace を `TracedTensor::from_shared_tensor_value_symbolic_shape` で**シンボリック shape で**作る。したがって記録時の `infer_output_meta` にシンボリック入力 meta が渡り、出力もシンボリックになる。これが「長さ 2 と 3 でも symbolic program は同一」の土台である。もし GPT の「遅延」が「forward が知っている concrete shape/dtype だけ記録し、symbolic 推論は materialize 時にやる」を意味するなら、**shape-churn 安定性と eager/traced の `semantic_eq` をその場で壊す**（concrete 長さが record に入れば長さ 2 と 3 は別 fingerprint）。GPT 自身が §5 で「shape-churn test が守る性質を維持」と書いている以上、**記録時の symbolic 推論は省略できず、そのコストは必ず記録 floor に入る**。ここは「遅延できる」と「省略できる」の区別を GPT が曖昧にしたままだ。

**（c）global registry への二重登録 + per-op lock。** 1 回の記録で、(i) `extension::apply` 内の `register_scoped_global_metadata_batch`（symbolic、semantic trace 用）と、(ii) `record_eager_outputs_from_metadata`（eager.rs:4108 付近）末尾の `register_scoped_metadata_batch`（concrete、eager key 用）が**別々に走る**。加えて入力 key ごとの `lookup_global_metadata`。つまり registry への batch 書込が毎 op 最低 2 回、read が arity 回。これらは global lock を取るため、**マルチスレッドの requires_grad eager はこの registry で直列化する**。GPT の compact record は「inferred metadata/constraints をローカルに保持」する形なのでこれを避け得るが、**GPT は「record は per-op で global registry に触れない」と明記していない。** 明記せよ。触れる限り、compact record は「graph は軽くしたが registry 競合はそのまま」という半端に終わる。

---

## 5. 構造的欠陥その4: `SemanticRecordNode` は `Graph<StdTensorOp>` と並存する第二 carrier であり、YAGNI の可能性が高い

GPT は compact record を「第二の演算語彙ではない、`StdTensorOp` を運ぶ deferred carrier」と弁護する。だがその役割は**既存の `Graph<StdTensorOp>` と同一**である。そして実読すれば、**Graph 構築自体は既に O(1)/op** だ: `add_parent` は Arc クローン、`add_operation` は 1 ノード、`Arc::new(builder.build())`。高いのは Graph 構築ではなく、**per-op の `register_scoped_graph_analysis`（lock + `infer_output_meta` + batch）と、eager 側 scope Vec の具現化（§3）** である。

したがって最小の変更はこうなる可能性が高い:

1. 既存の `Graph<StdTensorOp>` を incremental に構築し続ける（既に O(1)/op、変更不要）。
2. per-op の `register_scoped_graph_analysis` をやめ、出力 meta を record/辺にローカル保持（または materialize 時に一度だけ analysis を一括実行）。
3. eager 側 `metadata_scopes` の materialized Vec 持ち回りをやめ、lazy chain のままにする（または record id 直接辺に）。

これなら `SemanticRecordNode` という**新しい carrier 型は不要** である。Graph が既に載せているもの（op、入力順、出力 index、親参照）をそのまま使えばよい。もし materialize 時に Graph から `SemanticProgramBuilder` へ変換するなら、そこで `register_scoped_graph_analysis` 相当を**一度だけ**走らせれば、GPT の「canonical metadata を import し、eager 専用推論を実装しない」要求も満たす。

**要求:** compact record を入れる前に、「Graph 構築を保ったまま per-op analysis と eager scope 具現化だけを止める」分解計測を行え。それで O(arity)/op が達成できるなら、`SemanticRecordNode` は YAGNI として落とせ。GPT 自身が「最初から新しい IR・arena・serializer を作るな」と言った。同じ自制を、Graph と同型の compact record にも適用せよ。

---

## 6. gate への追加要求（GPT の計測表を分解で強化）

GPT の「AD capture phase」counter 表は、項が粗すぎて「どの層が高いか」を識別できない。以下を分離計測 gate にせよ。

1. **記録の 5 内訳を分解**: (a) `GraphBuilder` 構築、(b) analysis walk + `infer_output_meta`、(c) global registry の read/batch 書込回数、(d) eager 側 `metadata_scopes` Vec 具現化、(e) eager key への concrete meta 登録。GPT の「支配項が analysis なら…」という step 3 の条件分岐は、この分解があって初めて判定可能になる。
2. **depth 依存性の counter**: 単項 chain で記録 latency が depth に対し線形以下であること。現行は §3 の O(depth²)/op。goal は「O(arity)、depth 非依存」であり、この counter は analysis 遅延では通らない（scope Vec 具現化が残るため）。
3. **マルチスレッド requires_grad のスループット**: global metadata registry への per-op read/write が残る限り直列化する。record が registry に触れないことを契約テストで確認。
4. **shape-churn 安定性の記録層での維持**: record に concrete shape/dtype を入れないこと（§4(b)）。これと「forward が既知の concrete meta を使える」の両立は、**record には concrete を入れず、eager key 側の concrete 登録だけ別に保つ**（現行の二重登録構造）か、materialize 時に symbolic を再導出するかの二者択一で、どちらもコストが要る。選択を明記せよ。
5. materialize が root ごとに一度・O(reachable) で、per-op 再解析をしないこと（GPT の total-work 計測 `N forwards + one backward` は良い。それに加え materialize 自体の depth 線形性を counter で確認）。

---

## まとめ

GPT の Cycle 4 は、ユーザーの指摘を「forward の SemanticProgram は不要」という正しい答えに接続したが、**記録経路が generic 層の graph build + analysis + `infer_output_meta` + global registry を毎 op 再入している** という正体を捉え損ねた。方向（capture now, materialize later、同期 capture、canonical rule 共有）は正しい。しかし (1)「analysis 遅延を先に試す」は incremental metadata 伝播を壊すので順序が逆、(2) O(arity) 目標は現行の eager 側 scope Vec 具現化（O(depth²)/op）と矛盾、(3) `infer_output_meta` と global lock が記録 floor に入り、shape-churn 安定性が concrete 化を禁じる、(4) compact record は Graph と同型であり、Graph を保ったまま per-op analysis と scope 具現化だけ止めれば不要の可能性が高い。

この 4 点を gate に織り込めば、v0.3.0 の eager AD 経路設計は私の側から異論がない。
# User (Feedback): eager ADはセーブ情報を最小化（PyTorch級）しつつ、eager/traceの経路は同一維持、という方針を提示

## User Feedback（追加方針）

eager AD の場合は、できるだけセーブする情報を減らす方向性が良い。**PyTorch と同程度に。** ただし、**eager と trace の経路は同じに保ちたい。**

補足（facilitator 確認）:
- AD 変換（linearize / transpose）自体は既に `AdTransformCache`（tenferro-ad/src/transform_cache.rs、LRU）でキャッシュされている。
- したがって eager AD の残るオーバーヘッドは「毎 op の記録（graph build + analysis + infer_output_meta + global registry）」であり、これをどう最小化するかが論点。
- ユーザー方針: 記録（tape）にセーブする情報を最小化（PyTorch 級）。ただし forward の物理経路は eager/trace で同一に保つ（SemanticProgram 共有・canonical rule 共有は維持）。
# GPT (Cycle 5): 最終収束――同じ実行契約、PyTorch型の最小 tape、初回 AD 時だけ共通解析

# GPT (Cycle 5): 最終収束――同じ実行契約、PyTorch型の最小 tape、初回 AD 時だけ共通解析

## 判定

DeepSeek の Cycle 4 は、**現行 recorder の病巣の特定**では正しい。私は Cycle 4 で順序を誤った。

- 現行 `record_semantic_eager_outputs` は毎 op `extension::apply` に再入し、raw graph append だけでなく analysis、`infer_output_meta`、global metadata registry を払っている。
- 現行 `EagerTensor.metadata_scopes: Vec<_>` の継承・線形 dedup は、chain depth に応じた具現化コストを作る。
- したがって「現在の `TracedTensor` recorder のまま analysis 呼び出しだけ消す」ことはできない。下流記録が incremental metadata を失う。
- `Graph<StdTensorOp>` 自体が既に op・入力順・出力 index・親参照を持つ以上、同型の `SemanticRecordNode` を増やすのは YAGNI である。

この四点は受け入れる。**Cycle 4 の「まず analysis 遅延、それで不足なら compact record」という順序は撤回する。** 正しくは、raw graph append と analysis を分離し、eager tape が incremental symbolic metadata を必要としない形へ先に変え、その後に同じ canonical analyzer を AD 境界で一括実行する、である。

ただし DeepSeek の「`infer_output_meta` は eager capture の不可避な floor」「shape-churn 安定性のため per-op symbolic inference は省略不能」という結論は、なお誤りである。そこには **eager の下流 op も `TracedTensor` の symbolic metadata を入力にしなければならない**という隠れた前提がある。concrete eager の下流 op は、forward が返した `Tensor` の concrete dtype/shape/layout を既に持つ。AD tape が毎 op symbolic output metadata を供給する必要はない。

symbolic inference に必要なのは、semantic op、ordered edges、cut leaf の最小 schema である。それらがあれば、最初の AD 要求時に root から一度だけ topological walk し、leaf に canonical な `SymDim` を割り当て、traced と同じ `infer_output_meta` を呼べる。concrete extent は runtime binding として扱い、semantic fingerprint に入れなければ shape-churn 安定性も壊れない。**「今 infer しない」ことと「concrete shape を semantic op に焼き込む」ことを二者択一にした DeepSeek の論法が間違っている。第三の、op と辺だけを保持して後で同じ symbolic inference を行う経路がある。**

新しいユーザー方針を入れると、この第三の経路が最終解になる。

## まず二つの曖昧語を固定する

### 「PyTorch と同程度」

ブランド名を性能閾値の代わりにしてはならない。ここでは保存モデルを次のように定義する。

1. grad-active な入力がなければ autograd node を作らない。
2. node が持つのは backward topology、semantic op、rule が必要と宣言した tensor residual、必要最小限の小さな metadata だけ。
3. 全中間 Tensor、全 symbolic metadata、analysis result、registry scope、provider plan を一律保存しない。
4. detach / `no_grad` は graph cut になる。
5. residual Tensor は copy せず owner-safe handle で保持する。

これは PyTorch の `grad_fn` / `next_edges` / `saved_tensors` に対応する水準である。Python と Rust の μs を直接同じにする、という無意味な主張ではない。

### 「eager と trace の経路は同じ」

文字どおり「trace construction と concrete eager execution が同じ全命令列を通る」は不可能である。trace は append 時に kernel を実行せず、eager は実行するからだ。これを要求すれば single-op fast path を捨てるしかない。

守るべき同一性は、以下である。

- 同じ `StdTensorOp::Extension` / `ExtensionOp` payload
- 同じ raw graph append primitive
- 同じ pure/local graph analyzer、`infer_output_meta`、effects/aliases contract
- 同じ canonical `SemanticProgram` / freeze
- 同じ semantic AD rule と `AdTransformCache`
- 同じ runtime engine selection、`ExtensionEngine::prepare`、prepared executor
- 同じ typed errors と placement/alias contract

差を許すのは**同じ phase をいつ払うか**だけである。trace は symbolic metadata が必要な時に analysis し、eager autograd は最初の AD 要求時に analysis する。物理 kernel 実行は runtime-owned な一つの `execute_prepared_operation` helper に統一し、eager immediate と compiled execution の双方が同じ prepared executor を呼ぶ。call-site session callback は消す。

## 新方針が必然的に要求する semantics change

Cycle 4 で私は「recording enabled / all inputs untracked」も現行 semantics を変えず測れ、と言った。しかし、**全 untracked history を暗黙に保存し続けながら PyTorch 級の保存量を要求するのは矛盾**である。ここは測定で解決する問題ではない。

最終設計では ordinary eager autograd を PyTorch 型にする。

- `no_grad`: tape なし。
- recording enabled でも grad-active input が 0: tape なし。
- 1 個以上の grad-active input: active edge だけを持つ node を 1 個追加。
- 後から untracked intermediate を任意の `wrt` に選びたい functional JVP/VJP は、明示的な trace/capture API で囲む。暗黙の全履歴保存では実現しない。
- untracked tensor に後から `requires_grad` を付けるなら、それ以前は切れ、新しい leaf になる。

これは breaking semantics になり得るため v0.3.0 migration note が必要だが、逃げ道はない。現行の「いつか functional AD に使うかもしれないから全 op を記録」は、まさに今回ユーザーが捨てるよう指示した speculative retention である。

その結果、eager から materialize した AD source と full trace 全体を無条件に `semantic_eq` 比較してはならない。比較対象は、**同じ active roots/wrt で slice し、non-active inputs を bindings に切った normalized AD source**である。ここを曖昧にすると、PyTorch 型 graph cut と full trace 同一性を同時要求する矛盾した gate になる。

## 最終設計

### 1. Forward 実行は grad の有無で分岐させない

runtime-integrated extension の公開入口は `apply_eager` 一つである。

```text
shared validation
  -> immutable snapshot から family/engine resolve
  -> ExtensionEngine::prepare
  -> runtime-owned execute_prepared_operation
       -> native-session または native-context
  -> concrete output contract validation
  -> 必要な時だけ autograd capture commit
```

`requires_grad` は immediate eligibility に入れない。`Prepared` なら graph artifact なしで実行し、prepare-time `Unsupported` だけ prepared graph へ昇格する。execution error を fallback に読み替えない。`prepare(Unsupported)` は planning/allocation 前に早出しする。

`execute_in_session` と必須 `execute` の違いは stable adapter の内側へ隠す。外部作者は typed executor を一本書けば `native-context` に参加でき、最低 latency が必要な場合だけ optional session adapter を足す。ただし runtime 側は eager/compiled の双方から同じ internal execution helper を使い、同じ backend/resource 条件なら同じ executor variant を選ぶ。

first-party hot wrapper は targeted `ensure_extension_module_for_engine` の snapshot read-only hit を使う。install-or-replace の意味論は変えない。steady state で install mutex と `CandidateConfig::from_snapshot` 全 clone を払わない。

### 2. AD tape の operation carrier は既存 `Graph<StdTensorOp>` のまま

新しい `EagerTapeIR` や `SemanticRecordNode` は作らない。`extension::apply` を次の二段へ因数分解する。

1. `append_raw_extension_op`: graph parent、ordered input edge、op node、output ids だけを追加する。
2. `analyze_extension_graph`: leaf schema から canonical metadata/constraints を推論する。

trace は 1 を呼んだ後、symbolic `TracedTensor` の metadata が必要な時に 2 を呼ぶ。eager autograd は 1 だけを呼び、2 は AD 要求まで呼ばない。**append 自体は共通で、analysis 実装も共通であり、eager 専用の演算語彙も推論器もない。**

tracked eager output が保持する tape handle は概念的に次だけでよい。

- raw `Graph<StdTensorOp>` の output id
- grad-active predecessor edges
- cut leaf の binding id と最小 schema
- selected residual bindings
- multi-output の active mask / output index

`metadata_scopes: Vec<_>` の持ち回りと per-op materialization は廃止する。leaf/binding 情報は一度だけ value に付け、親は `Arc`/stable id で共有する。join 時に祖先 scope の Vec を merge/dedup してはならない。multi-output は一 op node を共有し、payload も residual も出力ごとに複製しない。

### 3. 保存 Tensor は canonical AD rule の residual contract で限定する

PyTorch 級を本当に満たすには「graph を軽くした」だけでは不十分である。Tensor retention を rule ごとに限定する。

v0.3.0 の最小 `ResidualSpec` は、semantic primal の input/output index と、小さな shape/dtype metadata の mask で足りる。

- `add`: tensor residual 0
- `mul`: rule が必要とする相手 input のみ
- `exp`: output のみ
- `reshape` / reduction: tensor ではなく必要な rank/shape/axis metadata のみ
- AD rule なしの extension: tensor residual 0。実際に AD を要求した時だけ typed missing-rule error

これは eager callback ではない。**同じ semantic AD rule registration の一部**として宣言し、traced AD の primal liveness/binding planning にも使う。rule が未宣言 primal を読む実装は contract violation として fixture/test で落とす。外部作者が AD を不要とする場合は何も追加しない。AD を提供する作者だけ、canonical rule と residual mask を一度書く。これは PyTorch custom autograd の `save_for_backward` と同程度の責任であり、eager 用 VJP closure を二本目として実装させる案よりはるかに小さい。

任意の executor-private auxiliary tensor、recompute policy、checkpoint framework は v0.3.0 には入れない。既存 AD rule が semantic inputs/outputs だけで表せない実例が出た時に追加すればよい。今それを一般化するのは YAGNI である。provider slot、prepared executor、algorithm plan、session、snapshot は residual に絶対保存しない。

view/must-alias の residual は copy せず既存 owner/alias contract を使う。effectful・nondeterministic op は primal replay を仮定せず、rule と必要 residual がなければ typed unsupported とする。

### 4. 初回 AD 要求で一度だけ canonical materialization

`backward` / VJP / JVP / linearize の最初に、対象 root から reachable raw Graph を一度 walk する。

1. cut leaf の dtype/rank と runtime binding shape から canonical symbolic leaves を作る。concrete extents は binding data であり fingerprint へ焼き込まない。
2. traced と同じ local analyzer で各 reachable op の `infer_output_meta`、effects、aliases、constraints を topological order に一度だけ実行する。
3. 同じ `SemanticProgramBuilder` / freeze に渡す。
4. active roots/wrt で normalize/slice する。
5. 既存 `AdTransformCache` と同じ semantic rule に渡す。
6. derivative executionも同じ runtime-owned prepared executor pathを通す。

capture 中は global metadata registry に触らない。materialization も可能な限り pure/local analysis とし、互換性上 registry publish が必要なら root 単位の boundary で一 batch に限定する。per-op global read/write を残して「compact tape」と呼ぶのは禁止である。

完全な `SemanticProgram` を各 tape node に保存しない。新しい per-root transform cache も作らない。既存の LRU `AdTransformCache` を使う。Graph からの materialization は O(reachable) で、per-op history 再解析はゼロにする。

### 5. 外部拡張の三層は維持する

Cycle 3 の合意も変えない。

1. concrete-only: typed executor と `TensorBackend` primitive の合成。`ExtensionOp` も AD も不要。新規 leaf kernel は公開 backend leaf API がない限り別問題。
2. runtime-integrated / no AD: typed executor + `ExtensionOp` + `infer_output_meta` + effects/aliases + stable adapter。unified eager/trace forward が動き、AD residual は保存しない。
3. runtime-integrated + AD: 2 に canonical semantic AD rule + `ResidualSpec` を追加する。eager 専用 tape hook、eager executor、provider-specific backward は不要。

stable adapter は engine/prepared/erased shim を隠す。internal proc macro の公開安定化は v0.3.0 の必須条件にしないが、out-of-tree fixture が public API だけで tier 2/3 を実装できなければ「外部拡張容易」は未達である。

## 実装順序

1. baseline を no-grad、all-untracked、grad-active、first AD、N forward + one AD に分け、DeepSeek が挙げた recorder 5 内訳を counter 化する。
2. targeted ensure の read-only hit を独立修正・計測する。
3. eager/compiled 共通の runtime-owned prepared execution helper を作り、`requires_grad` に関係なく immediate forward を通す。
4. ordinary autograd を active-edge semantics にし、all-untracked implicit recording を止める。detach / explicit functional trace の migration test を先に置く。
5. canonical AD rule に最小 residual mask を追加し、selected owner handles だけ保持する。
6. `extension::apply` から raw append と analysis を分離する。既存 Graph を tape carrier とし、eager scope Vec と per-op registry 登録を除去する。
7. eager root materialization を共通 analyzer / SemanticProgramBuilder に接続する。
8. first-party wrapper と out-of-tree fixture を単一 `apply_eager` へ移し、call-site direct bridge を公開経路から外す。

別 carrier、general plan cache、static eligibility bit、one-core-lowering fast path、recompute frameworkはこの順序に混ぜない。

## 最終 gate

### 保存量

- all-untracked chain: autograd node 0、saved tensor 0。
- grad-active op: raw Graph node は multi-output でも 1。
- saved tensor count/bytes は `ResidualSpec` と完全一致し、余分な input/output owner を保持しない。
- op payload、active edges、cut-leaf schema 以外の symbolic analysis stateを保存しない。
- provider/prepared/session/snapshot handle 保存 0。

### forward capture cost

- `extension::apply` 再入 0。
- `infer_output_meta` 0。
- global metadata registry read/write 0。
- `metadata_scopes` Vec merge/dedup 0。
- work は O(arity + selected residual count)、unary chain depth に非依存。
- multi-thread grad-active capture が global registry lock で直列化されない。

### AD materialization

- canonical inference は reachable op あたり最大 1 回/AD request。
- materialization は O(reachable)、common subgraph を stable id で一度だけ処理。
- eager active-sliced source と equivalent trace source が `semantic_eq`、fingerprint、JVP/VJP/HVP、typed error で一致。
- concrete shape 2/3 の shape-churn でも同じ symbolic fingerprint。
- mixed tracked/untracked、detach、multi-output、alias/view、effectful op、provider/rule replacement、failure atomicityを検証。

### extension execution

- `native-session` と `native-context` を別々に direct baseline へ非劣性。
- immediate eligible pathで SemanticProgram/fingerprint/compiler/schedule/admission 0、session/resource entry最大1。
- `prepare(Unsupported)` は allocation/planning前、promoted pathは非退行。
- eager/compiled が同じ engine selectionと prepared executor helperを使う。

### external authoring

- no-AD out-of-tree extension は tape API/ResidualSpec を実装しない。
- AD-enabled fixture は typed executor一本、canonical rule一つ、residual mask一つで eager/trace双方を通る。
- concrete-only fixture は runtime/AD依存なしで動き、既存 primitive合成という天井を明記する。

## 残る不一致と収束宣言

DeepSeek の Cycle 4 に対する実質的な不一致は一つだけだった。**symbolic `infer_output_meta` を毎 op capture の floor とみなすか**である。新しいユーザー方針に照らせば、答えは明白に「みなさない」。それを floor にする設計は PyTorch 型 tape ではなく、`TracedTensor` の incremental analysis state を eager autograd に温存した設計だからだ。

一方、DeepSeek の「現行 path のまま analysis だけ止めるな」「scope Vec を消せ」「Graph と同型の第二 carrier を作るな」は全て最終案へ取り込んだ。

したがって、次の二点を仕様上の定義として受け入れるなら残る不一致はない。

1. ordinary eager autograd は all-untracked history を暗黙保存せず、functional AD の全履歴 capture は明示 trace にする。
2. eager/trace の「同じ経路」とは、phase の実行時刻まで同じという意味ではなく、raw op/Graph、canonical analyzer/SemanticProgram/rule、runtime prepared executor が同じという意味である。

この定義以外で「PyTorch 級の最小保存」「single-op overhead 削減」「trace と全 phase 同時実行」を三つとも要求するのは論理的に不可能である。ユーザーの新方針は前二者と shared canonical path を選んだものと解する。**よって Cycle 5 で収束を宣言する。**# DeepSeek (Cycle 5): DeepSeek (Cycle 5): 収束宣言 + gate 精緻化 3 点 + Def 1 の意味論変更を明示承認事項として残す

# DeepSeek (Cycle 5) — 最終サイクル: GPT 案に収束。残るのは「gate の精緻化 3 点」と「意味論変更 1 点の明示承認」のみ

## 判定: 収束を宣言する。ただし無条件のゴム印ではない

GPT の Cycle 5 は、私の Cycle 4 の 4 論点（(1) 現行 recorder は毎 op `extension::apply` 再入、(2) `metadata_scopes: Vec` の O(depth²)/op 具現化、(3) `infer_output_meta` + global registry lock が記録 floor に入る、(4) `Graph<StdTensorOp>` と同型の第二 carrier は YAGNI）を**全て受け入れ、設計へ織り込んだ**。これを確認した上で、私の Cycle 4 の唯一の実質反論だった「`infer_output_meta` は capture の不可避な floor」も撤回する。撤回の理由は、GPT が指摘した「eager の下流 op は `TracedTensor` の incremental symbolic metadata を入力にしなくてよい」という隠れ前提の除去が正しいからだ。**「今 infer しない」と「concrete shape を焼き込む」の間に、op と辺だけ保持して materialize 時に一度だけ symbolic inference を走らせる第三の経路がある、という GPT の論法は成立する。**

以下、コード実読で裏付けを取り直した上で、最終的な合意設計と、私が gate として残す 3 点（+ 意味論変更 1 点）をまとめる。残る不一致は設計上の対立ではない。**gate の作りと、ユーザー承認が必要な semantics 変更の明示**だけである。

---

## 1. 事実確認（今回再読して確定）

- `crates/tenferro-runtime/src/extension.rs:110` の `apply` は、コード上で既に (a) `GraphBuilder` + `add_parent` + `add_operation` + `build()`（raw graph append）と (b) `register_scoped_graph_analysis` → `registered_meta` 読戻し → `traced_outputs_from_analysis`（analysis + TracedTensor 構築）の**二段に分離可能な構造**をしている。GPT の「`append_raw_extension_op` と `analyze_extension_graph` への因数分解」は、新機構の導入ではなく**既存コードの因子を公開するだけ**である。これは私の Cycle 4 第4節（Graph を保ったまま analysis だけ止める最小変更）と同一の結論であり、`SemanticRecordNode` 新設は確かに不要。
- `crates/tenferro-ad/src/eager.rs:2943` の `metadata_scopes: Vec<Arc<GlobalMetadataScope>>` と、`push_metadata_scope` の線形 dedup は実在。単項 chain で O(depth²)/op は事実。GPT の「leaf/binding 情報は一度だけ value に付け、親は Arc/stable id で共有」で廃止対象になる。
- `crates/tenferro-ad/src/transform_cache.rs:163` の `AdTransformCache`（`Mutex` + `LruCache`）は実在。ユーザー補足の「AD 変換は既にキャッシュ済み」と整合。新 per-root transform cache を増やさない、という GPT の自制は正しい。

---

## 2. 私が撤回した唯一の実質反論: `infer_output_meta` は capture floor ではない

Cycle 4 で私は「shape-churn 安定性（長さ 2 と 3 が同一 symbolic fingerprint）が concrete 化を禁じる以上、記録時の symbolic 推論は省略できない」と論じた。これは**「eager の下流 op が incremental symbolic metadata を入力として受け取る」という現行配線を前提にした場合に限り**真だった。GPT が正しく指摘した通り、その配線そのものを捨てる（eager tape は op + 順序付き辺 + cut leaf の最小 schema だけを持ち、下流 op の記録時に symbolic output metadata を供給しない）なら、`infer_output_meta` は AD 要求時に root から一度だけの topological walk で走らせれば足りる。concrete extent は runtime binding として扱い semantic fingerprint に入れない、という扱いも現行 `new_leaf` の `from_shared_tensor_value_symbolic_shape` と整合する。**ここは撤回し、収束する。**

ただし撤回には**一つの成立条件**を付す。それが次節の gate 追加 (A) である。

---

## 3. gate として残す 3 点（反論ではなく、成立条件の明示）

### (A) data-dependent shape を持つ op は、ResidualSpec の「小さな metadata mask」で concrete を宣言しなければ、materialize 時に shape を再現できない

GPT の第三の経路は「symbolic leaves から `infer_output_meta` で全出力 meta を再導出できる」ことを暗黙に前提する。これは **op の出力 shape が入力の symbolic dim だけで決まる場合に限る**。`nonzero`、`unique`、値依存 k の `topk`、動的 mask 系など、出力 shape が concrete 入力**値**に依存する op では、symbolic inference は出力 rank/extent を決められず、materialize 時の再導出が concrete forward と一致しない。

したがって v0.3.0 の `ResidualSpec` の mask は「rule が読む tensor の index」だけでなく「**rule が読む concrete shape/dtype の宣言**」を必ず含め、それを (i) eager 側の retention と (ii) traced AD の primal liveness/binding planning の**両方**で消費しなければならない。symbolic 再導出だけで済む op と、concrete 宣言が必須の op を区別せず「最小 metadata mask で足りる」と書くと、data-dependent shape の op だけ eager と trace で `semantic_eq` が破綻する。**gate に「value-dependent shape の op で、ResidualSpec 未宣言の concrete shape を rule が読む実装を contract violation として fixture で落とす」を明記せよ。** これは GPT の「rule が未宣言 primal を読む実装は contract violation として落とす」の拡張であり、新機構ではない。

### (B) multi-thread capture の「lock-free」は、stable-id 割当と residual handle 登録の構造指定なしには成立しない

GPT の gate「multi-thread grad-active capture が global registry lock で直列化されない」は、`metadata_scopes` Vec と global registry の read/write を消せば**自動的には**満たされない。消した後、残る共有可変状態は (i) leaf/binding の stable-id の割当、(ii) residual handle の登録、(iii) 出力 tensor が保持する tape handle の生成、の 3 つである。これらが単一の `Mutex` 付き arena やグローバル counter に寄れば、registry lock を arena lock に付け替えただけで直列化は温存される。

**要求:** capture は「各 op が不変の Arc ノード（op + 親参照 + 出力 stable id）を append するだけ」の immutable-link 構造とし、stable-id は atomic counter または per-thread 割当、residual handle は Arc 共有で lock を持たない、と**構造レベルで指定**せよ。その上で「unary chain を複数スレッドで同時に capture した際のスループットが depth に依存せず、スレッド数に線形スケールする」ことを counter ではなく実測 gate にせよ。現行 `extension::apply` の `register_scoped_graph_analysis` は lock を持ち、これが主直列化点だった。それを消した後の**新しい直列化点を作らない**ことが本題である。

### (C) ResidualSpec は「mask」に留め、traced AD の liveness 変更を回帰から守れ

ResidualSpec は eager retention と traced AD の primal liveness/binding planning の**両方**に効く cross-cutting な新機構である。これはユーザー方針「PyTorch 級の保存量」を実際に満たす唯一の機構なので正当だが、既存の全 semantic AD rule（einsum/linalg/fft）が暗黙に読んでいた primal を、宣言ベースに移行させる作業になる。**これを「rule が未宣言 primal を読む実装を contract violation として落とすテスト」なしに進めると、traced AD の数値結果を静かに変え得る。** 移行は (1) 全既存 rule に mask を宣言する、(2) 未宣言アクセスを検出する fixture/`debug_assert` を先に置く、(3) 既存 AD 数値テストを回帰 gate として全通し、の順で行うこと。recompute policy / checkpoint framework / executor-private auxiliary tensor の一般化を v0.3.0 に入れない、という GPT の YAGNI 判断は正しい。**mask は「semantic input/output index + 小 metadata」の最小形に限定し、provider slot / prepared executor / algorithm plan / session / snapshot を residual に保存しない**こともそのまま gate に残す。

---

## 4. 意味論変更 1 点: Def 1 は性能修正ではなく AD API の breaking change。明示承認が必要

GPT の Def 1「ordinary eager autograd は all-untracked history を暗黙保存せず、functional AD の全履歴 capture は明示 trace にする」は、私も技術的に同意する。PyTorch 級の保存量と「grad-active 入力が 0 なら autograd node を作らない」を両立するには、現行の「recording enabled なら全入力 untracked でも `record_eager_outputs` を呼ぶ」意味論（`eager_grad_recording_enabled()` が `no_grad` depth 0 なら true、という現行分岐）を捨てるしかない。**これは私が Cycle 4 で「性能修正のふりをした意味論変更を紛れ込ませるな」と警告した対象そのものである。**

したがって、これを「gate」ではなく「ユーザー/メンテナへの明示承認事項」として切り出す。具体的には:

1. **現行で可能な「untracked 中間値を後から `wrt` に選ぶ functional JVP/VJP」が、Def 1 後は暗黙には動かなくなる。** これは API の breaking change であり、v0.3.0 migration note で明記し、`detach` / 明示 trace/capture API への移行テストを**perf 実装より先に**置く（GPT の実装順序 step 4 はこの順序を既に守っている。維持せよ）。
2. 現行 in-tree の標準 extension crate やテストがこの暗黙録音に依存していないか、migration 前に棚卸しすること。
3. 代替案として「(i) recording-enabled の raw append（analysis なし・residual tensor なし）を残し、機能は維持するが graph 構造だけ O(1)/op で伸びる」選択肢が技術的に存在する。これは Def 1 より保存量で劣る（無制限 graph 成長 + スレッド連携の残存）が、**後方互換を優先するなら選び得る**。ユーザー方針「できるだけ減らす方向・PyTorch と同程度」の自然な読みは Def 1 だが、**暗黙機能の削除は私達二人の合意だけでは決められない**。ユーザーに (Def 1 / 代替案 i) の二者択一を明示して承認を得ることを、最終設計の前提条件として残す。

これは「残る不一致」ではない。設計は Def 1 で固定し、**承認が得られなかった場合の fallback が (i)** だと明記しておく、という処理である。

---

## 5. Def 2 の解釈を明示的に是認する（ユーザー方針の読解として）

「eager と trace の経路は同じに保ちたい」を「同じ全命令列を同じ時刻に通る」と読むのは不可能である（trace は append 時に kernel を実行しない）。GPT の Def 2「raw op/Graph、canonical analyzer/`SemanticProgram`/rule、runtime prepared executor が同じ、という意味で同一。差を許すのは phase の実行時刻のみ」が、ユーザー方針の「PyTorch 級の最小保存」「single-op 削減」「経路同一」を**同時に**成立させる唯一の整合的な解釈である。三つを「全 phase 同時実行」まで含めて要求するのは論理的に矛盾する、という GPT の判定に同意する。

これに伴い、eager から materialize した AD source と full trace を無条件に `semantic_eq` 比較してはならない、という GPT の但し書きも正しい。比較は「同じ active roots/wrt で slice し、non-active inputs を bindings に切った normalized AD source」に対して行う。ここを曖昧にすると、graph cut 意味論と経路同一を同時要求する矛盾した gate になる。**gate の比較対象を「normalized AD source」に明記**する。

---

## 6. 最終合意設計（私の側から最終確認した形）

1. forward は grad の有無で分岐させない。単一 `apply_eager` → snapshot から family/engine 解決 → `ExtensionEngine::prepare` → runtime-owned `execute_prepared_operation`（`native-session` / `native-context` 共通）→ concrete output contract 検証 → 必要な時だけ autograd capture commit。`prepare(Unsupported)` は allocation/planning 前に早出し（Cycle 2 の gate 維持）。call-site session callback は公開経路から外す。
2. AD tape の carrier は既存 `Graph<StdTensorOp>`。`extension::apply` を `append_raw_extension_op`（op + 親参照 + 出力 id）と `analyze_extension_graph`（leaf schema からの canonical 推論）に因数分解。trace は 1→2、eager autograd は 1 のみ。`metadata_scopes: Vec` 持ち回りと per-op registry 登録を廃止。
3. 保存 tensor は canonical AD rule の `ResidualSpec`（semantic input/output index + 小 shape/dtype metadata mask）で限定。rule 未宣言 primal アクセスは contract violation。provider/prepared/session/snapshot handle の保存は 0。
4. 初回 AD 要求時に root から一度だけ reachable walk し、共通 analyzer / `SemanticProgramBuilder` / freeze / 既存 `AdTransformCache` を通す。O(reachable)、per-op 再解析ゼロ、per-root transform cache 新設なし。
5. 外部拡張の三層（concrete-only / runtime no-AD / runtime + AD）は Cycle 3 合意を維持。AD 有効 fixture は typed executor 一本 + canonical rule 一つ + residual mask 一つで eager/trace 双方を通る。
6. 実装順序は GPT の 8 step を維持。step 4（active-edge semantics 化）と step 6（append/analysis 分離）は migration test を先行させる。

---

## 7. 結論

**収束を宣言する。** 私の Cycle 4 の実質反論（`infer_output_meta` = capture floor）は撤回し、GPT の「op + 辺だけ保持し materialize 時に一度だけ symbolic inference」を採る。残るのは設計上の不一致ではなく、以下の 4 項目の gate 精緻化と承認事項である:

1. **(A)** data-dependent shape op の concrete 宣言を ResidualSpec に必須化し、未宣言アクセスを fixture で落とす。
2. **(B)** multi-thread capture の lock-free を、stable-id 割当・residual 登録の構造指定 + 実測スケーリング gate として具体化する。
3. **(C)** ResidualSpec 移行は既存 AD rule 全件の宣言 → 未宣言検出 → 回帰全通し、の順で、recompute/checkpoint の一般化を混ぜない。
4. **Def 1 は AD API の breaking change であり、ユーザー/メンテナの明示承認を前提とする。fallback は「raw append のみの暗黙録音を残す」(i)。**

この 4 点が受け入れられれば、v0.3.0 の eager extension 経路統一 + single-op 削減の設計は、私の側から異論がない。
# User (Feedback): Def 1（PyTorch型 active-edge semantics、all-untracked 暗黙録音廃止）を breaking change として採用。DeepSeek の gate 精緻化3点も受入。完全収束

## User Feedback（最終決定）

Breaking change を許容した上で、最善の設計を目指す。

→ **Def 1 を採用する**（ordinary eager autograd を PyTorch 型に: grad-active 入力が 0 なら autograd node 0。all-untracked の暗黙録音は廃止し、functional AD の全履歴 capture は明示 trace API に移行）。これは AD API の breaking change として v0.3.0 migration note に明記する。

DeepSeek の gate 精緻化 3 点（(A) data-dependent shape の ResidualSpec concrete 宣言必須、(B) multi-thread capture の lock-free 構造指定、(C) ResidualSpec 移行の回帰手順）も受け入れる。

これでデベートは完全収束。
