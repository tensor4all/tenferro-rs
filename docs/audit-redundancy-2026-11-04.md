# 重複・冗長コード監査 (2026-11-04)

対象: `origin/main` @ `75eed022d`。P2 ブランチ `feat/2417-p2-context-core` は対象外。

本ドキュメントは**判断材料**であり、リファクタは行っていない。コードは 1 行も変更していない。
PR は文書 1 ファイルのみ。

## 方法

- 規模: `crates/` + `ext/` の Rust 897 ファイル / 385,480 行。
- 手段:
  1. `rg` による全数検索（定数の二重定義、同名関数、`pub(crate)` の参照ゼロ）。
  2. 関数本体の正規化ハッシュ照合（空白・コメント除去）。同一本体を全数検出。
  3. 識別子・リテラルを抽象化した構造ハッシュ照合。型だけ違う複製を検出。
  4. 呼び出し元カウント（`pub(crate)` / private のみを死んだコードと判定）。
  5. 検出候補は**両側の実体を読んで**確認。関数名が同じでも本体が違うものは除外。
- 除外: `target/`、生成物、`third_party/`、`extern/`、`ext/tenferro-cpu-tblis/`（空）。
- ランキング: 削減見込み行数 × 確度。上位 25 件。

**注意**: 行数は「消せる見込み」の概算であり、検証済みパッチの行数ではない。
risk は「その統合が数値・性能・公開契約を壊しうるか」の評価。

## カテゴリ別件数

上位 25 件と、その外側（参考節・触らない一覧のカテゴリ 3 項目）を含めた全検出での件数。

| # | カテゴリ | 上位25件 | 全検出 |
|---|----------|---------|--------|
| 1 | 同一ロジックの複製 | 19 | 30 |
| 2 | 死んだコード | 1 | 1 |
| 3 | 素通しのラッパ | 0 | 5 |
| 4 | 定数・表の二重定義 | 1 | 1 |
| 5 | 冗長な計算・変換 | 0 | 1 |
| 6 | 標準機能・既存APIの再実装 | 1 | 1 |
| 7 | テストの重複 | 3 | 5 |
|   | **合計** | **25** | **44** |

カテゴリ 3 と 5 は行数が小さい、または行数削減が 0（呼び出し当たりの固定費）なので、
行数×確度のランキングでは上位 25 件に入らない。参考節と触らない一覧に理由付きで残した。

上位 25 件の内訳: カテゴリ 1 が 19 件・削減見込み合計 ~1,590 行、カテゴリ 7 が 3 件で ~324 行、
カテゴリ 2・4・6 が計 3 件で ~104 行。上位 25 件の削減見込み合計は **約 2,020 行**。
テスト重複（カテゴリ 7）は共有機構が既にリポジトリ内にある
（`include!` / `tests/support/mod.rs`）ため、削減の実現性は高い。

## 上位 25 件

---

### R01 — コア演算 enum と派生 match 表が 4 重（`tenferro-core-ops`）

- category: 1
- A: `crates/tenferro-core-ops/src/catalog.rs:240` `define_std_tensor_op!` → `primitive_kind` は `:384`、`sample_from_kind` は `:446`
- B: `crates/tenferro-core-ops/src/catalog.rs:682` `define_exec_op!` → `primitive_kind` は `:803`、`sample_from_kind` は `:1041`、`from_std_tensor_op` は `:866`
- 追加: `crates/tenferro-core-ops/src/catalog.rs:566` `define_elementwise_fusion_op!` が
  elementwise 部分集合を 3 度目として持ち、`primitive_kind` は `:651`、`elementwise_fusion_op` は `:1006`
- duplicated: 54 variant の `StdTensorOp`（graph IR）と `ExecOp`（lowered IR）が variant 集合を
  共有するのは設計通りだが、その**対応表 4 本（各 50〜60 arm）が手書き**。
- est_lines_removed: ~380
- risk: medium
- why_it_exists: 2 つの IR を分ける判断は意図的。ただし同じファイルの
  `primitive_ops!`（`:81`）が既に「1 つの宣言的テーブルから生成する」パターンを示しており、
  他の 3 マクロはそれに倣っていない。
- evidence:
  - `:384` `pub fn primitive_kind(&self) -> Option<$crate::PrimitiveOpKind> { let kind = match self { Self::Add => $crate::PrimitiveOpKind::Add, ...`
  - `:803` 同じ 50 arm が `Self::Add => $crate::PrimitiveOpKind::Add` で並ぶ

### R02 — CPU / CUDA の prepare パイプラインがほぼ同一の約 250 行

- category: 1
- A: `crates/tenferro-cpu/src/runtime_adapter.rs:315` `prepare_cpu`, `:379` `cpu_operation_kind`,
  `:437` `minimum_specialization_requirements`, `:466` `concrete_axes_for_rank`,
  `:477` `merge_specialization_requirements`, `:491` `merge_input_requirements`,
  `:545` `core_operation_name`, `:603` `checked_specialization_heap_retained_bytes`,
  `:640` `checked_sum`, `:646` `checked_sum_options`
- B: `crates/tenferro-gpu/src/cuda/runtime_adapter.rs:324` `prepare_cuda`, `:391` `cuda_operation_kind`,
  `:449`/`:479`/`:490`/`:504`（同じ 4 ヘルパ）, `:558` `core_operation_name`,
  `:616`/`:653`/`:659`（同じ 3 ヘルパ）
- duplicated: `CoreSemanticOp` → kind の表、特殊化要件の合成算術、保持バイト数の会計、
  57 行の `core_operation_name` match が両側に同一。差分は kind 型名、
  `ExecutionContextIdentity`、class-id 定数 2 個、`expect` 文言の `"CPU"`/`"CUDA"` のみ。
- est_lines_removed: ~250
- risk: medium
- why_it_exists: バックエンド分離（`REPOSITORY_RULES.md` の「各バックエンドが自分の
  adapter を持つ」）。ただし共有半分（`PrepareCapability` / `SpecializationRequirements`）は
  既に `tenferro-runtime` が所有しており、パラメータ化できる。
- 検証: 対応する 11 関数を両側で行単位比較した。合計 262 行のうち 250 行以上が同一で、
  相違は型名（`CpuPreparedKind`/`CudaPreparedKind`）と `expect` 文言のみ。
- evidence:
  - cpu: `let minimum = minimum_specialization_requirements(actual_kind, context.inputs())?;`
  - gpu: 同じ行が `prepare_cuda` に存在し、`core_operation_name` の
    `CoreSemanticOp::PadToMatch { .. } => "pad_to_match",` / `_ => UNKNOWN_CORE_OPERATION`
    に至るまで一致

### R03 — complex⇔real の reinterpret ビューが 12 個

- category: 1
- A: `crates/tenferro-tensor/src/types.rs:3968`（`Complex32→f32` の `as_real_view`）と
  `:4023`（`Complex64→f64`）
- B: `crates/tenferro-tensor/src/types.rs:4077`/`:4131`（`as_complex_view` の C32/C64）、
  `:5314`/`:5373`/`:5430`/`:5489`（mutable view）、
  `:10693`/`:10769`/`:10845`/`:10924`（`TypedTensor` 本体）
- duplicated: `validate_representation_pair` → `reinterpret_{complex_to_real,real_to_complex}_layout`
  →（mutable のみ）`validate_mutable_no_overlap` → backend buffer 拒否 →
  `reinterpret_host_slice{,_mut}` → view 再構築、という 29 行の手順が 12 回。
  3968 と 4023 は型名を抽象化すると `DType::C32/F32` と `C64/F64` の 1 行以外完全一致。
- est_lines_removed: ~180
- risk: low
- why_it_exists: Rust に「同じ形の float/complex ペア」を表す trait bound が無く、
  組み合わせごとに固有 impl を書いた。mutable 版だけ overlap 検査が増える。
- evidence:
  - `:3968` `validate_representation_pair(op, DType::C32, DType::F32)?; let layout = reinterpret_complex_to_real_layout(self.shape(), self.strides(), self.offset(), self.buffer.len(), op)?;`
  - `:4023` 同文で `DType::C64, DType::F64`

### R04 — `ext/df64-proof` のテストヘルパ `payload` / `runtime_with_module` が 8 + 7 個

- category: 7
- A: `ext/df64-proof/tests/einsum.rs:32` `payload`
- B: `ext/df64-proof/tests/connected_conversion_ad.rs:30`, `extension_qr.rs:27`,
  `connected_qr_ad.rs:31`, `einsum_ad.rs:33`, `qr_derivative_references.rs:43`,
  `ext/scalar-consumer-application/tests/mixed_core_operations.rs:50`, `later_backward.rs:30`
- 追加（`runtime_with_module`、11 行 ×7）: `ext/df64-proof/tests/extension_ad.rs:41`,
  `einsum.rs:43`, `connected_conversion_ad.rs:41`, `connected_qr_ad.rs:42`, `einsum_ad.rs:44`,
  `extension_boundaries.rs:35`, `qr_derivative_references.rs:54`
- duplicated: `external_payload()` から `downcast_ref::<Df64>()` する 10 行と、
  拡張モジュールを登録した `Runtime` を組む 11 行。
- est_lines_removed: ~136
- risk: low
- why_it_exists: **共有機構は既にある**。`ext/df64-proof/tests/support/mod.rs` が存在し、
  `einsum_ad.rs:10` / `extension_boundaries.rs:6` / `qr_derivative_references.rs:20` などが
  既に `mod support;` で取り込んでいる。それでも `payload`/`runtime_with_module` は
  各ファイルに残っている（移し忘れ）。
- evidence: `einsum_ad.rs:33` の `payload` と `connected_qr_ad.rs:31` の `payload` が本体一致

### R05 — QR/eigh ゲージ補正がスカラー型ごとに 2 重定義

- category: 1
- A: `crates/tenferro-linalg/src/extension/gauge.rs:223` `canonicalize_eigh_gauge_f64` /
  `:245` `_f32`、`:319` `canonicalize_qr_gauge_f64` / `:347` `_f32`、
  `:441` `max_abs_pivot_f64` / `:454` `_f32`
- B: `crates/tenferro-linalg/src/extension/gauge.rs:267` `canonicalize_eigh_gauge_c64` /
  `:293` `_c32`、`:375` `canonicalize_qr_gauge_c64` / `:408` `_c32`、
  `:467` `max_abs_pivot_c64` / `:480` `_c32`
- duplicated: f64≡f32、c64≡c32 がバイト単位で一致（実数版と複素版は pivot のゼロ判定が
  `abs()` と `norm()` で本当に異なるので、そこは統合不可）。
- est_lines_removed: ~128
- risk: medium
- why_it_exists: 呼び出し側 `apply_positive_diagonal_qr_gauge` は既に
  `gauge_pair_operands_mut::<T>` で総称化されているのに、葉だけ型ごとに閉じている。
  `#[inline]` は付いておらず、数値カーネルのホットループでもない（分解後の後処理）。
- evidence:
  - `:223` `if vectors[base + pivot + n * col] < 0.0 { for row in 0..n { ... } }`
  - `:245` 同じ行が `vectors: &mut [f32]` になっただけ
  - `:267` `let norm = diagonal.norm(); if norm == 0.0 { continue; }`（複素版のみ）

### R06 — テストヘルパ `engine()` が `tenferro-gpu` の 14 ファイルに複製

- category: 7
- A: `crates/tenferro-gpu/tests/seam_structural.rs:12`
- B: `crates/tenferro-gpu/tests/seam_linalg_aux.rs:9`, `seam_dtype.rs:12`, `seam_raw.rs:9`,
  `seam_staging.rs:13`, `seam_publication.rs:16`, `seam_dot.rs:13`, `seam_launch.rs:17`,
  `seam_capability.rs:40`, `seam_indexing.rs:10`, `seam_transfers.rs:18`, `seam_runtime.rs:10`,
  `crates/tenferro-gpu/src/cuda/dispatch/prepare_tests.rs:7`,
  `crates/tenferro-gpu/src/cuda/structural/tests.rs:18`
- duplicated: `if !gpu_available() { return None } Some(CudaEngine::open(...).expect(...))` の 8 行。
- est_lines_removed: ~104
- risk: low
- why_it_exists: テストバイナリは独立コンパイル単位。**共有手段は既にある**:
  `crates/tenferro-gpu/tests/storage_provider_cuda_progress.rs:5` が
  `include!("support/storage_provider.rs");` を使っている。新しい規約の発明は不要。
- evidence: 14 箇所すべて本体一致

### R07 — unary / ordered の `*_read_with_pool` が dtype 分岐表を手書き

- category: 1
- A: `crates/tenferro-internal-cpu-kernels/src/elementwise.rs:1865` `neg_read_with_pool`
  （`conj`:1964, `abs`:2067, `sign`:2179 も同型）
- B: `crates/tenferro-internal-cpu-kernels/src/elementwise.rs:2292` `maximum_read_with_pool`
  （`minimum`:2398, `compare`:2514 も同型）
- duplicated: `CpuReadView` の全 variant を列挙する約 34 行の dtype 表が 7 本。
  差はカーネル識別子と末尾のエラー文字列だけ。
- est_lines_removed: ~90
- risk: low
- why_it_exists: 同じクレートが 1 画面先（`elementwise.rs:894` の `add_read_with_pool`）で
  既に `dispatch_read_presets!` マクロを使っている。unary/ordered は後から op 単位で
  足されたためマクロ適用から漏れた。
- evidence:
  - `:1865` `CpuReadView::C32(t) => Ok(Tensor::from_typed::<Complex<f32>>(typed_neg_with_pool(buffers, &t, exec)?)),`
  - `:2179` 同じ 34 行で `typed_sign_with_pool` に置換しただけ

### R08 — テストヘルパ `external()` が 15 ファイルに複製

- category: 7
- A: `ext/df64-proof/tests/extension_ad.rs:23`
- B: `ext/df64-proof/tests/external_dtype_boundaries.rs:13`, `einsum.rs:25`, `dispatch_overhead.rs:96`,
  `scratch_allocation.rs:63`, `connected_conversion_ad.rs:23`, `extension_qr.rs:20`,
  `connected_qr_ad.rs:24`, `retention_controls.rs:26`, `ad_rule_boundaries.rs:22`,
  `einsum_ad.rs:26`, `extension_boundaries.rs:24`, `qr_derivative_references.rs:36`,
  `ext/scalar-consumer-application/tests/mixed_core_operations.rs:43`, `later_backward.rs:23`
- duplicated: `Tensor::external(ErasedHostTensor::new(TypedTensor::from_host_vec_col_major(...)))` の 6 行。
- est_lines_removed: ~84
- risk: low
- why_it_exists: R04 と同じ。`ext/df64-proof/tests/support/mod.rs` が既にあるのに未適用。
- evidence: 15 箇所すべて本体一致

### R09 — シンボリック次元の AST が 2 本、算術評価器も 2 本

- category: 1
- A: `crates/tenferro-internal-ops/src/dim_expr.rs:25` `DimExpr`（評価器 `:99` `eval`）、
  変換 `:340` `from_concrete` / `:426` `From<usize>`
- B: `crates/tenferro-internal-ops/src/sym_dim.rs:34` `RawSymDim`（定数畳み込み `:167`
  `raw_constant_value`）、変換 `:183` `From<usize>`
- 追加: 相互変換 `sym_dim.rs:89` `from_dim_expr` と `:189` `raw_to_dim_expr` が鏡像
- duplicated: `Const` / 軸参照 / `Add` / `Sub` / `Mul` / `FloorDiv` / `Min` / `Max` が
  variant 単位で一致し、どちらも `checked_add`/`checked_sub`/`checked_mul`/`checked_div`/
  `min`/`max` の再帰評価を持つ。
- est_lines_removed: ~80
- risk: medium
- why_it_exists: `DimExpr` は op payload 用の直列化可能な形、`RawSymDim` は
  tensor-id ベースの軸参照を持つ traced 用。AST と算術は共有できるが、
  **評価器は同一ではない**（`DimExpr::eval` は詳細な `DimExprEvalError` を返し、
  `raw_constant_value` は非定数なら `None`）。統合時はこの差を保つ必要がある。
- evidence:
  - `dim_expr.rs:25` `Const(usize), InputDim { input_idx, axis }, Add(Box<DimExpr>, Box<DimExpr>), ...`
  - `sym_dim.rs:34` `Const(usize), TensorAxis { tensor_id, axis }, Add(Box<RawSymDim>, Box<RawSymDim>), ...`

### R10 — `checked_sum` / `checked_sum_options` が 9 + 6 個

- category: 1
- A: `crates/tenferro-runtime/src/runtime/cache.rs:1925`
- B: `crates/tenferro-runtime/src/runtime/preparation.rs:2632`, `runtime/schedule.rs:1114`,
  `runtime/signature.rs:405`, `program/metadata.rs:308`, `program/semantic.rs:307`,
  `shape_constraint/solver.rs:99`, `crates/tenferro-cpu/src/runtime_adapter.rs:640`,
  `crates/tenferro-gpu/src/cuda/runtime_adapter.rs:653`
- duplicated: `try_fold(0usize, |sum, v| sum.checked_add(v))` の 5 行。
  `checked_sum_options`（`Option<usize>` 版）も 6 箇所に同じ本体。
- est_lines_removed: ~65
- risk: low
- why_it_exists: 各モジュールが private ヘルパを持ち、モジュール横断の共有場所が無かった。
- evidence: 7 ファイルすべて正規化ハッシュ一致

### R11 — `simd_math::apply` が f64 本体を f32 用に複製、テストも複製

- category: 1（テスト部分は 7）
- A: `crates/tenferro-cpu/src/simd_math.rs:41` `apply` の f64 arm
  （`TypeId` 判定 → `from_raw_parts` 2 回 → `kernels64::*` の 8 分岐）
- B: `crates/tenferro-cpu/src/simd_math.rs:41` の f32 arm（同手順を `kernels32::*` で）、
  テストは `:124` `f64_kernels_match_the_scalar_definition_over_a_compact_buffer` と
  `:148` `f32_kernels_match_...`
- duplicated: `unsafe` 再解釈 + 8 分岐のブロック、および同じ比較本体のテスト。
- est_lines_removed: ~60
- risk: low
- why_it_exists: ホットパス。ベクトル化リーフからクロージャ間接を排除するため幅ごとに
  書き下している。`macro_rules!` で両 arm を生成しても同じコードが展開される。
- evidence: f64 arm `match kernel { VectorKernel::Exp => kernels64::exp(src, dst), ... }`
  と f32 arm の同型

### R12 — `dtype()` / `shape()` の 7 分岐 match が 4 + 3 個

- category: 1
- A: `crates/tenferro-tensor/src/types.rs:7608`（`TensorView::dtype`）,
  `:7977`（`TensorViewMut::dtype`）, `crates/tenferro-tensor/src/default_scalars.rs:103`, `:460`
- B: `crates/tenferro-tensor/src/types.rs:7631`, `:7989`,
  `crates/tenferro-tensor/src/default_scalars.rs:126`（`shape()` 側 3 個）
- duplicated: `Self::F32(_) => DType::F32, ...` の 7 arm と
  `Self::F32(t) => t.shape(), ...` の 7 arm。
- est_lines_removed: ~55
- risk: low
- why_it_exists: 4 つの enum がそれぞれ手書き。同ファイルは既に
  `tensor_view_mut_constructor!` マクロを使っており、マクロ化は既存の流儀に沿う。
- evidence: `types.rs:7608` と `default_scalars.rs:103` が正規化ハッシュ一致

### R13 — AD の 2 つのキャッシュが LRU 維持機構を丸ごと複製

- category: 1
- A: `crates/tenferro-ad/src/transform_cache.rs:225` `AdTransformCacheStore`
  （`set_limits` / `clear` / `stats`）
- B: `crates/tenferro-ad/src/eager.rs:3778` `PreparedDerivativeCache`
  （`set_limits` / `clear` / `stats`）
- duplicated: 上限変更→退避、clear 回数を保った統計リセット、保持バイト数の減算を伴う
  LRU 退避。`clear` は 8 行がバイト一致。
- est_lines_removed: ~50
- risk: medium
- why_it_exists: 意味変換キャッシュと prepared derivative キャッシュでキー・値・
  バイト見積りが異なるため独立に育った。退避・統計の機構だけ共有できる。
- evidence:
  - A `:231` `let clears = self.stats.clears.saturating_add(1); self.entries.clear(); self.stats = CacheStats {`
  - B `:3784` 同じ 3 行

### R14 — `checked_shape_product` / `checked_product` が 9 箇所、0 次元の扱いが不揃い

- category: 6（既存の公開 API の再実装）
- A: `crates/tenferro-tensor/src/validate/mod.rs:126` `checked_shape_product`（**既存の公開 API**）
- B: `crates/tenferro-fft/src/cpu.rs:656`, `crates/tenferro-fft/src/cuda/mod.rs:519`,
  `crates/tenferro-cpu/src/structural.rs:74`, `crates/tenferro-cpu/src/indexing.rs:1373`,
  `crates/tenferro-linalg/src/gpu/linalg.rs:3980`,
  `crates/tenferro-linalg/src/cpu/linalg/faer_linalg.rs:374`,
  `crates/tenferro-linalg/src/cpu/linalg/lapack_linalg/helpers.rs:592`,
  `crates/tenferro-einsum/src/planning/plan.rs:370`
- duplicated: `shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))` の再実装。
  しかも正規版だけが `if shape.contains(&0) { return Ok(0) }` の短絡を持つため、
  `[usize::MAX, 2, 0]` で正規版は `Ok(0)`、複製側は overflow エラーになる（挙動差）。
- est_lines_removed: ~45
- risk: low（ただし挙動差の解消を含むのでテストで固定すること）
- why_it_exists: クレートごとにエラー型（`crate::Result` / `tenferro_tensor::Result` /
  `EinsumResult`）が違うため写した。`map_err` 1 行で共有できる。
- evidence:
  - 正規 `:126` `if shape.contains(&0) { return Ok(0); }` + try_fold
  - `fft/cpu.rs:656` try_fold のみ

### R15 — indexed-plan キャッシュの 4 ファミリが同じ lookup/insert 本体

- category: 1
- A: `crates/tenferro-cpu/src/indexed_plan_cache.rs:253` `gather`
- B: `crates/tenferro-cpu/src/indexed_plan_cache.rs:266` `scatter`, `:279` `dynamic_slice`,
  `:292` `dynamic_update_slice`、さらに `:346` `indexed_plan_retained_bytes` が同じ 4 分岐を再掲
- duplicated: hit→return / miss→compile→insert の 13 行と、保持バイト数の 4 分岐 match。
- est_lines_removed: ~40
- risk: low
- why_it_exists: `strided_kernel` の 4 つの `Erased*Plan` 型に共通 trait が無く、
  variant ごとに書き下した。
- evidence: `:253` `if let Some(IndexedPlan::Gather(plan)) = self.lookup(&key) { return Ok(plan); }`
  と `:292` の `IndexedPlan::DynamicUpdateSlice(plan)` 版

### R16 — scope ツリーの `materialize` / `as_slice` が 2 個

- category: 1
- A: `crates/tenferro-runtime/src/metadata.rs:98` `as_slice`
- B: `crates/tenferro-runtime/src/shape_constraint/mod.rs:167` `as_slice`
- duplicated: `OnceCell` に詰める DFS（`seen_scopes` / `seen_nodes` / `visited_nodes`）の 18 行。
  `materialize` / `materialize_with_visit_count` も対で複製。
- est_lines_removed: ~36
- risk: medium
- why_it_exists: `GlobalMetadataScope` と `ShapeConstraintScope` が別型。
  共通の `ScopeNode<T>` に括れるが、2 つの意味体系を 1 実装へ寄せる判断が要る。
- evidence: 両側とも `self.node.materialized.get_or_init(|| { ... })` で同一手順

### R17 — `apply_slice_axis_config` が eager / traced で複製

- category: 1
- A: `crates/tenferro-ad/src/shape_packing.rs:146`
- B: `crates/tenferro-runtime/src/shape_packing.rs:231`
- duplicated: 全軸の starts/limits/strides 初期化、take 選択の skip、step/range 検証、
  選択軸の上書きという純粋な選択準備。
- est_lines_removed: ~35
- risk: medium
- why_it_exists: eager と traced が別々の `AxisSelection` 型を持ち、意図的に異なる
  validation エラー variant を返す。純粋準備だけ共有できる。
- evidence: 両側とも `let mut starts = vec![0; shape.len()]; let mut limits = shape.to_vec(); let mut strides = vec![1; shape.len()];`

### R18 — `tenferro-tensor` の storage group に参照ゼロの `pub(crate)` が 3 個

- category: 2
- A: `crates/tenferro-tensor/src/storage/group.rs:876` `from_backend_root`
- B: `crates/tenferro-tensor/src/storage/group.rs:1374` `backend_root_buffer`,
  `:1392` `backend_root_buffer_mut`
- duplicated: （重複ではなく死んだコード）3 つとも `rg` で定義以外の参照ゼロ。
  `from_backend_root` は `:841` `from_backend_buffer` に、root アクセサ 2 つは
  `:1283` `backend_buffer` / `:1381` `backend_buffer_mut`（実際に呼ばれる側）に置き換え済み。
- est_lines_removed: ~33
- risk: low
- why_it_exists: slot 添字アクセサに置き換える前の配管。`storage/mod.rs:3` が
  `#![allow(dead_code)]` なので警告が出ない。
- evidence: `rg -n "\bbackend_root_buffer\b" crates/ ext/` → 定義 1 件のみ

### R19 — `index_select_config` が eager / traced で複製

- category: 1
- A: `crates/tenferro-ad/src/shape_packing.rs:48`
- B: `crates/tenferro-runtime/src/shape_packing.rs:91`
- duplicated: host 位置の検証、`[positions.len(), 1]` の I64 テンソル化、
  単軸 `GatherConfig` の構築。
- est_lines_removed: ~30
- risk: medium
- why_it_exists: eager は具体的な gather 入力を要し、traced は出力 shape hint と
  graph-build エラーを追加する。位置→indices/config の構築だけ共有できる。
- evidence: 両側とも `let config = GatherConfig { offset_dims, collapsed_slice_dims: vec![axis], ...`

### R20 — `matrix_dims` と QR 状態検証が faer / LAPACK で複製

- category: 1
- A: `crates/tenferro-linalg/src/cpu/linalg/faer_linalg.rs:1885` `matrix_dims`,
  `:1899` `validate_compact_state`
- B: `crates/tenferro-linalg/src/cpu/linalg/lapack_linalg/helpers.rs:516` `matrix_dims`,
  `crates/tenferro-linalg/src/cpu/linalg/lapack_linalg/qr.rs:67` `validate_state`
- duplicated: `matrix_dims` は 10 行完全一致。`validate_compact_state` / `validate_state` は
  19 行が構造一致（rank・factor 長・batch 整合）。
- est_lines_removed: ~29
- risk: low
- why_it_exists: `native`（faer）と `blas`（LAPACK）が排他的代替バックエンドで、
  それぞれ独立に書かれた。`helpers.rs` 側は `pub(crate)` なので共有は容易。
- evidence: `if input.shape().len() != 2 { return Err(tenferro_tensor::Error::rank_mismatch(op, 2, input.shape().len())); }` が両側に同一

### R21 — capability の別名・行ビルダ・`UNKNOWN_CORE_OPERATION` が CPU/CUDA で二重

- category: 4
- A: `crates/tenferro-cpu/src/capability.rs:8` `const U/F/N`、`:42` `owned_read`、`:61` `same_dtype`、
  `:94` `dot_native`、`crates/tenferro-cpu/src/runtime_adapter.rs:30` `UNKNOWN_CORE_OPERATION`
- B: `crates/tenferro-gpu/src/cuda/capability.rs:8` `const U/F/N`、`:46` `owned_level`、
  `:69` `same_dtype`、`:73` `dot_native`、`crates/tenferro-gpu/src/cuda/runtime_adapter.rs:219`
  `UNKNOWN_CORE_OPERATION`
- duplicated: 3 つの `SupportLevel` 別名、行ビルダ const fn 群、`"unknown-core-operation"` リテラル。
  **145 行の `*_CAPABILITIES` 表そのものは重複ではない**（CPU は `write_output: F`、
  CUDA は `U`、複素超越関数の行も違う）。
- est_lines_removed: ~26
- risk: low
- why_it_exists: バックエンド分離。表を導出せず書き下す判断は意図的だが、
  別名とビルダは共通化できる。
- evidence: cpu `const fn same_dtype(...) { owned_read(op, dtype, dtype, level) }` と
  gpu の同型 `owned_level(...)`

### R22 — `select_outer_product_label_order` が `tenferro-einsum` 内に 2 個

- category: 1
- A: `crates/tenferro-einsum/src/eager.rs:213`
- B: `crates/tenferro-einsum/src/builder.rs:105`
- duplicated: 22 行が完全一致（ラベル順序の妥当性検査 → `target_labels.to_vec()`）。
- est_lines_removed: 22
- risk: low
- why_it_exists: eager 経路とグラフ構築経路が別モジュールで、共有 util を経由していない。
- evidence: 両者とも `let Some(target_labels) = target_labels else { return canonical_labels.to_vec() };`

### R23 — `default_pinv_rtol` が `tenferro-linalg` 内に 3 個

- category: 1
- A: `crates/tenferro-linalg/src/traced.rs:1963`
- B: `crates/tenferro-linalg/src/eager_composites.rs:295`（A と完全一致）、
  `crates/tenferro-linalg/src/tensor_ext.rs:2768`（`shape: &[usize]` を受ける版）
- duplicated: dtype で `f32::EPSILON` / `f64::EPSILON` を選び `eps * max_dim` する 11 行。
- est_lines_removed: ~22
- risk: low
- why_it_exists: traced / eager / concrete の 3 経路が独立に育った。
- evidence: `DType::F32 | DType::C32 => f32::EPSILON as f64,` が 3 箇所で同一

### R24 — 軸正規化が eager / traced で複製

- category: 1
- A: `crates/tenferro-ad/src/shape_packing.rs:8` `normalize_existing_axis`、`:24` `normalize_insert_axis`
- B: `crates/tenferro-runtime/src/shape_packing.rs:23` `normalize_existing_axis`、`:51` `normalize_insert_axis`
- duplicated: 負軸を checked 減算で正規化し、既存軸と挿入軸の境界を区別する算術。
- est_lines_removed: ~20
- risk: medium
- why_it_exists: eager 側は tensor エラー、traced 側は `ErrorPhase::GraphBuild` を付ける。
  算術だけ共有し、エラー変換を境界に残せる。
- evidence: 両側とも `rank.checked_sub(axis.unsigned_abs()).ok_or_else(|| ...)?`

### R25 — `kernel_dtype` / `typed_bytes` が `tenferro-cpu` 内で 2 個ずつ

- category: 1
- A: `crates/tenferro-cpu/src/reduction.rs:153` `kernel_dtype`, `:169` `typed_bytes`
- B: `crates/tenferro-cpu/src/indexing.rs:81` `kernel_dtype`, `:97` `typed_bytes`
- duplicated: `DType` → `KernelDType` の 7 arm match（`External(_) => unreachable!` の
  INVARIANT コメントまで同一）と、`from_raw_parts` による `typed_bytes`（SAFETY コメントまで同一）。
- est_lines_removed: ~20
- risk: low
- why_it_exists: `KernelDType` の語彙を各モジュールが自前で持つ。`typed_bytes` は
  `tenferro-cpu-fused/src/lib.rs:174` と `tenferro-internal-cpu-kernels/src/read_into.rs:124`
  にもう 1 つずつある（計 4 個）。
- evidence: `reduction.rs:153` と `indexing.rs:81` の本体一致

---

## 触らない一覧（意図的な重複・除外）

1. **`#[global_allocator]` のカウンタアロケータ（`alloc`/`dealloc` が 10 テストファイル）**
   — `#[global_allocator]` は 1 バイナリに 1 つ。テストバイナリは独立コンパイル単位なので
   共有不可能。例: `crates/tenferro-runtime/tests/binary_allocations.rs:37`。
2. **`ext/df64-proof` ⇔ `ext/bf16-proof` の `ExtensionOp` 実装ボイラープレートと `offset_for`**
   — 2 つの proof クレートは相互依存せず、2 つの異なるスカラーで拡張 SPI が成立することを
   示すのが目的。同一本体は 68 行あるが、共有には 3 つ目のクレートが必要で
   proof の独立性を損なう。`offset_for` は `df64-proof/src/extension.rs:2091` と
   `bf16-proof/src/einsum.rs:252`。
3. **`cpu/linalg/faer_linalg.rs` と `cpu/linalg/lapack_linalg/` の演算本体** —
   `native` / `blas` の排他的代替バックエンド。`cpu/linalg/mod.rs:3,7` の
   `#[cfg_attr(..., allow(dead_code, unused_imports))]` が示す通り片方は常に
   ビルド対象外。共通化は抽象の二重化を招く。R20 の純粋ヘルパのみ対象。
4. **`LinalgBackend` / `FftBackend` の owned 版と `_read` 版の対** —
   `crates/tenferro-linalg/src/backend.rs:361,544,877,916` の `_read` 既定実装は
   `Error::Unsupported` を返し、各バックエンドが別実装を持つ。read 経路は
   materialize を避ける性能設計（`cpu/backend.rs:132,344,367` の `faer_strided_read_ok` 分岐）。
   統合すると転送が増える。
5. **`traced.rs` と `eager_composites.rs` の `vector_norm` / `matrix_norm` の並行実装**
   — 一方はグラフを組み、他方は即時実行する。実行モデルが違うため本体は共有できない
   （共有できるのは R23 の `default_pinv_rtol` だけ）。
6. **`tenferro-tensor/src/types.rs` の `tensor_view_*_constructor!` 展開** — 生成コード。
7. **`crates/tenferro-internal-ops/src/ad/context.rs:487–820` の 8 個の `pub fn`** —
   `extend_metadata`, `insert_shape_source`, `into_typed_source`, `refresh_global_metadata`,
   `set_transpose_primal_outputs`, `take_deferred_shape_error`,
   `transpose_primal_outputs_were_used`, `with_extension_ad_dispatcher` は `rg` で
   定義以外の参照ゼロだが、**P2（`feat/2417-p2-context-core`）が追跡中の context core
   公開面**であり本監査の対象外。
8. **`crates/tenferro-runtime/src/runtime/preparation.rs:514`
   `logical_retained_bytes_for_test`** — `#[cfg(test)]` かつ
   `#[allow(dead_code, reason = "available to integration tests of root sharing/accounting")]`。
   作者が意図的に残したテストフック。
9. **`crates/tenferro-gpu/src/cuda/engine.rs:164,171,318` の `#[doc(hidden)] pub`** —
   `vendor_resource_stats` / `clear_vendor_resources` / `runtime_mut` は下流向けフック。
10. **`crates/tenferro-xla` の lowering 公開入口 4 つ** —
    `src/lib.rs:80,112` の自由関数と `src/executor.rs:205,236` の `XlaExecutor` メソッドが
    すべて `src/lowering/mod.rs:11` へ転送する。公開 API の削除は破壊的変更なので、
    統合するなら非推奨化を挟む。カテゴリ 3 の代表例として記録。
11. **`crates/tenferro-linalg/src/cpu/tlinalg_workspace.rs:42,46,75,79`** —
    `tlinalg_blas::Workspace` / `IndexWorkspace` の trait 実装。外部クレートが呼ぶ。
12. **`crates/tenferro-internal-extension-macros/src/lib.rs:118`
    `derive_extension_family_id`** — マクロ展開用の生成コード。
13. **ベンチハーネスの `engine()` / `bench_threads()` / `PHYS_DIM` / `CHIS`** —
    ベンチは独立した測定単位。共有するとハーネス間の独立性が失われる。
    例: `crates/tenferro-cpu/benches/blas1.rs:14` と `dot_general_overhead.rs:15`。
14. **`crates/tenferro-cpu/src/capability.rs:98` と
    `crates/tenferro-gpu/src/cuda/capability.rs:98` の `*_CAPABILITIES` 表本体** —
    エントリごとの support level が意図的に違う（`write_output: F` vs `U`、
    複素超越関数 `N` vs `U`）。`tests/seam_capability.rs` が実測に対して固定している。
15. **`crates/tenferro-cpu/src/plan_cache.rs:224-256` の `gather`/`scatter` 等の一行転送** —
    `IndexedPlanCache` のメソッドが `&mut self` で `Mutex` を取るため、計算前に
    ロックを解放する役割がある。消すとロックが呼び出し側へ漏れる。
16. **`crates/tenferro-cpu/src/simd_math.rs:14-16` の `pow` 不在** —
    クレート doc が「SLEEF 由来の `pow` は libm より遅い」として意図的な欠落を明記し、
    `elem-simd-math/docs/findings-pow.md` の実測を参照している。
    「カーネル集合を完成させる」変更はしない。
17. **`crates/tenferro-tensor/src/types/col_major.rs:29` の `linear_offset`** —
    `#[inline(always)]` の固定ランク・ホットパス。R14 系の一般形と統合しない。

## 参考: 25 件外（確度は高いが行数が小さい、または判断が要る）

- **`plan_spec_hash` / `plan_spec_retained_bytes` が `tenferro-einsum` 内に 2 個ずつ**
  （`src/eager_ad.rs:500,506` ⇔ `src/extension.rs:1168,1174`、19 行）。
  下請けの `saturating_sum` / `vec_retained_bytes` は既に `src/cache.rs:40,48` から
  共有されており、上位 2 関数だけが写されている。カテゴリ 1・risk low。
- **`with_linalg_backend` ⇔ `with_fft_exec_session`**
  （`tenferro-linalg/src/tensor_ext.rs:2451` ⇔ `tenferro-fft/src/lib.rs:1532`、19 行）。
  「マーカーを probe → exec session を取り直す」同型実装。カテゴリ 1・risk medium。
- **`decode_col_major_index` が `ext/tropical` 内に 2 個**
  （`src/einsum.rs:904` ⇔ `src/extension.rs:1402`、18 行）。カテゴリ 1・risk low。
- **`flat_to_multi` 系が 3 個**
  （`tenferro-tensor/src/types.rs:11671`、`src/backend.rs:685`、`src/types/col_major.rs:29`、22 行）。
  `types.rs` 側は `#[allow(dead_code)]` で crate 内テスト用に残置。カテゴリ 1・risk low。
- **`col_major_strides` が 2 層に**
  （`tenferro-tensor-core/src/lib.rs:355` ⇔ `tenferro-tensor/src/types.rs:8846`、15 行）。
  差はエラー型だけ。カテゴリ 1・risk low。
- **インデックスの rank/bounds 検証が 2 個**
  （`tenferro-tensor/src/types.rs:8863` ⇔ `:8900`、20 行が文字単位で同一）。カテゴリ 1・risk low。
- **`require_same_shape` ⇔ `ensure_same_shape`**
  （`tenferro-internal-cpu-kernels/src/scalar_ops.rs:26` ⇔ `src/elementwise.rs:270`、12 行）。
  同一クレート内で同名別名。`scalar_ops.rs:22` `strides_for` は `col_major_strides` への
  素通し（カテゴリ 3）。risk low。
- **`tenferro-cpu/src/test_support.rs` に「erased engine で closure を走らせる」が 3 通り**
  （`:42` `on_engine`、`:50` `with_cpu_session`、`:59` `run_erased`、15 行）。
  `:48` のコメントが「古い呼び出し側をそのまま読めるように」と理由を書いている。
  カテゴリ 3・risk low。
- **`broadcast_binary_in` / `broadcast_ternary_in` が typed / erased で 2 組**
  （`tenferro-runtime/src/tensor.rs:470,482` ⇔ `src/typed_tensor.rs:673,685`、20 行）。
  差は `TensorRead::from_tensor(x)` と `T::tensor_read(x)` だけ。カテゴリ 1・risk low。
- **`ensure_float_or_complex` が `tenferro-linalg` 内に 2 定義**
  （`src/validation.rs:73` ⇔ `src/tensor_ext.rs:2752`、8 行）。エラー型が
  `Result` と `tenferro_tensor::Result` で違うため分かれた。カテゴリ 1・risk low。
- **`promote_dtype_pow_like` → `promote_dtype` への素通し**
  （`tenferro-runtime/src/shape_infer.rs:67` → `:43`）。カテゴリ 3。
- **`eigvals` → `eig_values` への素通し**（`tenferro-linalg/src/traced.rs:1451`）。カテゴリ 3。
- **型付き linalg が能力 probe を 1 呼び出しで 2 回行う**（カテゴリ 5・行数削減 0）—
  `tenferro-linalg/src/tensor_ext.rs:2162` の `TypedTensorLinalgExt::svd` が
  `with_linalg_backend` を通したうえで `svd_read(backend)` を呼び、`:1940` の
  `TensorReadLinalgExt::svd_read` が再び `with_linalg_backend` を通る。
  1 回の型付き呼び出しで downcast probe が最大 4 回（本来 2 回）。
  svd/qr/lu/cholesky ほか全型付きメソッドに波及する。
- **`tenferro-gpu/tests` の `as_bytes`(6) / `f64_bytes`(5) / `from_f64_bytes`(4) /
  `region_indices`(2) / `write_through_view`(2)** — R06 と同じ根本原因、行数は小さい。
- **`crates/tenferro-cpu/src/tests.rs:156,169` と
  `crates/tenferro-linalg/src/cpu/tests/mod.rs:135,148` の `matmul_f64` / `matmul_c64`** —
  テスト用行列積ヘルパが 2 クレートで同型。
- **`ext/tropical/src/extension.rs:208` と `ext/sparse/src/extension.rs:280` の `prepare`**
  （27 行が構造一致）。ただし別クレートの拡張 trait 実装で、共有は拡張の独立性を損なう。

## 監査中に見つけた文書とコードの不一致

本監査の対象範囲で、文書とコードが食い違っている箇所が 2 つある。
（重複ではないので上表には含めていない。）

1. **`AGENTS.md:177` は「No `unwrap()`/`expect()` in library code」と定めるが、
   ライブラリコードに `.expect(...)` がある。**
   `crates/tenferro-cpu/src/runtime_adapter.rs:460,473,521`、
   `crates/tenferro-gpu/src/cuda/runtime_adapter.rs:473,486,534`。
   いずれも builder 不変条件の表明だが、規則の文言は例外を認めていない。
2. **`REPOSITORY_RULES.md:911` は「the engine is the first argument」と定めるが、
   実際は最後の引数である。**
   `crates/tenferro-runtime/src/tensor.rs:25`
   `fn add(&self, rhs: &Tensor, session: &mut dyn Engine)`。
   規則の意図が「`self` を除いた先頭」なのか「引数リストの先頭」なのか不明瞭で、
   コードは後者に従っていない。

## 検証状況

- 読み取り専用。`cargo` のビルド・テスト・bench は一切実行していない。
- 行数は概算。`risk` は統合時の影響評価であって、実測ではない。
- 上位 25 件のうち R02 は対応する 11 関数を両側で行単位比較（262 行中 250 行以上が同一）、
  R01・R07・R11・R15 は行番号と本体一致を機械照合で確認し、代表箇所を目視した。
  R03・R05・R09・R10・R13・R14・R16・R19・R20・R24 は両側の実体を読んで確認した。
- 文書中の全 92 個の `path:line` 参照を機械検証した（ファイル存在と行番号の範囲）。
- 未確認として残る範囲: `cuda/dispatch/`、`cuda/structural/`、`cuda/fusion/` の
  内部本体（入口のみ確認）、`ext/` の参照ゼロ `pub(crate)` 全数掃引。
