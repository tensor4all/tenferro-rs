# #2552 公開 API 文法：実装指示書

この文書は、[#2552](https://github.com/tensor4all/tenferro-rs/issues/2552) で決まった公開 API の文法を実装する人（人間とエージェント）への指示書です。何を作り、何を消し、どの順で進め、何をもって完了とするかを、迷わずに作業できる粒度で書きます。

- 設計の理由と全体像は [提案書](./proposal.html) を読んでください。この文書は理由を繰り返しません。
- ここに書いた判断は、#2552 の決定を実装の言葉に直したものです。スレッドで決まっていなかった点は、本書が **[本書で決定]** として決めています。異論があれば実装前に #2552 で議論し、この文書を直してから進めてください。黙って別の形で実装しないでください。
- 基準は `origin/main` の `90348dc0` です。行番号は目安で、作業時に必ず読み直してください。

---

## 0. 作業を始める前に

### 0.1 必ず読むもの

| 順 | 文書 | 読む理由 |
|---|---|---|
| 1 | `AGENTS.md`、tensor4all 共有ルール（`rules/index.md`） | リポジトリ全体の作法 |
| 2 | `REPOSITORY_RULES.md` | 実装前・PR 作成前に読み直す（AGENTS.md の指示）。ただし「Public API Convention」節は本作業で書き換える対象（§8） |
| 3 | `PERFORMANCE_TIPS.md` 全文 | 実行経路に触るため。特に「Materialization And Copies」 |
| 4 | [`api-consolidation-policy.md`](../api-consolidation-policy.md) の R3 と R8 | レイアウト変換の扱い。実体化ポリシーは撤回済み |
| 5 | #2552 の決定コメント群 | 下の §0.3 |
| 6 | プロトタイプブランチ `prototype/2552-api-grammar` の `ext/api-grammar-proof/` | 各試作の実装と `findings/p1.md`〜`p6.md`。新しい形の動く見本 |

### 0.2 ツールチェーン

- ワークスペースは `rust-version = "1.96"` を要求します。1.96 以上の stable で作業してください。
- プロトタイプは 1.93 nightly に `RUSTFLAGS="-Zcrate-attr=feature(atomic_try_update)"` と `--ignore-rust-version` を付けて動かしました。この回避策を PR の検証に使ってはいけません。
- `cargo public-api` は nightly の rustdoc JSON を使います。CI の `placement` ジョブが既にインストールしているものと同じ版を使ってください。

### 0.3 決定事項の出典

| 項目 | 出典（#2552 のコメント） |
|---|---|
| 文法・7 ルール・決定 1〜6 | 6760660549（Decision: one call grammar） |
| 結果型、エラー、einsum の出力先、FFT、例外の考え方 | 6762219245（inventory v0.1） |
| v0.1 への修正（policy の位置、`Svd<V, R>` など） | 6762266144（review of v0.1） |
| 例外表 E01〜E09、エラー supertrait の見送り、プランとキャッシュ所有者 | 6762340249（inventory v0.2） |
| プロトタイプの結果 | 6763055062 |
| 実体化ポリシーの撤回、公開カウンタを要件にしない | 6763124519、6764423689（後者が最終） |

`MaterializationPolicy`、`InputMaterializationForbidden`、公開の正規化カウンタは **作りません**。設計資料やプロトタイプにこれらが出てきても、実装しないでください。

---

## 1. 作業の進め方

### 1.1 ブランチと PR

- 文法の移行は、`origin/main` から切った作業ブランチ **`api-grammar-0.12`** 1 本に積みます（AGENTS.md「Working Branches For Multi-Commit Work」）。
- main への PR はバッチが終わってから 1 回だけ出します。ファミリーごとに main へ PR を出してはいけません。理由は次の2つです。
  - 文法の変更と規約ファイルの書き換えは同じ PR に入れる決まりです（提案書ルール 7）。途中で main に入れると、新旧どちらの文法も通る期間ができます。
  - CI のコストを抑えるためです。
- 作業ブランチの中では、下の段階ごとにコミットを分け、各段階の終わりで `cargo test --workspace` が通る状態にします。
- 例外として、§9 の独立した修正（#2565 の共役バグ、隠れたコピーの削減）は、それぞれ main から切った別ブランチで、通常の PR にします。文法の変更を待つ必要はありません。
- この移行は破壊的変更で、0.12 で 1 回だけ出します。旧 API を非推奨として残すことも、互換シムを置くこともしません（REPOSITORY_RULES.md「No compatibility shims」）。

### 1.2 段階

| 段階 | 内容 | 節 | 前提 |
|---|---|---|---|
| S0 | 現行の公開 API スナップショットを記録し、CI に入れる | §2 | なし。main への単独 PR にしてよい |
| S1 | 共通の型を追加する（`Dest`、軸、制御用の列挙型、結果構造体） | §3 | S0 |
| S2 | 演算レジストリと生成マクロ、例外表を作る | §4 | S1 |
| S3 | linalg を移行する | §5 | S2 |
| S4 | einsum を移行する | §6 | S2。#2565 の修正が main に入っていること |
| S5 | FFT を移行する | §7.1 | S2 |
| S6 | core 演算を移行する（runtime と ad） | §7.2 | S2 |
| S7 | 規約・仕様・ガイド・スキルを書き換え、旧チェッカーを整理する | §8 | S3〜S6 |
| S8 | スナップショットを更新し、PR を出す | §10 | S7 |

S3〜S6 は互いに独立しているので、作業ブランチから枝を切って並行して進め、終わったら作業ブランチへマージしてかまいません。ただし同じファイルを触る場合は順番に進めてください（`tenferro-tensor` の共通型は S1 で確定させておく）。

---

## 2. S0：現行 API の基準スナップショット

**目的。** 移行前の公開 API を記録し、移行中と移行後の差分を、レビューで必ず目に入るようにします。新しい設計の承認ではありません。

**作るもの**

1. `api/` ディレクトリをリポジトリ直下に作ります。
2. `scripts/api-snapshot.sh` を作ります。
   - `--write`：公開クレートごとに `cargo public-api -p <crate> --features <固定の組み合わせ>` を実行し、`api/<crate>.txt` に書き出します。
   - `--check`：同じ出力を作り、`api/` のファイルと `diff` します。差分があれば終了コード 1 です。
   - 対象クレートは workspace の公開クレート全部です（`publish = false` は除く）。ext の独立マニフェスト（`ext/tropical`、`ext/sparse`）は `--manifest-path` で別に扱います。
   - feature の組み合わせはクレートごとにスクリプトの中で固定し、そのことを `api/README.md` に書きます。最低限、既定の feature と、`autodiff` / `native` のように公開面を増やす feature を有効にした組み合わせの2つを取ります。
3. `.github/workflows/ci.yml` の `placement` ジョブ（`cargo-public-api` と nightly を既にインストールしている）に、`scripts/api-snapshot.sh --check` を実行するステップを足します。
4. `api/README.md` に、スナップショットの更新方法（`--write` を実行して差分をコミットする）と、「スナップショットはレビューの証拠であって、振る舞いの証明ではない」ことを書きます。

**完了条件**
- main で `--check` が成功し、CI の `placement` ジョブが緑になること。
- わざと公開関数を 1 つ足すと `--check` が失敗すること。これは PR の Verification 節に出力を貼って示します。

---

## 3. S1：共通の型

置き場所の原則は次のとおりです。
- 複数ファミリーが使う型は `tenferro-tensor` に置きます。
- 1 つのファミリーだけが使う型は、そのファミリーのクレートに置きます。

新しい公開型には、どれも `/// # Examples` を付け、doctest が実際に実行されるようにします（`ignore` と `no_run` は使わない）。

### 3.1 出力先 `Dest`（`tenferro-tensor`）

```rust
/// Where an `_into` operation writes, and how it combines with what is there.
#[derive(Debug)]
#[non_exhaustive]
pub enum Dest<W> {
    /// out = op(...); the previous contents are never read.
    Overwrite(W),
    /// out = op(...) + out.
    Accumulate(W),
    /// out = alpha * op(...) + beta * out; with beta == 0 the previous
    /// contents are never read (NaN or -0 in out does not leak).
    Scaled { out: W, alpha: ContractionScalar, beta: ContractionScalar },
}

impl<W> Dest<W> {
    pub fn overwrite(out: impl Into<W>) -> Self;
    pub fn accumulate(out: impl Into<W>) -> Self;
    pub fn scaled(out: impl Into<W>, alpha: impl Into<ContractionScalar>,
                  beta: impl Into<ContractionScalar>) -> Self;
}
```

- `W` は erased では `TensorWrite<'a>`、typed では `TypedTensorWrite<'a, T>` です。どちらも `&mut Tensor` や `&mut TypedTensor<T>` からの `From` が既にあります。
- `Dest` は「どこに、どう書くか」だけを表します。共役は演算の意味なので、`Dest` には入れず `XxxOptions` に入れます（v0.1 の合意）。
- 既存の `DotGeneralAccumulation { lhs_conj, rhs_conj, alpha, beta }`（`crates/tenferro-tensor/src/backend.rs:387`）との対応は次のとおりです。
  - `Overwrite` は `DotGeneralAccumulation::overwrite(dtype)` に当たります。
  - `Accumulate` は `add_to(dtype)` に当たります。
  - `Scaled` は `scaled(alpha, beta)` に当たります。
  - `lhs_conj` / `rhs_conj` は演算の options から埋めます。
  - この変換は `Dest` の非公開ヘルパー 1 か所に書きます。プロトタイプ P2 の `p2_einsum_dest.rs` に動く見本があります。
- `DotGeneralAccumulation` 自体はバックエンドのフック（`TensorDot::dot_general_read_into_accum` など）の引数として残します。公開の演算シグネチャからは消します。**[本書で決定]** バックエンド trait は公開面の op 文法の対象外です（例外 E10。§4.3 で表に追加する）。
- `Accumulate` と `Scaled` を受け付けるのは、レジストリで `into: [overwrite, accumulate, scaled]` と宣言した演算だけです。宣言していないモードは、実行前に `Error::Unsupported` を返します。

### 3.2 軸（`tenferro-tensor-core` に関数、型は `isize`）

- 公開シグネチャの軸はすべて `isize` にします。負の値は末尾から数えます（`-1` が最後の軸）。
- 長さ、次元、形状、IR の識別子は `usize` のままです。対象は「軸を指す引数」だけです。名前で言えば `axis`、`axes`、`axis_a` / `axis_b`、`dim`（軸の意味のもの）、`perm`（転置の置換）です。
- 正規化は 1 か所に書きます。

  ```rust
  pub fn normalize_axis(axis: isize, rank: usize) -> Result<usize, ValidationError>;
  pub fn normalize_axes(axes: &[isize], rank: usize) -> Result<AxisVec, ValidationError>;
  ```

  範囲外は既存の `ValidationError::AxisOutOfBounds` を返します。`AxisOutOfBounds { axis: usize, .. }` は元の `isize` を保持できないので、`axis: isize` に変えます。重複は `DuplicateAxis` を返します。
- FFT は既に `isize` なので、そのまま使えます。移行対象は runtime（約 77 シグネチャ）、linalg の `norm`（`dim` 引数）、ad の `EagerSession` の固有メソッドです。数は棚卸しのスクリプトで数えた概数なので、作業時に grep で数え直してください。

### 3.3 制御フラグの列挙型（`bool` を置き換える）

公開シグネチャの制御用 `bool` は、次の列挙型に置き換えます。真偽値のデータ（`TypedTensor<bool>`、述語、マスク）は対象外です。

| 旧 | 新（置き場所） |
|---|---|
| `triangular_solve(b, left_side, lower, transpose_a, unit_diagonal)` | `TriangularSolveOptions { side: Side, uplo: Uplo, op: MatOp, diag: Diag }`（linalg） |
| `norm(ord, dim, keepdim)` | `NormOptions { ord: NormOrd, axes: Option<AxisVec>, keep_dims: KeepDims }`（linalg） |
| `svd_with_exec(.., full: bool)` | E08 の対象なので今回は触らない（§5.6） |
| `dot_general_with_conj(rhs, config, lhs_conj, rhs_conj)` | `dot_general_with(&rhs, config, &DotGeneralOptions { lhs: Conj, rhs: Conj }, exec)`（tensor） |
| `LinalgBackend::full_piv_lu_solve(.., transpose_a)` | バックエンド trait なので E10 として対象外 |

```rust
pub enum Side { Left, Right }
pub enum Uplo { Lower, Upper }
pub enum MatOp { None, Transpose, Adjoint }   // 既存の transpose_a は None/Transpose に写す
pub enum Diag { NonUnit, Unit }
pub enum Conj { No, Yes }
pub enum KeepDims { No, Yes }
```

- どれも `Clone, Copy, Debug, PartialEq, Eq, Hash` を derive し、`#[non_exhaustive]` は付けません（値が増える見込みがないため）。
- 既定値は `Default` で明示します。現行の暗黙の既定に合わせて、`Side::Left`、`Uplo::Lower`、`MatOp::None`、`Diag::NonUnit`、`Conj::No`、`KeepDims::No` とします。
- `MatOp::Adjoint` は現行の4つの bool では表せません。**[本書で決定]** 複素入力で `Adjoint` が来た場合は、実装できていなければ `Error::Unsupported` を返します。黙って `Transpose` として扱ってはいけません。

### 3.4 Options の形（全演算共通の約束）

すべての演算に `XxxOptions` を 1 つ置きます。設定項目のない演算も例外ではなく、空の `DetOptions` などを置きます（決定 2）。

```rust
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct SvdOptions {
    pub shape: SvdShape,        // Thin / Full
    pub gauge: SvdGauge,
    pub driver: SvdDriver,
    pub derivative_eps: f64,
}
impl SvdOptions {
    pub fn shape(mut self, v: SvdShape) -> Self { self.shape = v; self }
    // 1 フィールドにつき 1 セッター。名前はフィールド名と同じ。
}
```

- `#[non_exhaustive]` を付けるので、利用者は構造体リテラルで作れず、`Default::default()` とセッターで作ることになります。
- セッターの名前はフィールド名と同じにし、`with_` は付けません。現行の `SvdOptions::gauge()` などがこの形なので、それに合わせます。
- `Copy` は付けられる場合だけ付けます。`EinsumOptions` はツリーを持つので `Clone` だけです（§6.2）。
- 意味の違いはすべてここに入れます。実行器、出力先、入力は入れません。

### 3.5 結果構造体（linalg）

`crates/tenferro-linalg/src/results.rs` を新設し、次の構造体を定義します。

```rust
pub struct Svd<V, R = V> { pub u: V, pub s: R, pub vh: V }
pub struct Qr<V> { pub q: V, pub r: V }
pub struct Lu<V, R = V> { pub p: V, pub l: V, pub u: V, pub parity: R }
pub struct FullPivLu<V, R = V> { pub p: V, pub l: V, pub u: V, pub q: V, pub parity: R }
pub struct Eig<C> { pub values: C, pub vectors: C }
pub struct Eigh<V, R = V> { pub values: R, pub vectors: V }
pub struct Slogdet<V, R = V> { pub sign: V, pub log_abs_det: R }
// 既存のものはそのまま使う
pub use crate::rank_revealing_qr::RankRevealingQrResult; // <V, I = V>
pub use crate::householder::HouseholderQr;
```

- フィールドはすべて `pub` にします。分解して受け取る `let Svd { u, s, vh } = ...` を主な使い方とするためです。derive は `Clone, Debug` だけにします。
- フィールドの順序と意味は、現行のタプルの順序と一致させます。現行のタプル別名は `TypedSvd<T>` = `(U, S, Vh)`、`TypedLu<T>` = `(P, L, U, parity)`、`TypedFullPivLu<T>` = `(P, L, U, Q, parity)`、`TypedEig<T>` = `(values, vectors)` です（`tensor_ext.rs:272-329`）。`eigh` は現行 `(values, vectors)` の順です。移行時に、タプルの順序から構造体のフィールドへの対応をテストで固定してください。
- LU の `p` は、現行どおり置換行列のテンソルです。ピボットの添字ベクトルに変えてはいけません（v0.1）。
- 型パラメータは次のように決まります。
  - erased の層では、`V`、`R`、`C` はどれも `Tensor`（traced は `TracedTensor`、eager は `EagerTensor`）です。
  - typed の層では、たとえば `Svd<TypedTensor<T>, TypedTensor<T::Real>>`、`Eig<TypedTensor<T::Complex>>` になります。`T::Real` と `T::Complex` は `LinalgScalar` の関連型で、既にあります。
- タプル別名 `TypedSvd`、`TypedLu`、`TypedFullPivLu`、`TypedEig`、`TypedRankRevealingQrResult` と、`private` モジュール内の `Svd<T>` は削除します。

---

## 4. S2：演算レジストリ、生成マクロ、例外表

### 4.1 形

**[本書で決定]** レジストリは Rust の宣言マクロ呼び出しとして書き、それ自体を唯一の出典にします。TOML などの別ファイルは作りません。理由は、別ファイルにすると、宣言と生成されたコードの一致を確かめる仕組みがもう 1 つ必要になるからです。

- マクロの定義は既存の `tenferro-internal-extension-macros` に `api_ops!` として置きます。
- 呼び出しは、ファミリーごとに `crates/<family>/src/api/registry.rs` に 1 つ書きます。core は `tenferro-runtime` に置きます。
- core の `primitive_ops!`（`crates/tenferro-core-ops/src/catalog.rs:81`）は IR プリミティブの一覧で、公開 API の一覧ではありません。流用せず、そのまま残します。公開 API のレジストリとは別物です。

1 エントリの形は次のとおりです（項目は v0.1 と v0.2 が求めたもの）。

```rust
api_ops! {
    family: linalg,
    op svd {
        operands: (a),                     // 受け手のあとに続く必須の入力。受け手を含む
        options: SvdOptions,
        output: Svd<V, R>,                 // V/R/C/I の対応で書く
        dtypes: floating_or_complex,       // 入力 dtype と出力 dtype の関係
        layers: [concrete, eager, traced],
        into: [],                          // 空なら _into を生成しない
        layout: "Strided rank-2 read directly; negative strides and \
                 non-compact batches of rank >= 3 are gathered once \
                 (same placement and dtype).", // R3 の文書化
        errors: [Validation, NumericalFailure(NonConvergence), Unsupported],
        omissions: [],                     // 層やバックエンドで未対応のもの（理由つき）
    }
    op solve {
        operands: (a, b),
        options: SolveOptions,
        output: V,
        dtypes: floating_or_complex,
        layers: [concrete, eager, traced],
        into: [overwrite],
        layout: "...",
        errors: [Validation, NumericalFailure(Singular)],
        omissions: [],
    }
}
```

### 4.2 マクロが生成するもの（と生成しないもの）

生成するもの
- 各層のトレイトのメソッド宣言と、本体への転送。`op`、`op_with`、`op_into`（`into` が空でないときだけ）の3形です。`op` は `op_with(&Default::default(), ..)` に転送するだけのものにします。
- rustdoc。これには次が含まれます。
  - 1 行の説明
  - `layout` の文
  - `errors` の一覧
  - レジストリのエントリに書いた例を使った `# Examples`
- テスト用のメタデータ定数 `pub(crate) const API_OPS: &[ApiOpDescriptor]`。

生成しないもの
- 数値計算の本体。本体は手で書いた非公開関数 `fn svd_impl(view: TensorRead<'_>, opts: &SvdOptions, exec: &mut dyn Engine) -> Result<Svd<Tensor>>` などで、マクロはそこへ転送するだけです（v0.2 D）。
- `Dest` を書き込みの形に変える処理。これは §3.1 の共通ヘルパーで行います。

REPOSITORY_RULES.md「Wrapper DRY And Codegen」が求めることも守ります。
- 生成された名前、ドキュメント、feature gate、エラーの振る舞いが、呼び出し側から見て分かること。
- 生成された転送のどれかが壊れたら失敗するテストを置くこと。具体的には §4.4 のテストです。

### 4.3 例外表

レジストリと同じマクロ呼び出しの中に `exceptions { ... }` 節を置きます。v0.2 の形 `id, scope, category, owner_module, reason, verification, exit_condition` をそのまま使います。

| ID | 範囲 | 今回の扱い |
|---|---|---|
| E01 | 消費型 FFT（`fft_in_place` / `ifft_in_place`） | 存続。受け手を session に移す（§7.1） |
| E02 | `FftExecutor`（キャッシュ所有者） | 存続。引数順だけ文法に合わせる（§7.1） |
| E03 | `ConcreteEinsumPlan::clear_cpu_cache` / `cpu_cache_stats` | 存続。名前は変えない |
| E04 | TraceContext / compile / prepare / run のライフサイクル | 文法の対象外。別スナップショットで固定 |
| E05 | 記述子だけの操作とメタデータ問い合わせ（`shape`、`dtype`、`into_dynamic`、ビュー作成など） | 文法の対象外。データをコピーしないことをテストする |
| E06 | 承認した外部型（`num_complex::Complex` など）と bridge | シンボル単位の許可リスト。`api/allowlist.txt` |
| E07 | `TraceValue` の einsum 入口 | **退役**（§6.4）。表には履歴として残す |
| E08 | `*_with_exec` 自由関数 6 本 | 今回は存続。§5.6 |
| E09 | パーサーとプランナーの領域エラー、所有権を返す失敗 | 存続 |
| E10 | **[本書で決定・追加]** バックエンド trait（`Engine`、`TensorDot`、`LinalgBackend`、`FftBackend` など）と、その `*_read_into_accum` フック | op 文法の対象外。provider 向けの契約として別に扱う |

例外表のエントリは、モジュール単位ではなく、具体的なシンボルの名前で書きます。

### 4.4 生成物を守るテスト

`crates/<family>/src/api/tests/registry.rs` に、少なくとも次のテストを置きます。

1. **転送のテスト：** `API_OPS` の全エントリ × 全層について、`op` と `op_with(&Default::default())` が同じ結果を返すこと。数値は固定の小さな入力で比べ、bit 一致を要求します。
2. **入力の形のテスト（ルール 2）：** 1 つの演算について、次の5種類の入力が同じ結果を返すこと。数値計算の契約が違う場合は、許容誤差を明記します。
   - 所有テンソル
   - コンパクトな借用ビュー
   - 転置したビュー
   - 負ストライドのビュー
   - typed のテンソル
3. **`_into` のテスト：** `into` を宣言した演算で、次の3点を確かめること。
   - `Overwrite` は、NaN で埋めた出力先を読まない。
   - `Accumulate` は加算になる。
   - `Scaled { beta: 0 }` は、NaN や `-0` を漏らさない。
4. **宣言どおりのテスト：** 宣言していない `into` モードと `omissions` に書いた組み合わせが、実行前に `Unsupported` を返し、出力先を変えないこと。

### 4.5 生成するドキュメント

`docs/spec/generated/api-ops.md` に、次の3つを書き出します。
- 演算 × 層の表
- 接尾辞の一覧
- 例外表

書き出しは `cargo test -p <family> api_docs_are_current` が行います。テストは生成した内容とコミット済みのファイルを比べ、違えば失敗し、`UPDATE_API_DOCS=1` で上書きします。`docs/_quarto.yml` の Specification 節に、このファイルを追加します。

---

## 5. S3：linalg

### 5.1 消すもの

- トレイト：`TensorLinalgExt`、`TensorReadLinalgExt`、`TypedTensorLinalgExt`、`TracedTensorLinalgExt`、`TraceContextLinalgExt`、`EagerSessionLinalgExt`、`EagerTensorLinalgExt`。
- 名前：`*_read`、`svd_full`、`svd_with_options`、`qr_with_options`、`eigh_with_options`、`pinv_with_rtol`、`solve_read_into`。
- タプル別名（§3.5）と `prelude` 内の旧トレイト。

### 5.2 作るもの

| 層 | トレイト | 受け手 | 形 |
|---|---|---|---|
| concrete | `LinalgOps` | `AsView` を実装するすべての型（§5.3） | `a.svd(exec)`、`a.svd_with(&opts, exec)`、`a.solve_into(&b, &opts, exec, dest)` |
| traced | `TracedLinalgOps` | `TracedTensor` | `t.svd()`、`t.svd_with(&opts)` |
| eager | `EagerLinalgOps` | `EagerSession<'_>` | `sess.svd(&x)`、`sess.svd_with(&x, &opts)` |

- `exec` は `&mut dyn Engine` で、最後の引数です。`_into` のときだけ、そのあとに `dest` が来ます。
- 演算の一覧は次のとおりです。21 演算で、どの層も同じ集合にします。

  > svd, svdvals, qr, householder_qr, rank_revealing_qr, lu, full_piv_lu, full_piv_lu_solve, solve, lstsq, triangular_solve, cholesky, eig, eigvals, eigh, eigvalsh, slogdet, det, inv, pinv, norm

- 現行は層ごとに集合が違います（concrete に `lstsq` がない、traced と eager に `svdvals` がない、typed に `householder_qr` がない）。実装できるものは実装します。すぐに実装できないものは、レジストリの `omissions` に理由つきで書きます。黙って抜けたままにしてはいけません。
- 戻り値のエラー型は次のとおりです。
  - concrete は `tenferro_tensor::Result`。
  - traced と eager は `tenferro_runtime::Result`。`tenferro_ad::Result` は同じ型の再エクスポートです。
  - `tenferro_linalg::Error`（`NonConvergence`、`NonFinite`、`Singular`、`UnsupportedDType`）は、外側エラーの `source()` から `downcast_ref` で取り出せるようにします（§5.5）。
  - `tenferro_linalg::Result` 別名は、linalg 固有の補助関数のために残します。

### 5.3 入力アダプタ `AsView`

```rust
pub trait AsView {
    type Repr: LinalgRepr;              // erased か typed<T> かを表す
    fn as_read(&self) -> TensorRead<'_>; // メタデータだけの変換。コピーしない
}
pub trait LinalgRepr {
    type Value; type Real; type Complex; type Index;
    fn wrap_value(t: Tensor) -> Result<Self::Value>;   // typed は検査つきダウンキャスト
    fn wrap_real(t: Tensor) -> Result<Self::Real>;
    /* ... */
}
impl AsView for Tensor { type Repr = Erased; /* ... */ }
impl AsView for TensorRead<'_> { type Repr = Erased; /* ... */ }
impl<T: LinalgScalar> AsView for TypedTensor<T> { type Repr = Typed<T>; /* ... */ }
impl<T: LinalgScalar> AsView for TypedTensorView<'_, T> { type Repr = Typed<T>; /* ... */ }
```

- 本体は erased の `TensorRead` の上に 1 つだけ書きます。typed は、結果を検査つきのダウンキャストで typed に戻します。現行の `TypedTensorLinalgExt` の実装（`tensor_ext.rs:2361`）が既にこの形なので、それを一般化します。プロトタイプ P1 で、これで1つの実装を共有できることを確かめています。
- 検査つきダウンキャストが失敗したら、内部エラーとして `Error::Internal` を返します。unsafe なキャストや `unwrap` を使ってはいけません。
- `LinalgOps` は `impl<X: AsView> LinalgOps for X` という blanket impl で与えます。利用者が `Out` の型を書かなくてよいことを、doctest で確かめます。

**[本書で決定]** 負ストライドの扱いをそろえます。現行の typed の借用入口（`svd_with_exec`）は負ストライドを拒否し、erased の経路は gather して受け付けます（P1）。新しい入口は erased と同じく受け付け、そのことをレジストリの `layout` に書きます。

### 5.4 Options の対応

| 演算 | Options | 中身（旧からの対応） |
|---|---|---|
| svd | `SvdOptions` | `shape: SvdShape::{Thin, Full}`（旧 `svd_full`）、`gauge`、`driver`、`derivative_eps` |
| svdvals | `SvdvalsOptions` | `driver` |
| qr | `QrOptions` | `gauge`（既存） |
| householder_qr | `HouseholderQrOptions` | 空 |
| rank_revealing_qr | `RankRevealingQrOptions` | 既存（`gauge`、`rtol`、`atol`） |
| lu / full_piv_lu | `LuOptions` / `FullPivLuOptions` | 空 |
| full_piv_lu_solve | `FullPivLuSolveOptions` | 空 |
| solve / lstsq | `SolveOptions` / `LstsqOptions` | 空 |
| triangular_solve | `TriangularSolveOptions` | §3.3 |
| cholesky | `CholeskyOptions` | `uplo`（現行は下三角固定なので、既定を `Lower` にし、`Upper` は未対応なら `Unsupported`） |
| eig / eigvals | `EigOptions` / `EigvalsOptions` | 空 |
| eigh / eigvalsh | `EighOptions` / `EigvalshOptions` | 既存（`gauge`、`driver`、`derivative_eps`） |
| slogdet / det / inv | `SlogdetOptions` / `DetOptions` / `InvOptions` | 空 |
| pinv | `PinvOptions` | `rtol: Option<f64>`（旧 `pinv_with_rtol`） |
| norm | `NormOptions` | §3.3 |

二点、注意があります（どちらもプロトタイプ P1 で分かったこと）。
- **`SvdShape::Full` と gauge の組み合わせ：** 今は実装がありません。`svd_with_options*` は thin のみで、`apply_svd_gauge` は `pub(crate)` です。full と既定以外の gauge の組み合わせは、実装するまで `Unsupported` を返し、そのことをレジストリの `omissions` に書きます。
- **`SvdDriver`：** 指定しても結果が bit 単位で変わりません。**[本書で決定]** フィールドは残します。ドキュメントには「provider が driver を選べる場合のヒントで、現行の CPU provider では効果がない」と書きます。効果が出るまで、テストで「受け付けること」だけを確かめます。

### 5.5 エラー原因の保存

プロトタイプ P5 で、次のことが分かっています。
- `tenferro_ad::Error` は `tenferro_runtime::Error` と同じ型です。外側のエラー型は2つです。
- transparent なラッパーは `source()` を素通しするので、`source()` をたどっても、包まれた `tenferro_tensor::Error` そのものが見つかりません。
- traced の einsum では、`kind()` と、たどれる原因が食い違うことがあります。

やることは次の4つです。
1. linalg の数値的な失敗は、`tenferro_tensor::Error::Extension { op, family, kind, source }` に、`source` が `tenferro_linalg::Error` になる形で入れます。`kind` は `source.kind()` から作ります。現在の作り方を確認し、直接埋めている箇所があれば `source` から導く形に直します。
2. runtime と eager の層で包むときに `source` を保ち、`err.source()` の連鎖から `tenferro_linalg::Error` へ `downcast_ref` できることを、3 層それぞれでテストします。P5 の `p5_error_causes.rs` をテストの見本にしてください。
3. `#[error(transparent)]` の variant は、`source()` で包んだエラーそのものを返さず、その原因を返します。`tenferro_tensor::Error` そのものをたどって見つけたい場合のために、**[本書で決定]** runtime 側の `TensorRuntime(#[from] tenferro_tensor::Error)` から `#[error(transparent)]` を外し、`#[error("{0}")] TensorRuntime(#[from] tenferro_tensor::Error)` にします（`#[from]` が `#[source]` を兼ねる）。表示される文字列は変わりませんが、`source()` が包んだエラー自身を返すようになります。
4. 新しい公開エラートレイト（`TensorError` など）は追加しません（v0.2）。

### 5.6 `*_with_exec`（E08）

`svd_with_exec`、`svd_values_with_exec`、`qr_with_exec`、`eigh_with_exec`、`eigh_values_with_exec`、`eig_values_with_exec`（`with_exec.rs`）は、**今回は名前も形も変えません**。

v0.1 と v0.2 で、削除の条件は次のように決まっています。`Exec` と `BufferPool` を構築時に受け取る借用エンジンの試作で、次の4点を確かめること。
- ライフタイム
- バッファの再利用
- 入れ子の実行
- 非同期に保持される場合

この試作はまだありません。これらの関数は E08 として例外表に載せ、移行対象から外します。`full: bool` もそのままです。

試作は別の Issue として起こします。その試作がこの作業の途中で終わった場合も、この作業ブランチには入れず、次のリリースで扱います。

### 5.7 `EagerTensor::solve`

- `EagerTensorLinalgExt::solve(&self, engine, b)`（`eager_ext.rs:75`）は削除し、`sess.solve(&x, &b)` に一本化します（決定 3）。
- 現行の `solve` が保っている「呼び出しスレッドの `no_grad` を尊重する」振る舞いは、`EagerSession` の経路で同じになることをテストで確かめてから削除します。同じにならない場合は、削除せずに #2552 で報告してください。

---

## 6. S4：einsum

### 6.1 前提

#2565（N 入力の einsum が共役の指定を黙って捨てる）の修正が先に main に入っている必要があります。最低限、未対応の共役指定を `Unsupported` として拒否するようになっていることが条件です。

### 6.2 Spec と Options

```rust
/// Text, parsed notation or subscripts; parsing and validation happen
/// inside the operation, so construction never fails.
pub enum Spec<'a> { Text(&'a str), Notation(&'a EinsumNotation), Subscripts(&'a EinsumSubscripts) }
impl<'a> From<&'a str> for Spec<'a> { .. }
impl<'a> From<&'a EinsumNotation> for Spec<'a> { .. }
impl<'a> From<&'a EinsumSubscripts> for Spec<'a> { .. }

#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct EinsumOptions {
    pub optimize: EinsumOptimize,
    pub conj: ConjMask,        // オペランドごとの Conj。既定はすべて No
}
```

- `From` は表現を包むだけで、検証はしません。文字列の解析エラーは、演算の中で `InvalidSubscripts` として返します（v0.1 の合意）。
- `EinsumOptimize` と `ContractionTree` に `Clone` を derive します。現在は `EinsumOptimize` が `Debug` だけで、`ContractionTree` は derive がありません（`optimize.rs:54`、`planning/tree.rs:110`）。`&opts` でツリーを渡すために必要です（P6）。
- `ConjMask` は、2 入力の場合、`DotGeneralAccumulation` の `lhs_conj` と `rhs_conj` に写します。3 入力以上で共役が指定され、その経路が未対応の場合は `Unsupported` を返します（#2565 の方針に合わせる）。
- **`EinsumOptimize` のうち `Auto` 以外を concrete のプランが守れない問題（P2）について：** `ConcreteEinsumPlan` は常に `ContractionTree::optimize` を実行しています。**[本書で決定]** `Tree`、`Path`、`Nested` を指定された場合は、その木を使うように直します。直せない組み合わせは `Unsupported` を返します。黙って無視してはいけません。

### 6.3 concrete

| 旧 | 新 |
|---|---|
| `TensorEinsumExt`、`TypedTensorEinsumExt`、`TensorReadEinsumExt`、`TypedTensorReadEinsumExt`、`*EinsumIntoExt`（計 8 トレイト、それぞれ bare / `_notation` / `_subscripts`） | `EinsumOps`。`[&a, &b].einsum(spec, exec)`、`.einsum_with(spec, &opts, exec)`、`.einsum_into(spec, &opts, exec, dest)` |
| `TensorTensordotExt`、`TypedTensorTensordotExt` | `TensordotOps`。`a.tensordot(&b, axes, exec)`、`a.tensordot_with(&b, axes, &opts, exec)` |
| `ConcreteEinsumPlan::prepare{,_subscripts,_notation,_typed*,_read*}`（9 本） | `ConcreteEinsumPlan::prepare(inputs, spec, &opts)`（1 本） |
| `execute`、`execute_typed`、`execute_read`、`execute_into`、`execute_typed_into`、`execute_read_into`、`execute_read_into_accum`（7 本） | `plan.execute(inputs, exec)`、`plan.execute_into(inputs, exec, dest)` |

- `EinsumOps` の受け手は、`[X]` と `[X; N]` です（`X: AsView`）。所有テンソルと借用ビューを混ぜて渡したい場合は、呼び出し側で `[a.as_read(), b]` のように借用ビューのスライスにそろえます。データのコピーは起きません（v0.1）。
- `prepare` は入力のメタデータ（dtype と形状）だけを読みます。`inputs` には、`AsView` を実装する型のスライスを受け付けます。
- `execute` の出力は、入力の `Repr` で決まります。typed の入力なら typed で返します。
- `clear_cpu_cache` と `cpu_cache_stats` は E03 なので、そのまま残します。
- プランのキャッシュキーにはレイアウトが入りません（P4）。レイアウトが変わると、キャッシュが一度ミスして入れ替わります。結果は正しく、ポリシーは撤回済みなので、この挙動のままでかまいません。

### 6.4 traced と eager

- traced：`trace.einsum(&[&t1, &t2], spec)` と `trace.einsum_with(&[..], spec, &opts)` を、`TracedEinsumOps for TraceContext` として提供します。受け手は `&self` で、戻り値は `TracedTensor` です（決定 4、案 C）。
  - 現行の `TraceContextEinsumExt` は `&mut self` を取り、`TraceValue` を返す6メソッドです。名前が衝突するので、**この trait は削除して置き換えます**（E07 を退役）。
  - P6 の計測では、TracedTensor 経路の構築時間は TraceValue 経路と誤差の範囲で、割り当て回数は 12〜20% 多いという結果でした。この差は受け入れます。
- `TracedTensor` から `TraceValue` を得る公開の変換はありません。`derive_tensor` は doc(hidden) です。新しい入口は `TracedTensor` を受けるので、この変換は不要です。
- `TracedTensorEinsumExt::tensordot` は、`TracedEinsumOps for TracedTensor` の `t.tensordot(&u, axes)` に移します。
- eager：`EagerSessionEinsumExt` を `EagerEinsumOps for EagerSession` にし、`sess.einsum(&[&x, &y], spec)`、`sess.einsum_with(..)`、`sess.tensordot(&x, &y, axes)` を提供します。
- `TraceContext::input`、`finish`、`add_op`、`add_extension` と `TraceValue` は、E04 の IR ビルダー層としてそのまま残します。
- **`finish` と `compile_many` の不一致：** `finish` から `compile_traced_graph` を経ると `input_keys` が空になり、`compile_many` では埋まります（P6）。**[本書で決定]** `finish` の経路でも `input_keys` を埋め、両者が同じ入力順と枝刈りになることをテストで固定します。
- **下流への影響：** yao-rs は `=0.4.0` を固定しているので、すぐには壊れません。ただし移行先（`Runtime::prepare` → `Runtime::run`）を、`docs/getting-started/api-migration.md` に書きます。

### 6.5 エラー

einsum は、現在どの層でも `tenferro_einsum::Error` を返しています。ルール 5 に合わせて、次のように変えます。
- concrete は `tenferro_tensor::Error` を返します。`Error::Extension { family: EINSUM_EXTENSION_FAMILY_ID, kind, source: tenferro_einsum::Error }` に包みます。
- traced と eager は `tenferro_runtime::Error` を返します。
- `tenferro_einsum::Error` と `PlanningError` は、領域のエラーとして残します（E09）。パーサー（`parse_einsum_subscripts` など）とプランナーは、引き続きこの型を返します。
- 同じ不正な spec に対する分類を、3 層でそろえます。P5 では、concrete と eager は `InvalidSubscripts`、traced は `InvalidArgument` でした。**[本書で決定]** 3 層とも `kind() == Validation(InvalidArgument)` にし、`source()` から `tenferro_einsum::Error::InvalidSubscripts` に届くようにします。

---

## 7. S5・S6：FFT と core

### 7.1 FFT

| 層 | 旧 | 新 |
|---|---|---|
| concrete | `TensorFftExt`（1 次元のみ）、`TensorReadFftExt` | `FftOps for X: AsView`：`fft`、`ifft`、`rfft`、`irfft`、`fftn`、`ifftn`、`rfftn`、`irfftn` と、それぞれの `_with` |
| traced | `TracedTensorFftExt`、`TraceContextFftExt` | `TracedFftOps for TracedTensor`。`TraceContextFftExt` は削除（決定 4） |
| eager | `EagerSessionFftExt`（1 次元のみ）、`EagerTensorFftExt` | `EagerFftOps for EagerSession`：8 演算と `_with`。さらに `fft_in_place(x, &opts)` と `ifft_in_place(x, &opts)`（E01） |
| キャッシュ所有者 | `FftExecutor::fft(&mut self, input, n, axis, norm, session)` | `FftExecutor::fft_with(&mut self, &input, &opts, exec)`。`new`、`plan_cache`、`plan_cache_mut`、`cache_stats`、`clear_cache` は残す（E02） |

- Options は次のとおりです。
  - `FftOptions { n: Option<usize>, axis: isize, norm: FftNorm }`（`fft`、`ifft`、`rfft`、`irfft` で共通）
  - `FftnOptions { shape: Option<Vec<usize>>, axes: Vec<isize>, norm: FftNorm }`
  - 既定値は現行の既定値と同じにします。`axis` は `-1` です。
- 現在、concrete と eager にない N 次元の FFT（`fftn` など）を足します。実装は `FftExecutor` の N 次元経路を呼べば済みます。
- **in-place の FFT：** 長さを変えられない演算なので、`FftOptions` の `n` を指定したら `Error::Validation(InvalidArgument)` を返します。**[本書で決定]** in-place 専用の別の Options 型は作りません。
- **`ifft_in_place` の検証エラーのラベル：** 現在は `"fft_in_place"` になっているので、`"ifft_in_place"` に直します（P3）。
- **`EagerFftInPlaceError`：** `?` を使うと、取り戻せるはずの所有者が捨てられます（P3）。**[本書で決定]** 型は変えません。rustdoc の例で「`?` の前に `match` して、`Rejected { input, .. }` から所有者を取り戻す」書き方を示し、そのことをテストします。

### 7.2 core（runtime と ad）

| 層 | 旧 | 新 |
|---|---|---|
| concrete | `TensorSessionOpsExt`、`TypedTensorSessionOpsExt`、`TypedTensorMaskSessionOpsExt` | `CoreOps for X: AsView`（`where_select` は受け手が `TypedTensor<bool>` のときだけ使える）。`concatenate` は `CoreOps::concatenate(&[..], axis, exec)` の関連関数 |
| traced | `TracedTensor` の固有メソッド | 固有メソッドのまま。名前と引数を文法に合わせる。`_with` は Options を持つ演算にだけ生成する |
| eager | `EagerSession` の固有メソッド | 固有メソッドのまま。同上 |

- concrete の `exec` は最後の引数です。現在のコードはすでにそうなっています。REPOSITORY_RULES.md の「engine は第1引数」という記述のほうが間違っているので、§8 で直します。
- `dot_general_with_conj` は削除し、`dot_general_with(&rhs, config, &DotGeneralOptions { lhs: Conj, rhs: Conj }, exec)` にします。`add_into`（typed のみ）は、`add_into(&rhs, &AddOptions::default(), exec, Dest::overwrite(&mut out))` にします。それ以外の `_into` は、今回は増やしません（決定 5：レジストリで演算ごとに宣言する）。
- 軸の引数は `isize` にします（§3.2）。
- 型変換（`convert`、`cast`）は明示的な演算のまま残します。データの移動が必要な `reshape` を、メタデータだけの操作として扱ってはいけません。
- 層ごとの演算の集合の差（たとえば `EagerSession` に `softmax` などがない）は、この作業では埋めません。レジストリの `layers` と `omissions` に、現状をそのまま書きます。

---

## 8. S7：規約・仕様・ドキュメント・スキル

文法の変更と同じ作業ブランチ（同じ PR）で、次をすべて書き換えます。どれかが古いままだと、`repository-rules-review.py` と規約に沿ったレビューが、新しい文法を違反と判定します。

| ファイル | 書き換える内容 |
|---|---|
| `REPOSITORY_RULES.md`「Public API Convention」（903〜964 行） | 次の4つを書き換える：(1) 文法の3形と層ごとの形、(2) 接尾辞は `_with` と `_into` だけで、累積は `Dest`、(3) トレイトの一覧を新しい 12 個に、(4) 例外は例外表を参照する。「Output And Write Surface Vocabulary」（`_read`、`_read_into`、`_add_to`、`_into_accum`）は、`_in_place`（E01）を残して削除し、生成された `docs/spec/generated/api-ops.md` を参照させる |
| `REPOSITORY_RULES.md`「Public Surface Discipline」 | `_read` と `_view` の命名について書いた箇条を、新しい文法に合わせる |
| `REPOSITORY_RULES.md`「Wrapper DRY And Codegen」 | レジストリによる生成を許す条件として、§4.2 の「生成するもの・しないもの」を書く |
| `PERFORMANCE_TIPS.md`「Materialization And Copies」 | 新しい R3 と照らし合わせる。「`*_read` フック」は、公開名ではなくバックエンドのフックを指す言葉として残る。食い違いがあれば直す |
| `scripts/repository-rules-review.py` | `##` 節を足したり名前を変えたりした場合は、`SECTION_TRIGGERS`、`ALWAYS_SECTIONS`、`HUMAN_ONLY_SECTIONS` を更新する（AGENTS.md の指示） |
| `docs/spec/api-conventions.md` | 命名規則 3（`_read`）と形状規則 4（einsum の 8 トレイト）を削除し、生成文書を参照させる。存在しない「Naming rule 10」への参照（122 行目）を直す |
| `scripts/check-api-consistency.py` | CI で実行されていない。`_read` の検査（`read_suffix_without_read_input`）は新しい文法では意味がないので削除する。残す検査（`try_` 接頭辞、`traced_` 接頭辞など）は、新しい文法と矛盾しないか確かめる。**[本書で決定]** スクリプトとテストは残し、CI への組み込みは公開 API スナップショット（S0）に任せる |
| `docs/design/api-consolidation-policy.md` | R2（`_into` を公開の基本形とする）を、「`_into` はレジストリで演算ごとに宣言する」に改める |
| tensor4all-agent-rules | 「no hidden … dense-layout conversions」の規則を、新しい R3 と合わせる。別のリポジトリなので、そちらに PR を出し、この PR の説明からリンクする |
| `.agents/skills/tenferro-compute/` と `.claude/skills/tenferro-compute/` | `SKILL.md`、`references/api-cheatsheet.md`、`references/performance-idioms.md`。2 つは別々の実ディレクトリなので、両方を同じ内容で直す。直したら `scripts/check-agent-skills.py` を実行する |
| `docs/guides/*.md`、`docs/getting-started/*.md`、`docs/tutorials/*.md`、`docs/tutorial-code/src/bin/*.rs`、`README.md` | 旧トレイト名と旧メソッド名をすべて置き換える。`docs/tutorial-code` はコンパイルされるので、`cargo build` が通ることを確かめる |
| `docs/getting-started/api-migration.md` | 旧名 → 新名の対応表を載せる（§11 の表を元にする） |
| `docs/internals/public-boundary-overhead-inventory.{json,md}`、`docs/spec/operation-categories.md` | `check-public-boundary-inventory.py` と `check-operation-categories.py` が読むので、名前と面を新しいものにする |

旧名が残っていないことは、次のコマンドが何も出力しないことで確かめます。

```bash
rg -n 'TensorLinalgExt|TensorReadLinalgExt|TypedTensorLinalgExt|TraceContextLinalgExt|TraceContextFftExt|TraceContextEinsumExt|TensorEinsumIntoExt|TensorReadEinsum|TypedTensorReadEinsum|TensorFftExt|TensorReadFftExt|EagerTensorLinalgExt|EagerTensorFftExt|TensorSessionOpsExt|_into_accum\b|svd_full|_with_rtol|prepare_typed|execute_read' \
  --glob '!docs/plans/**' --glob '!docs/design/api-2552-convergence/**' --glob '!docs/worklogs/**' --glob '!api/**'
```

`docs/plans/` と `docs/worklogs/` は履歴なので、書き換えません（AGENTS.md）。

---

## 9. 文法の移行とは別に進める修正

それぞれ main から切った別ブランチで、通常の PR にします。

| 修正 | 内容 | 完了条件 |
|---|---|---|
| #2565 | N 入力の einsum で共役の指定を捨てない。最低限、未対応なら拒否する | 2 入力と 3 入力、および `ii,ij->j` で、共役を指定したテストが、正しい値を返すか `Unsupported` になる |
| eager einsum の毎回のコピー | フォールバックが、コンパクトな入力も含めてすべてのオペランドを `to_contiguous_read` でコピーしている（P4） | コンパクトな入力でコピーが起きないことを、`to_contiguous_read` のスパイ（またはカウンタ）で確かめるテスト。わざと元の実装に戻すと失敗する対照テストも置く（PERFORMANCE_TIPS.md の要求） |
| rank 3 の strided SVD バッチ | `svd_with_exec` のカーネルはそのまま読めるのに、ディスパッチの段階で gather している（P4） | rank 3 の非コンパクトなバッチで gather が起きず、結果が変わらないことのテスト |
| CUDA linalg の転置 gather | 転置したビューを gather している | GPU でしか確かめられないので、`local_gpu_gate.py` で検証する（REPOSITORY_RULES.md の CI Cost Discipline） |

公開のカウンタは作りません。計測は、テストの中のスパイと、ベンチマークで行います。割り当て回数とコピー回数を同じものとして扱ってはいけません（決定更新 6764423689）。

---

## 10. 完了の定義と検証

### 10.1 演算ごとの完了条件

1 つの演算は、次をすべて満たしたときに完了です。

- [ ] レジストリにエントリがある。`layout`、`errors`、`omissions` が埋まっている。
- [ ] 3 層（または宣言した層）で `op` / `op_with`（/ `op_into`）が生成されている。
- [ ] §4.4 の 4 種類のテストが通る。
- [ ] 数値の結果が、旧 API と比べて変わっていない。旧 API を消す前に、旧 API の結果と比べるテストを一時的に置いて確かめ、比べ終わったら旧 API と一緒に消す。
- [ ] 旧名が、コード、ドキュメント、スキルから消えている（§8 の `rg`）。
- [ ] rustdoc の例があり、doctest として実行されている。

### 10.2 PR を出す前

AGENTS.md の「Pre-Push / PR Checklist」に従います。最低限、次を実行します。

```bash
bash scripts/check-pr-fast.sh --coverage-reviewed --test 'cargo test --workspace'
python3 scripts/repository-rules-review.py --base origin/main --head HEAD \
  --output-json /tmp/repository-rules-review.json \
  --dry-run --llm-skipped-reason "local deterministic review"
bash scripts/api-snapshot.sh --check
python3 scripts/check-agent-skills.py
cargo doc --workspace --no-deps
```

- `api/` のスナップショットは `--write` で更新してコミットします。PR の説明では、その差分を「消えたもの」と「増えたもの」に分けて要約します。この差分が、そのまま退役リストになります。
- PR 本文には `## Verification` 節が必要です（`body` チェック）。実行したコマンドとその出力を貼ります。
- GPU を触る変更を含む場合は、`gpu-validated-locally` ラベルを付け、`local_gpu_gate.py` で検証します。

### 10.3 受け入れ条件（#2552 の決定更新から）

- 同じ演算なら、どの入力の形（adapter）を使っても、レイアウトの扱いが同じであること。
- provider が直接読める strided の経路が残っていること。
- レイアウトの制約と変換が、演算ごとに文書化されていること（レジストリの `layout` から生成）。
- 既知の避けられるコピーに、回帰テストか性能テストがあること（§9）。
- 数値、所有権、エイリアス、出力先の更新についての保証が変わっていないこと。

---

## 11. 旧名 → 新名の対応（抜粋）

| 旧 | 新 |
|---|---|
| `a.svd(eng)` → `(u, s, vh)` | `let Svd { u, s, vh } = a.svd(eng)?` |
| `a.svd_full(eng)` | `a.svd_with(&SvdOptions::default().shape(SvdShape::Full), eng)` |
| `r.svd_read(eng)`（`r: TensorRead`） | `r.svd(eng)` |
| `a.svd_with_options(opts, eng)` | `a.svd_with(&opts, eng)` |
| `a.pinv_with_rtol(rtol, eng)` | `a.pinv_with(&PinvOptions::default().rtol(Some(rtol)), eng)` |
| `a.triangular_solve(&b, true, true, false, false, eng)` | `a.triangular_solve_with(&b, &TriangularSolveOptions::default(), eng)` |
| `r.solve_read_into(b, out, eng)` | `r.solve_into(&b, &SolveOptions::default(), eng, Dest::overwrite(out))` |
| `a.norm(None, Some(&[0]), false, eng)` | `a.norm_with(&NormOptions::default().axes(Some([0].into())), eng)` |
| `x.solve(engine, &b)`（`EagerTensor`） | `sess.solve(&x, &b)` |
| `ctx.svd(a_value)`（`TraceContextLinalgExt`） | `t.svd()`（`TracedTensor`） |
| `[&a, &b].einsum_notation(&n, eng)` | `[&a, &b].einsum(&n, eng)` |
| `[r1, r2].einsum_read_into("ij,jk->ik", eng, out)` | `[r1, r2].einsum_into("ij,jk->ik", &EinsumOptions::default(), eng, Dest::overwrite(out))` |
| `plan.execute_read_into_accum(inputs, eng, acc, out)` | `plan.execute_into(inputs, eng, Dest::scaled(out, alpha, beta))`。共役はプランを作るときの `EinsumOptions::conj` で指定する |
| `ConcreteEinsumPlan::prepare_typed_notation(inputs, &n)` | `ConcreteEinsumPlan::prepare(inputs, &n, &EinsumOptions::default())` |
| `ctx.einsum_subscripts_with(&[v1, v2], s, opt)`（`&mut`、`TraceValue`） | `ctx.einsum_with(&[&t1, &t2], s, &EinsumOptions { optimize: opt, ..Default::default() })` |
| `a.fft(None, -1, FftNorm::Backward, eng)` | `a.fft(eng)`、または `a.fft_with(&FftOptions::default().norm(..), eng)` |
| `x.fft_in_place(engine, axis, norm)` | `sess.fft_in_place(x, &FftOptions::default().axis(axis).norm(norm))` |
| `exec.fft(&input, n, axis, norm, eng)`（`FftExecutor`） | `exec.fft_with(&input, &opts, eng)` |
| `a.dot_general_with_conj(&b, cfg, true, false, eng)` | `a.dot_general_with(&b, cfg, &DotGeneralOptions::default().lhs(Conj::Yes), eng)` |
| `a.reduce_sum(Some(&[0usize]), eng)` | `a.reduce_sum(Some(&[0isize]), eng)`（`-1` も使える） |

---

## 12. 迷ったとき

- この文書と #2552 の決定が食い違う場合は、#2552 を優先し、この文書を直す PR を出してください。
- この文書に書いていない判断が必要になった場合は、次の2つのどちらかにします。
  - 推測で実装せずに #2552 で質問する。
  - レジストリの `omissions` か例外表に「未決」として記録し、`Unsupported` を返す形で先に進む。
- 数値の結果を変えてしまう変更は、どんな理由があっても、この移行の中で行ってはいけません。必要なら別の Issue を立ててください。
