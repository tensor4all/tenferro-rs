# README / Docs / Blog Audience Redesign — Design

- Date: 2026-07-04
- Status: approved in discussion; pending maintainer spec review
- Scope: `README.md`, `docs/index.md`, `docs/getting-started/` (3 pages),
  the "Introducing tenferro-rs" blog post (separate website repository),
  and the GitHub repository description.

## Goal

Make each documentation surface serve a clearly chosen audience with one
consistent project identity, so that a first-time visitor can answer
"what is this, is it for me, how is it different, how do I start" within
the first screen of the README.

## Decisions

### 1. Primary audience

**Rust developers and researchers doing numerical / scientific
computing.** They know NumPy/JAX/PyTorch concepts, are not tensor-network
specialists, and arrive cold from crates.io, web search, or GitHub.
Optimizing the front door for this group also serves the secondary
audiences in deeper sections.

Secondary audiences and their treatment:

| Audience | Treatment |
| --- | --- |
| JAX/PyTorch migrants | Served by `getting-started/pytorch-jax-mapping.md`, framed as a translation guide, not as the project identity |
| Library authors building on tenferro | Served by extension/custom-operations content and design docs, linked from README |
| tensor4all researchers | Reached mostly through tensor4all-rs / Tensor4all.jl; README provides ecosystem context |
| Contributors and AI coding agents | Served by CONTRIBUTING, REPOSITORY_RULES, and docs internals; moved off the README front door |
| Evaluators (papers, talks, reviewers) | Served by compressed Project section (stability, engineering discipline) plus the blog narrative |

### 2. Positioning (unified one-liner)

> A Rust-native tensor & autodiff stack for scientific computing.

- PyTorch/JAX appear as **explanatory anchors** ("eager like PyTorch,
  traced like JAX"), never as the identity.
- Explicitly not a deep-learning framework; the README says so and points
  to candle/burn for that use case, and to JAX/PyTorch when the host
  language is Python.
- Differentiators to lead with: column-major / LAPACK-Fortran-Julia
  alignment, dynamic shapes in compiled graphs, extensible operations and
  AD rules, explicit backend/device control.

### 3. Surface roles

| | README | docs site | blog post |
| --- | --- | --- | --- |
| Role | Front door: 30-second fit decision | Workspace: task completion | Narrative: why the project exists |
| Main reader | Cold-arriving Rust numerical-computing user | Someone who decided to try tenferro | Community, physics audience, social/aggregator readers, evaluators |
| Temporality | Evergreen, always matches current state | Evergreen, synced with code (CI-verified examples) | Dated snapshot; may describe its moment |
| Voice | Project voice, facts only | Neutral, task-oriented | Author's first person; opinions and story welcome |
| Update policy | Continuous edits | Continuous edits | Append-oriented: major changes become new posts |

The one-line identity is the only text shared verbatim across surfaces:
README opening, docs landing opening, blog opening, and the GitHub
repository description. Everything else varies by surface.

### 4. Story moves to the blog

The blog post becomes the canonical home of the project narrative
(Julia-to-Rust background, AI-era development philosophy, verification
philosophy, cross-ecosystem collaboration stance). The README compresses
each of these to one paragraph plus a link to the blog post. Content that
functions as durable policy (how changes are validated, contribution
rules) stays in CONTRIBUTING / REPOSITORY_RULES / docs, not in the blog.

## README redesign (329 lines → target ≈170)

New section order, following the reader's decision sequence:

| # | Section | Content |
| --- | --- | --- |
| 1 | Title + one-liner + badges | Unified one-liner, 2–3 sentence capability summary, crates.io/docs/CI/license badges (new) |
| 2 | Quick Example | Current `cpu_quickstart` snippet moved up (~line 30); snippet-source markers preserved |
| 3 | Is tenferro for you? | Rework of "When tenferro Is a Good Fit" + "Why Build On tenferro-rs": dynamic shapes, Rust-native AD, column-major alignment; one-line respectful positioning vs ndarray, candle/burn, faer; "if Python, use JAX/PyTorch" |
| 4 | Which API Should I Use? | Current table kept as-is |
| 5 | Crates | Core + extension tables kept; implementation crates compressed to one note line |
| 6 | Documentation | Docs-site links, Getting Started first |
| 7 | Project | One paragraph each: why it exists (+blog link), stability policy, engineering discipline (oracles, coverage, benchmarks), AI-assisted development (+blog link) |
| 8 | Community | Matrix, issues, mailing list (kept) |
| 9 | Acknowledgments | Kept, tightened by ~20–30% |
| 10 | Contributing | Key points + CONTRIBUTING/GOVERNANCE links |

Disposition of current sections: "Design Principles" bullets are absorbed
into §3 and §7; "Why Build On tenferro-rs" merges into §3; "Benchmarks And
Numerical Validation" merges into §7. No information is deleted outright;
everything is compressed, relocated, or delegated to the blog.

## docs/index.md

Rewrite the opening paragraph only: unified one-liner + capability
summary, with PyTorch/JAX demoted to explanatory anchors. Keep the
"Where To Start" table, "First CPU Example", "Mental Model", and
"Get In Touch" sections unchanged.

## getting-started/

Alignment edits only; no structural or content reorganization.

- `index.md`, `core-concepts.md`: align opening framing with the
  one-liner.
- `pytorch-jax-mapping.md`: add an intro positioning it as a translation
  guide for PyTorch/JAX users.

## Blog post (separate repository)

Source: `tensor4all/tensor4all.github.io` (Jekyll site), checked out at
`~/tensor4all/tensor4all.github.io`. The post exists in three language
versions — `blog/introducing-tenferro-rs`, `…-rs-ja`, `…-rs-zh` — and
every change below must be applied to all three in sync. Blog changes
ship as a separate PR in that repository.

1. Align the opening italic line with the unified one-liner (currently a
   feature list). The H1 ("… a differentiable tensor stack for
   scientific computing …") is already close and stays.
2. Fix any drift against the current project state (post published
   2026-06-23; expected minor).
3. Absorb narrative elements compressed out of the README that the post
   does not already cover (e.g., the cross-ecosystem collaboration
   stance). The post already covers the Julia-to-Rust background,
   AI-development model, and verification philosophy, so no major
   additions are expected.
4. Confirm/strengthen the closing pointer to docs Getting Started.

## Success criteria

1. The first README screen answers "what / for whom / how is it
   different".
2. A runnable example appears near line 30 of the README.
3. The one-liner matches across README, docs landing, blog opening, and
   the GitHub repository description.
4. snippet-source sync and CI doc checks (`check-docs-site.py`, doc
   snippet tests) keep passing; `docs/_quarto.yml` is unchanged.

## Constraints

- README example stays snippet-source-synced with
  `crates/tenferro-runtime/examples/cpu_quickstart.rs`.
- tenferro-rs changes go through the normal PR flow (auto-merge, squash);
  the implementation PR links a work log under `docs/worklogs/`.
- References to other projects (ndarray, candle, burn, faer) are factual
  and respectful.
- crates.io per-crate `description` fields are out of scope for this
  round.

## Rejected alternatives

- **"JAX/PyTorch in Rust" positioning** — invites head-on comparison with
  candle/burn and sets deep-learning expectations (NN layers, optimizers)
  the project deliberately does not serve.
- **Dynamic-shape-first positioning** — strongest differentiator but too
  abstract as a cold opening; readers must already understand traced
  execution.
- **Full docs-site restructure** — the existing Getting
  Started/Tutorials/Guides/API/Internals structure is sound; cost and
  risk outweigh the entry-page misalignment actually observed.
- **Keeping full-length philosophy sections in the README** — blocks the
  30-second fit decision for the primary audience; the narrative now has
  a canonical home in the blog.
- **New "Project" section on the docs site for the philosophy** — adds a
  third surface to keep in sync; the blog post already exists, is fresh,
  and fits the narrative role.
