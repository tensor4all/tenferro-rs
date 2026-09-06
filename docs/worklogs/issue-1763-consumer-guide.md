# #1763: ordinary calls and prepared-execution consumer guidance

## Outcome and scope

Phase 3 of #1771 adds one guide, its README/sidebar/llms.txt routes, and updates
the bundled consumer skill and both mirrors. It distinguishes ordinary calls,
compatible `ConcreteEinsumPlan` reuse, and programmatic labels without changing
library behavior or adding API aliases/caches. The existing tutorial binary and
snippet synchronization mechanism own the executable recipe; no generator,
benchmark framework or new testing API is introduced.

#1771 supersedes #1758's broader infrastructure and consolidation requirements.
Phase 2's explicit production-change deferral is respected. Its evidence is
linked at immutable commit `552b4793`, independently of the evidence PR's merge
timing. The guide reports historical observations with timing/worker/host scope,
not a conflicting “latest results” table or a default/GPU performance promise.

## Sources and decisions

Read #1771/#1763, Phase 2's record, the repository/shared docs, consumer,
performance and numerical rules, existing einsum guide and consumer references,
`ConcreteEinsumPlan` preparation/execution, `EinsumSubscripts`/`EinsumNotation`,
and the existing snippet/mirror/site checkers. Worktree started at current main
`7dfc01127f4a8752a8bb504641feb396683576c3`; repository rules match the Phase 2
checkout already read in this session.

- The recipe performs ordinary string execution, prepares once, executes with
  two different value sets under unchanged metadata, and constructs an integer
  equation. It asserts every output element, not merely shape or finiteness.
- Reuse constraints reflect stored input specs and execution validation, not a
  claim that plans own inputs, sessions, device placement or results.
- Flat notation rejects parentheses; the guide explicitly prevents flattening
  grouping into integer labels and links the supported traced path controls.
- Existing eager, typed/read/output and traced examples remain canonical; the
  new page routes to them rather than copying every API family.
- No independent agent review was requested. Direct executable consumer checks
  and rendered-page inspection were used; do not call this an independent
  source-blind audit.

## Verification and corrections

- `scripts/check-doc-snippets.py --check` and `scripts/check-agent-skills.py`:
  passed; the new guide and skill share one extracted Rust source block.
- `bash scripts/check-pr-fast.sh --coverage-reviewed --test 'cargo run --manifest-path docs/tutorial-code/Cargo.toml --bin tenferro_compute_skill'`:
  passed formatting, CI-parity clippy and all recipe numerical/AD assertions.
  `CARGO_TARGET_DIR` reused the idle Phase 2 target serially; no concurrent Cargo
  jobs used that target.
- `scripts/check-guide-dependency-snippets.py`: passed all four existing guides.
- `cargo doc --workspace --no-deps`: passed.
- `quarto render docs`: rendered all 117 pages. Inspected the new page's HTML
  headings, extracted recipe, numerical assertions, units and provenance links.
  Only Quarto's existing output-directory cleanup/configuration warnings remain;
  no unresolved link warning remains for the new guide.
- `scripts/check-docs-site.py --doc-root <Phase-2-target>/doc`: passed after full
  rendering and the canonical skill-reference copy step from
  `build_docs_site.sh`; verifies site links, rustdoc inventory and llms.txt.
- Initial partial site checks were not passes: a 60-second dependency-check
  attempt timed out (no owned descendants remained), and a single-page render
  lacked the site's republished references. The complete checks above supersede
  those attempts. A virtual skill-reference link caused a render warning and
  was replaced with the canonical traced guide link; the corrected page was
  rerendered. HTML syntax-highlight spans required token-aware text inspection,
  rather than an exact raw multi-number text match.
- Diff self-review: no production code, dependency, feature, public API, AD rule,
  transfer or ownership changes. Existing examples/README behavior is preserved.

Required hosted CI and merge are checked on the submitted PR separately; local
checks alone do not prove completion. This log records artifact verification,
not new durable architecture or a mandate to optimize further routes.
