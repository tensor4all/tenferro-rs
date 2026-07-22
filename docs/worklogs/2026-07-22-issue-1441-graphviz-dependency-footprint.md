# Issue 1441 Graphviz Dependency Footprint

## Summary

Issue #1441 reports that dependency arrows in
`docs/assets/dependency-footprint.svg` terminate in whitespace rather than at
their intended crate boxes. The SVG encoded every box and edge as hand-written
coordinates even though the repository already had a Cargo-to-DOT generator.
This change makes that generator the source of truth, delegates layout and edge
attachment to Graphviz, and checks the generated SVG's semantic inventory.

## Context Read

- Issue #1441 and its related issue #1318 and PR #1327
- `AGENTS.md`, `CONTRIBUTING.md`, and `REPOSITORY_RULES.md`
- shared tensor4all repository and documentation/test rules
- `ai/contribution-workflows/bugfix-pr.md`
- `ai/contribution-workflows/repository-remediation.md`
- `scripts/gen_dep_graph.py`
- `scripts/build_docs_site.sh`
- `scripts/check-docs-site.py` and `scripts/test-check-docs-site.py`
- `scripts/ci/run_profile.py` and its tests
- `README.md`, `docs/index.md`, and the original dependency-footprint SVG

## Reference And Root Cause

`scripts/build_docs_site.sh` already rendered the API dependency graph by
piping `scripts/gen_dep_graph.py` into `dot -Tsvg`. The separate checked-in
footprint SVG was only described as being “seeded” by that generator and was
then laid out manually. The incorrect endpoints were therefore data in the
hand-written SVG, not a Graphviz routing defect.

The generator's old `extern/`, `extension/`, and root-prefix classification
also predated the current `crates/`, `docs/tutorial-code`, and standard
operation crate layout. Using its output unchanged would have put every current
workspace member into one `core` cluster.

## Decisions

- Keep Cargo manifests and the existing transitive reduction as the dependency
  source of truth.
- Classify every current workspace member into the five conceptual layers used
  by the public diagram: foundation, tensor/backends, runtime/AD, standard
  operation extensions, and runnable documentation examples. An unclassified
  future crate falls into a visible fallback cluster, while the regression test
  requires all current crates to be classified.
- Remove invisible ordering edges. Graphviz receives only real dependency edges
  and owns node placement, orthogonal routing, and arrow attachment.
- Add `--format svg`, `--output`, and `--check-svg` to the existing generator so
  the docs build and checked-in asset use one rendering interface.
- Make `--check-svg` compare Graphviz node and edge groups with the canonical
  DOT inventory. This is semantic rather than byte-for-byte comparison because
  coordinates and generator comments vary across Graphviz releases.
- Preserve the original image accessibility with root `role="img"`, an
  accessible title, and a dependency-direction description added after
  Graphviz rendering.
- Use a white SVG background so labels remain legible when GitHub or a browser
  uses a dark page theme.

## Alternatives Rejected

- Adjusting the incorrect SVG path coordinates would repair the current image
  but retain the source of the regression.
- Mermaid would add a second graph description despite the existing DOT
  generator and Graphviz installation in documentation CI.
- A byte-for-byte generated-file check would make contributors regenerate with
  the exact CI Graphviz release. The semantic inventory check catches stale
  crates and dependencies without coupling correctness to layout-engine
  coordinates.

## Content-Sizing Refinement

After visual review, the canonical DOT was tightened without introducing fixed
geometry: nodes use `fixedsize=false` and compact label margins, clusters use
local ranking and an explicit 8-point content margin, and graph/rank/node
spacing was reduced. Cluster headings are plain, content-sized layout
nodes because native cluster labels are not obstacles to Graphviz's edge
router. The regenerated image was inspected at 1800 pixels wide; labels,
cluster headings, borders, and arrowheads remain unclipped and aligned, and no
dependency stroke crosses a heading.

## Verification Performed

- TDD red/green cycles for current layer classification, Graphviz invocation,
  checked-in SVG inventory, docs-build integration, and accessible SVG metadata
- `python3.12 scripts/test-gen-dep-graph.py`
- `python3.12 scripts/gen_dep_graph.py --check-svg docs/assets/dependency-footprint.svg`
- `PATH=/tmp/tenferro-python312-bin:$PATH python3.12 -m unittest discover -s scripts/ci/tests -v` (155 tests)
- `PATH=/tmp/tenferro-python312-bin:$PATH bash scripts/check-pr-fast.sh --coverage-reviewed --test '/opt/homebrew/bin/python3.12 scripts/test-gen-dep-graph.py' --test '/opt/homebrew/bin/python3.12 -m unittest scripts.ci.tests.test_run_profile -v'`
- Independent code review found no critical or important issues; its two minor
  test-hardening suggestions (edge direction/reduction invariants and
  `aria-labelledby` reference validation) were incorporated
- Rendered the generated SVG to a 1600-pixel PNG with `rsvg-convert` and checked
  that arrowheads meet crate-box borders without overlapping crate labels

## Residual Risks

- Graphviz releases may choose different but semantically equivalent layouts.
  The checked-in SVG can be regenerated when a new layout is preferred; the
  inventory check intentionally does not reject version-only coordinate drift.
- Adding a workspace crate requires selecting its user-facing layer. Until it
  is classified, it remains visible in the fallback cluster and the current
  workspace-layer regression test fails.
