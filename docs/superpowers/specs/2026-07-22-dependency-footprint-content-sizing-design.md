# Dependency Footprint Content-Sizing Design

## Context

Issue #1441 replaces a hand-positioned dependency SVG with Graphviz output.
The generated arrows now meet crate boxes correctly, but the diagram still
uses generous node, cluster, rank, and canvas spacing. In particular,
`newrank=true` makes `dot` use one global ranking across clusters, which can
stretch a cluster far beyond the content it visually groups.

## Goal

Make every visual container in the checked-in dependency footprint fit its
content closely while preserving dependency semantics and arrow attachment:

- each rounded crate box fits its crate-name label;
- each colored cluster fits its heading and member crate boxes;
- the SVG canvas fits the complete drawing;
- spacing remains large enough to keep labels, borders, and arrows legible.

This is static content-based sizing. Responsive browser resizing and manual SVG
coordinate post-processing are out of scope.

## Design

Keep `dot` as the only layout engine and express sizing intent in canonical DOT
attributes:

- set node `fixedsize=false` explicitly so label size plus node `margin`
  determines the rounded rectangle;
- reduce node margin from the current oversized value to a compact, readable
  value;
- remove `newrank=true` so clusters receive local layout treatment instead of
  being stretched by a single global rank system;
- represent each cluster heading as a plain, content-sized node at the
  cluster's source rank. Native cluster labels are
  not edge-routing obstacles, so dependency strokes can cross their text;
  title nodes make the heading part of Graphviz's layout without coordinates;
- set an explicit compact cluster margin;
- reduce `nodesep` and `ranksep` to eliminate avoidable whitespace;
- set graph margin to zero and retain only a small drawing pad around the outer
  content.

Do not add fixed `width`, `height`, coordinates, or SVG post-processing for
geometry. Graphviz must continue to calculate all bounding boxes and edge
attachment points from labels and graph structure.

## Verification

Add a generator regression test that pins the content-sizing contract in the
DOT source: automatic node sizing is enabled, no fixed node dimensions are
introduced, local cluster ranking is preserved, and compact node, cluster,
rank, and canvas margins are present.

Regenerate `docs/assets/dependency-footprint.svg`, run the existing semantic
node/edge and accessibility checks, and render the SVG to PNG for visual
inspection. The image passes when:

- no label touches or crosses its rounded box;
- cluster headings and member nodes remain inside their cluster borders;
- no dependency stroke crosses a cluster heading;
- clusters no longer contain large areas created only by global rank
  alignment;
- arrowheads still terminate on node borders;
- no label or arrow is clipped by the SVG canvas.

Coordinate equality is intentionally not tested because Graphviz versions may
produce different but correct layouts.

## Risks

Removing global ranking can change the overall arrangement and edge lengths.
The semantic inventory test prevents dependency drift, while the rendered-image
inspection covers clipping, overlap, and readability. Compact spacing values
will remain explicit and centralized in `generate_dot` so they can be tuned
without editing generated SVG coordinates.
