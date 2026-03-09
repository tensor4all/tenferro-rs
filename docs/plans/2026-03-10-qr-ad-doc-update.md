# QR AD Doc Update Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update the QR AD formula notes so the wide reduced-QR backward section matches the current `tenferro-linalg` implementation.

**Architecture:** Keep the existing full-rank QR reverse rule intact. Rewrite only the `M < N` reverse-mode section in `docs/AD/qr.md` to describe the PyTorch-aligned reduced-QR backward formula that `tenferro-linalg::qr_rrule` now uses, including the helper operator and embedding of the leading block into the full-width matrix.

**Tech Stack:** Markdown docs, `tenferro-linalg` source, `docs/AD/qr.md`

---

### Task 1: Capture the implementation-backed formula

**Files:**
- Read: `tenferro-linalg/src/lib.rs`
- Read: `docs/AD/qr.md`
- Modify: `docs/AD/qr.md`

**Step 1: Inspect the current wide-case reverse rule**

Confirm the exact implementation shape in `qr_rrule` for the `m < n` branch and note the helper objects used in the code.

**Step 2: Rewrite the wide-case section**

Replace the existing partition-based backward description with the actual reduced-QR formula used in the implementation:
- define `R_1`
- define the skew-lower helper used in the leading block
- describe the embedded leading-block term and the `Q \bar{R}` term

**Step 3: Add a short implementation note**

State that the current rule is aligned with PyTorch's reduced-QR backward for the real case.

### Task 2: Sanity-check the updated note

**Files:**
- Read: `docs/AD/qr.md`

**Step 1: Re-read for notation consistency**

Verify symbols (`Q`, `R`, `R_1`, `\bar{Q}`, `\bar{R}`) are used consistently with the surrounding document.

**Step 2: Verify references remain accurate**

Ensure the section still reads as a formula note under `docs/AD/index.md` without requiring index changes.
