#!/usr/bin/env node
/**
 * Tests for `scripts/review-waiver.js` — run with
 * `node --test scripts/test-review-waiver.js`.
 *
 * Fixtures are plain JSON under `scripts/fixtures/review-gate/waiver/` so each
 * decision can be replayed from a recorded comment set without the network.
 *
 * `legacyEvaluate` is kept here as the rule the workflow shipped before this
 * change: label presence plus a head-update timestamp read from `created_at` on
 * `committed` timeline events. Ordinary `committed` events do not carry that
 * field, which is why the staleness branch never fired and a waiver survived
 * any push while the label stayed applied.
 */

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const FIXTURES = path.join(__dirname, "fixtures", "review-gate", "waiver");
const waiver = require("./review-waiver.js");

function load(name) {
  return JSON.parse(fs.readFileSync(path.join(FIXTURES, name + ".json"), "utf8"));
}

/** The pre-fix decision: label presence + a timestamp the events do not carry. */
function legacyEvaluate(fixture) {
  const events = fixture.timeline || [];
  const labelEvents = events.filter(
    (event) => event.event === "labeled" && event.label && event.label.name
  );
  if (labelEvents.length === 0) return { allowed: false, code: "no-label-event" };
  if (fixture.action === "synchronize") return { allowed: false, code: "synchronize" };
  const labelTime = Date.parse(labelEvents[labelEvents.length - 1].created_at);
  const headTimes = events
    .filter(
      (event) =>
        ["committed", "head_ref_force_pushed"].includes(event.event) && event.created_at
    )
    .map((event) => Date.parse(event.created_at))
    .filter((time) => Number.isFinite(time));
  const latest = headTimes.length ? Math.max(...headTimes) : null;
  if (latest !== null && labelTime <= latest) {
    return { allowed: false, code: "label-older-than-head" };
  }
  if (!fixture.labelPresent) return { allowed: false, code: "no-label" };
  return { allowed: true, code: "legacy-label-trusted" };
}

function evaluateFixture(name) {
  return waiver.evaluate(load(name));
}

test("a waiver naming the current head is accepted", () => {
  const fixture = load("match");
  const decision = waiver.evaluate(fixture);

  assert.equal(decision.allowed, true, decision.reason);
  assert.equal(decision.code, "waiver-matches-head");
  assert.equal(decision.namedSha, fixture.headSha.toLowerCase());
});

test("a waiver naming an older head is rejected after a push", () => {
  const fixture = load("stale");
  const decision = waiver.evaluate(fixture);

  assert.equal(decision.allowed, false);
  assert.equal(decision.code, "stale-waiver");
  assert.match(decision.reason, /waive-review/);
});

test("the pre-fix logic accepted that same stale waiver (pinned)", () => {
  const fixture = load("stale");

  // The old rule trusted the label: `committed` events carry no `created_at`,
  // so `latestHeadUpdate` was always null and the staleness branch never ran.
  assert.equal(legacyEvaluate(fixture).allowed, true);
  assert.equal(waiver.evaluate(fixture).allowed, false);
});

test("reopening the PR with the same head keeps the waiver valid", () => {
  const fixture = load("reopen-same-sha");
  const decision = waiver.evaluate(fixture);

  assert.equal(fixture.action, "reopened");
  assert.equal(decision.allowed, true, decision.reason);
  assert.equal(decision.code, "waiver-matches-head");
});

test("a comment by someone below write access is rejected", () => {
  const fixture = load("non-maintainer");
  const decision = waiver.evaluate(fixture);

  assert.equal(legacyEvaluate(fixture).allowed, true);
  assert.equal(decision.allowed, false);
  assert.equal(decision.code, "non-maintainer-waiver");
  assert.match(decision.reason, /CONTRIBUTOR/);
});

test("a repository maintain role is accepted even when the association is odd", () => {
  const decision = evaluateFixture("permission-maintain");

  assert.equal(decision.allowed, true, decision.reason);
});

test("a repository write role is below the waiver threshold", () => {
  const fixture = load("permission-write");
  const decision = waiver.evaluate(fixture);

  // The pre-fix rule let the label stand regardless, so it accepted this too.
  assert.equal(legacyEvaluate(fixture).allowed, true);
  assert.equal(decision.allowed, false);
  assert.equal(decision.code, "non-maintainer-waiver");
});

test("the label alone grants nothing", () => {
  const fixture = load("label-only");
  const decision = waiver.evaluate(fixture);

  assert.equal(legacyEvaluate(fixture).allowed, true);
  assert.equal(decision.allowed, false);
  assert.equal(decision.code, "no-waiver");
  assert.match(decision.reason, /waive-review/);
});

test("an abbreviated sha is not a waiver", () => {
  const fixture = load("abbreviated");
  const decision = waiver.evaluate(fixture);

  assert.equal(decision.allowed, false);
  assert.equal(decision.code, "shorthand-waiver");
  assert.equal(decision.namedSha.length, 12);
});

test("the newest maintainer waiver wins over an older one", () => {
  const decision = evaluateFixture("newest-wins");

  assert.equal(decision.allowed, false);
  assert.equal(decision.code, "stale-waiver");
});

test("a missing or malformed head sha fails closed", () => {
  const fixture = load("match");

  for (const headSha of [undefined, null, "", "not-a-sha", fixture.headSha.slice(0, 12)]) {
    const decision = waiver.evaluate({ ...fixture, headSha });
    assert.equal(decision.allowed, false, `head ${headSha}`);
    assert.equal(decision.code, "invalid-head");
  }
});

test("the bare word does not grant a waiver", () => {
  const decision = waiver.evaluate({
    headSha: "a".repeat(40),
    comments: [
      {
        id: 1,
        user: { login: "maint" },
        author_association: "MEMBER",
        created_at: "2026-01-01T00:00:00Z",
        body: "I would waive-review this, but let me look first",
      },
    ],
  });

  assert.equal(decision.code, "no-waiver");
});

test("a hex run that is not exactly 40 characters is rejected", () => {
  const headSha = "a".repeat(40);

  for (const named of ["a".repeat(7), "a".repeat(12), "a".repeat(39), "a".repeat(41), "a".repeat(64)]) {
    const decision = waiver.evaluate({
      headSha,
      comments: [
        {
          id: 1,
          user: { login: "maint" },
          author_association: "MEMBER",
          created_at: "2026-01-01T00:00:00Z",
          body: "waive-review " + named,
        },
      ],
    });

    assert.equal(decision.allowed, false, `length ${named.length}`);
    assert.equal(decision.code, "shorthand-waiver", `length ${named.length}`);
  }
});

test("both `waive-review <sha>` and `waive-review: <sha>` are accepted", () => {
  const sha = "b".repeat(40);

  for (const body of [
    `waive-review ${sha}`,
    `waive-review: ${sha}`,
    `waive-review \`${sha}\``,
    `LGTM\n\nwaive-review ${sha}.`,
  ]) {
    const decision = waiver.evaluate({
      headSha: sha,
      comments: [
        {
          id: 1,
          user: { login: "maint" },
          author_association: "OWNER",
          created_at: "2026-01-01T00:00:00Z",
          body,
        },
      ],
    });
    assert.equal(decision.allowed, true, body);
  }
});

test("the decision never reads a timeline or committer date", () => {
  const fixture = load("stale");

  // Rewriting every timestamp must not change the verdict: freshness is a
  // SHA comparison, not a clock comparison.
  const rewritten = {
    ...fixture,
    comments: fixture.comments.map((comment) => ({
      ...comment,
      created_at: "2099-12-31T23:59:59Z",
    })),
    timeline: [],
  };

  assert.equal(waiver.evaluate(rewritten).code, "stale-waiver");
});

test("the github-script glue fails the job when the waiver is stale", async () => {
  const fixture = load("stale");
  const failures = [];
  const core = {
    setFailed: (message) => failures.push(message),
    warning: () => {},
    info: () => {},
  };
  const github = {
    paginate: async () => fixture.comments,
    rest: {
      issues: { listComments: async () => ({ data: fixture.comments }) },
      repos: {
        getCollaboratorPermissionLevel: async ({ username }) => ({
          data: { role_name: username === "maint" ? "maintain" : "read" },
        }),
      },
    },
  };
  const context = {
    repo: { owner: "tensor4all", repo: "tenferro-rs" },
    payload: {
      action: fixture.action,
      pull_request: {
        number: 42,
        head: { sha: fixture.headSha },
        labels: [{ name: "rules-review:waive" }],
      },
    },
  };

  const decision = await waiver.run({
    github,
    core,
    context,
    labelName: "rules-review:waive",
  });

  assert.equal(decision.allowed, false);
  assert.equal(failures.length, 1);
  assert.match(failures[0], /waive-review/);
  assert.match(failures[0], /rules-review:waive/);
});

test("the github-script glue passes the job when the waiver matches", async () => {
  const fixture = load("match");
  const failures = [];
  const core = {
    setFailed: (message) => failures.push(message),
    warning: () => {},
    info: () => {},
  };
  const github = {
    paginate: async () => fixture.comments,
    rest: {
      issues: { listComments: async () => ({ data: fixture.comments }) },
      repos: {
        getCollaboratorPermissionLevel: async () => ({ data: { role_name: "read" } }),
      },
    },
  };
  const context = {
    repo: { owner: "tensor4all", repo: "tenferro-rs" },
    payload: {
      action: "labeled",
      pull_request: {
        number: 42,
        head: { sha: fixture.headSha },
        labels: [{ name: "rules-review:no-llm" }],
      },
    },
  };

  const decision = await waiver.run({
    github,
    core,
    context,
    labelName: "rules-review:no-llm",
  });

  assert.equal(decision.allowed, true, decision.reason);
  assert.equal(failures.length, 0);
});
