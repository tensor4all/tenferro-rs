#!/usr/bin/env node
/**
 * SHA-bound review waiver.
 *
 * The rule this replaces bound the exemption to "a maintainer applied the label
 * at some point" and tried to date the last head update from `created_at` on
 * `committed` timeline events. Those events do not carry `created_at`, so the
 * staleness branch never fired: the label kept working after any push.
 * `committer.date` is not a substitute — the author controls it.
 *
 * The rule implemented here:
 *
 *   A waiver is a comment by a maintainer containing
 *   `waive-review <40-character sha>`. It is valid only while that SHA is the
 *   pull request's current head. The label is a pointer for humans; it grants
 *   nothing on its own.
 *
 * Because the binding is to a commit and not to a timeline event, close/reopen
 * needs no special case: the same head keeps its waiver, a pushed head loses it.
 *
 * Plain CommonJS, no dependencies, no network inside the decision.
 */

"use strict";

/** Associations GitHub treats as write access or better. */
const MAINTAINER_ASSOCIATIONS = new Set(["OWNER", "MEMBER", "COLLABORATOR"]);

/** Repository roles fetched with `getCollaboratorPermissionLevel`. */
// Narrower than "write": a waiver removes the LLM review gate for one head, and widening the set of
// people who may do that is a policy change, not part of binding the waiver to a sha.
const MAINTAINER_PERMISSIONS = new Set(["admin", "maintain"]);

const FULL_SHA = /^[0-9a-fA-F]{40}$/;

/**
 * `waive-review <sha>`, optionally followed by punctuation, in the same
 * paragraph. Hex runs shorter than 40 characters are matched too so they can be
 * rejected explicitly instead of being silently ignored.
 */
const WAIVER_TOKEN = /\bwaive-review\b[ \t]*[:=]?[ \t]*`?([0-9a-fA-F]{7,64})`?(?![0-9a-zA-Z])/g;

function isMaintainer(comment) {
  const association = (comment.author_association || "").toUpperCase();
  if (MAINTAINER_ASSOCIATIONS.has(association)) return true;
  const permission = (comment.author_permission || "").toLowerCase();
  return MAINTAINER_PERMISSIONS.has(permission);
}

/** Every waiver token in one comment body, in order of appearance. */
function tokens(body) {
  const found = [];
  if (typeof body !== "string") return found;
  WAIVER_TOKEN.lastIndex = 0;
  let match;
  while ((match = WAIVER_TOKEN.exec(body)) !== null) {
    found.push(match[1].toLowerCase());
  }
  return found;
}

function orderKey(comment, index) {
  const at = Date.parse(comment.created_at);
  return [Number.isFinite(at) ? at : Number.MAX_SAFE_INTEGER, comment.id ?? index, index];
}

function newestFirst(comments) {
  return comments
    .map((comment, index) => ({ comment, key: orderKey(comment, index) }))
    .sort((a, b) => {
      for (let i = 0; i < a.key.length; i += 1) {
        if (a.key[i] !== b.key[i]) return a.key[i] - b.key[i];
      }
      return 0;
    })
    .reverse()
    .map((entry) => entry.comment);
}

function evidence(comment, sha) {
  return {
    commentId: comment.id ?? null,
    author: comment.user?.login ?? null,
    association: comment.author_association ?? null,
    createdAt: comment.created_at ?? null,
    namedSha: sha,
    url: comment.html_url ?? null,
  };
}

/**
 * Decide whether the current head may skip the review.
 *
 * @param {object} input
 * @param {string} input.headSha        current `pull_request.head.sha`
 * @param {Array}  [input.comments]     issue comments (`issues.listComments`)
 * @param {boolean}[input.labelPresent] whether the exemption label is applied
 * @param {string} [input.action]       `github.event.action`; recorded only —
 *                                      the SHA binding makes it irrelevant
 * @returns {{allowed: boolean, code: string, reason: string, namedSha?: string,
 *            evidence?: object}}
 */
function evaluate({ headSha, comments = [], labelPresent = false, action = null } = {}) {
  if (typeof headSha !== "string" || !FULL_SHA.test(headSha)) {
    return {
      allowed: false,
      code: "invalid-head",
      reason: `pull request head is not a full 40-character sha (${headSha === undefined ? "missing" : JSON.stringify(headSha)})`,
    };
  }
  const head = headSha.toLowerCase();

  const withTokens = [];
  for (const comment of newestFirst(comments || [])) {
    for (const sha of tokens(comment.body)) {
      withTokens.push({ comment, sha });
    }
  }

  const maintainerTokens = withTokens.filter((entry) => isMaintainer(entry.comment));

  // The newest maintainer instruction wins: re-issuing a waiver for another head
  // revokes the earlier one instead of leaving two live approvals.
  if (maintainerTokens.length > 0) {
    const { comment, sha } = maintainerTokens[0];
    if (sha === head) {
      return {
        allowed: true,
        code: "waiver-matches-head",
        reason:
          `waiver from ${comment.user?.login ?? "?"} (${comment.author_association ?? "?"}) ` +
          `names the current head ${head}`,
        namedSha: sha,
        evidence: evidence(comment, sha),
      };
    }
    if (sha.length !== 40) {
      return {
        allowed: false,
        code: "shorthand-waiver",
        reason:
          `waive-review in ${comment.user?.login ?? "?"}'s comment names "${sha}", ` +
          `which is not a full 40-character sha; the waiver cannot be tied to a commit`,
        namedSha: sha,
        evidence: evidence(comment, sha),
      };
    }
    return {
      allowed: false,
      code: "stale-waiver",
      reason:
        `the waiver in ${comment.user?.login ?? "?"}'s comment names ${sha}, ` +
        `but the pull request head is ${head}; ask a maintainer to post ` +
        `\`waive-review ${head}\` after the latest push`,
      namedSha: sha,
      evidence: evidence(comment, sha),
    };
  }

  const foreign = withTokens.find((entry) => !isMaintainer(entry.comment));
  if (foreign) {
    return {
      allowed: false,
      code: "non-maintainer-waiver",
      reason:
        `waive-review in a comment by ${foreign.comment.user?.login ?? "?"} ` +
        `(${foreign.comment.author_association ?? "?"}) does not grant a waiver; ` +
        `write access or better is required`,
      namedSha: foreign.sha,
      evidence: evidence(foreign.comment, foreign.sha),
    };
  }

  return {
    allowed: false,
    code: "no-waiver",
    reason: labelPresent
      ? `an exemption label is applied but no maintainer comment says \`waive-review ${head}\`; ` +
        `the label grants nothing on its own`
      : `no maintainer comment says \`waive-review ${head}\``,
  };
}

/**
 * `actions/github-script` entry point. Fails the job when the waiver does not
 * bind to the current head.
 */
async function run({ github, core, context, labelName = "rules-review:waive" }) {
  const owner = context.repo.owner;
  const repo = context.repo.repo;
  const pr = context.payload.pull_request;
  const issue_number = pr.number;

  const comments = await github.paginate(github.rest.issues.listComments, {
    owner,
    repo,
    issue_number,
    per_page: 100,
  });

  // `author_association` is enough for the common case; a collaborator whose
  // association is stale still gets a role lookup so the two paths agree.
  for (const comment of comments) {
    if (comment.author_permission || !comment.user?.login) continue;
    try {
      const { data } = await github.rest.repos.getCollaboratorPermissionLevel({
        owner,
        repo,
        username: comment.user.login,
      });
      comment.author_permission = data.role_name;
    } catch (error) {
      core.warning?.(
        `could not read the repository role of ${comment.user.login}: ${error.message}`
      );
    }
  }

  const decision = evaluate({
    headSha: pr.head?.sha,
    comments,
    labelPresent: (pr.labels || []).some((label) => label.name === labelName),
    action: context.payload.action,
  });

  if (decision.allowed) {
    core.info?.(`waiver accepted: ${decision.reason}`);
    return decision;
  }
  core.setFailed(
    `Review waiver rejected (${decision.code}) for ${labelName}: ${decision.reason}`
  );
  return decision;
}

module.exports = {
  FULL_SHA,
  MAINTAINER_ASSOCIATIONS,
  MAINTAINER_PERMISSIONS,
  WAIVER_TOKEN,
  evaluate,
  isMaintainer,
  run,
  tokens,
};
