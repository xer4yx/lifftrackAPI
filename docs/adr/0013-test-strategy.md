# Test strategy for the v3 rewrite

**Status:** accepted

## Context

Step 1 consolidated to one pipeline; Steps 2-3 fix the surviving path's
correctness bugs and introduce the per-exercise angle-range table (ADR
[0005](0005-per-exercise-angle-range-table.md)). A `/code-review` pass over the
consolidated path confirmed the bugs are almost all pure functions over
keypoints (inverted body-alignment reference, confidence used as a z-coordinate
in the angle math, missing-keypoint defaults, exercise-name vocabulary drift).
The recurring failure mode is a metric that runs but returns a constant — the
`load_control`-always-0 lesson — so "the test executed the code" is not enough.

## Decision

A full pyramid, with each level given a distinct job:

- **Unit (the bulk).** Pure-function tests on `core/service`
  (`extract_body_alignment`, `extract_joint_angles`, `FormAnalysisService`,
  `FeatureMetricService`). This is where the correctness bugs are killed.
- **Integration.** Through `ComVisUseCase` with an in-memory Firebase fake
  (implements the DB interface, injected via `dependency_overrides`), asserting
  what was persisted and under which key.
- **End-to-end.** Drawn at the **keypoints-in contract**
  (`process_frame_with_keypoints`: keypoints dict -> `FormAnalysis` + features),
  deterministic because no server ML is involved. Promoted to a transport-level
  websocket test when the Step 4 endpoint lands (ADR
  [0012](0012-contract-first-synchronized-cutover.md)), reusing the same
  fixtures.

Supporting decisions:

- **Fixtures are the contract artifact.** Synthetic, analytically-known
  keypoints (upright torso -> vertical alignment ~0 deg, right-angle elbow ->
  90 deg) for exact golden assertions, plus 1-2 recorded good/bad reps per
  exercise. All in the normalized [0,1], y-down convention of ADR
  [0003](0003-keypoint-contract-17-coco.md).
- **Authoring is strict red-green per finding.** Each fix ships a regression
  test that provably fails against the pre-fix code.
- **Metric guardrail: discrimination + anti-degeneracy.** Every metric test
  asserts (a) a good rep scores higher than a bad rep, and (b) the score is not
  a degenerate constant (0 or 100) across a varied sequence. Stateful metrics
  (deque history) get a multi-frame sequence and a `reset_history` assertion.
- **Angle-range table (Step 3) tests three layers:** data integrity (every
  canonical exercise present; each angle has `lo < hi` and positive tolerance),
  parametrized band logic (inside -> green, within tolerance past an edge ->
  orange, beyond -> red, including boundaries), and a **vocabulary-closure**
  test that the table's exercise and angle names exactly match what the pipeline
  emits — the test that catches the `romanian_deadlift`/`rdl` class of drift.
- **Property-based testing** (`hypothesis`) for the geometry core: translation
  invariance, symmetry `angle(a,b,c) == angle(c,b,a)`, output bounded [0,180],
  and "confidence must not affect the angle" (which fails on the 3-tuple bug).
- **Cross-repo contract** lives in a versioned `contract/` dir (JSON schema plus
  golden keypoint/bands/coaching messages), canonical in this repo and copied to
  the app; a checksum/version test each side turns drift into a failing check.
- **CI gate.** `run-tests.yml` triggers must include the `v3.0.0` base (they
  currently fire only on `main`/`master`, so the rewrite PRs run no tests). The
  pytest job is a required check; coverage stays informational (a coverage-%
  gate is what let `load_control` through). The app repo gains an equivalent
  `flutter test`/`analyze` workflow.

## Consequences

- New test tooling: `pytest-asyncio`, `pytest-cov`, `hypothesis`.
- **Scope pull-forward:** the keypoints-in e2e boundary means the
  `PoseFeatureService.process_features` bugs (positional `BodyAlignment`
  construction; `speed=` vs the `speeds` field) move from Step 4 into Step 2.
  They are dead code today, so the change is low-risk.
- The API repo owns the canonical `contract/` dir since it defines the
  metrics/bands semantics; the app consumes a copy.
- Dormant code slated for Step 4 removal (`compute_lc_score`, the
  `InferenceUseCase` object/action branches) is intentionally not tested.
