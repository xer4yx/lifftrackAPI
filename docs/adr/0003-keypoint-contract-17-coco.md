# Keypoint contract: 17 COCO keypoints (MoveNet)

**Status:** accepted

## Context

Under [0001](0001-client-side-pose-estimation.md) the client's on-device pose
model output becomes the server's API contract. The realistic choices were
MoveNet-tflite (17 COCO keypoints, matching the server's existing
`KEYPOINT_DICT`) and ML Kit / BlazePose (33 landmarks, easier Flutter
integration, better fitness accuracy, richer injury-relevant landmarks).

## Decision

Pin the contract to the **existing 17 COCO keypoints**. The scoring code
(`PoseFeatureService`, all `joint_pairs`, form rules) keeps its current
keypoint assumptions unchanged; only the *source* of keypoints moves from
server MoveNet to the client.

**Update — on-device source is ML Kit, not MoveNet-tflite.** The Flutter client
had no pose infrastructure, and ML Kit Pose Detection (`google_mlkit_pose_detection`,
BlazePose, 33 landmarks) is far less client effort than a hand-rolled MoveNet
tflite interpreter. BlazePose is a superset of the 17 COCO points, so the client
runs ML Kit, **downselects to the 17 COCO keypoints**, normalizes to [0,1], and
maps ML Kit `likelihood` to confidence. The wire contract stays exactly 17 COCO
keypoints; the server is unaffected. This is also forward-compatible with the
roadmap pose upgrade — the client already produces 33 landmarks, so widening the
contract later needs no model swap.

BlazePose is the better long-term model and is the obvious thing a future
reader would reach for; we deliberately did not adopt it now to avoid coupling
the protocol migration to a full scoring remap. The upgrade path is in
`ROADMAP.md`.

## Consequences

- Zero change to keypoint names/indices server-side; the migration is a
  source swap, not a scoring rewrite.
- The `neck` and `waist` joint-pairs (absent from both MoveNet and BlazePose)
  remain dead and should be removed during cleanup regardless.
- Keypoints are streamed in **normalized [0,1] image coordinates**. Angles are
  scale-invariant and unaffected; the pixel-based motion thresholds
  (displacement, speed, stability, the resting check) must be recalibrated to
  the [0,1] scale. Normalized coordinates are resolution-independent but not
  framing-independent — body-scale normalization is deferred (see `ROADMAP.md`).
- Injury-relevant landmark detail is limited to shoulders/hips/knees/ankles
  until the roadmap pose-model upgrade.
