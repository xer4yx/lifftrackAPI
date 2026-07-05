# Client-side pose estimation; server scores keypoints

**Status:** accepted

## Context

The live exercise-tracking websocket originally streamed raw YUV420 camera
frames from the Flutter client to the server, which ran three pixel models
(MoveNet pose, Roboflow object detection, a 3D-CNN action classifier),
computed form features, scored them, and returned suggestions. This put image
transport and all model inference on the server's critical path, doubled image
traffic once we wanted to return a rendered heatmap, and made the server the
sole owner of a coordinate pipeline that had systemic geometric bugs
(aspect-ratio distortion, confidence treated as a spatial coordinate).

## Decision

Pose estimation moves **on-device** (Flutter, via an on-device pose model).
The client streams **keypoints** (not pixels) over the websocket. The server
computes form features, scores them, persists scores to Firebase RTDB, and
returns per-region risk scores as JSON. The **client renders the heatmap
overlay** from those scores. The heatmap rendering approach is recorded in a
later ADR.

## Consequences

- The inbound wire protocol changes from binary YUV frames to JSON keypoints;
  the keypoint schema (names, count, coordinate space, axis convention)
  becomes a pinned API contract.
- The server no longer loads MoveNet on the live path. Object detection and
  action recognition need pixels and therefore cannot run on keypoints alone —
  their fate is a separate decision.
- Feature-metric thresholds are defined in pixel space; the keypoint contract
  must pin coordinate space (or thresholds must be recalibrated).
- Firebase persistence, auth, and the scoring math are unaffected.
