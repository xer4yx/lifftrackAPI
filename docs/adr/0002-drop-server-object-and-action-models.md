# Drop object detection and action recognition from the live path

**Status:** accepted

## Context

Under [0001](0001-client-side-pose-estimation.md) the client streams keypoints,
not pixels. Roboflow object detection and the 3D-CNN action classifier both
require pixel frames and therefore cannot run on keypoints. We had to decide
whether to keep streaming frames for them, re-home them, or remove them.

## Decision

Remove both from the live path for now.

- **Action recognition** is redundant: the client already passes the chosen
  `exercise_name` as a required websocket parameter, so server-side exercise
  classification only ever overrode a value the user explicitly selected. It
  was also the largest single contributor to server startup cost.
- **Object detection** fed only the `load_control` feature metric, whose
  scoring was already structurally broken (always returned 0). Removing it
  drops a metric that did not work rather than one that did.

## Consequences

- The server loads **no heavy pixel models** at startup, which is the main
  lever for the startup-time and per-app-separation work.
- Two product capabilities go dormant, tracked in `ROADMAP.md`: automatic
  exercise classification, and equipment-aware form cues + `load_control`
  (the bench-press barbell/dumbbell branch is inert until then).
- `VideoActionInferenceService`, `RoboflowInferenceService`, and their DI
  wiring become dead on the live path and are candidates for the legacy
  teardown.
