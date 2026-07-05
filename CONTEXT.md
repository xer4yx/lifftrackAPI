# LiftTrack

Real-time weightlifting form coaching. A Flutter client captures the lifter on
camera and streams pose data to a FastAPI backend, which scores exercise form
and returns feedback the client renders live.

## Language

**Live Session**:
A single websocket connection during which a lifter performs an exercise and
receives real-time form feedback. Ends on disconnect or a completion signal.
_Avoid_: stream, workout (a workout may span several sessions)

**Keypoint**:
A single tracked body landmark with a position and a confidence. Produced
on-device by the pose model and streamed to the server. The contract is the
17 COCO keypoints (MoveNet) in normalized [0,1] image coordinates.
_Avoid_: joint (a joint is the angle between three keypoints), landmark, point

**Pose**:
The full set of a lifter's keypoints for one frame.

**Form Score**:
A 0–1 accuracy value for how well a rep matches correct form for the exercise,
with human-readable suggestions.
_Avoid_: accuracy (used in code, but overloaded), grade

**Bands Message**:
The lightweight, high-frequency (~10–15/sec) websocket message carrying the
current per-joint Risk Bands (colors only) for live heatmap rendering. Carries
no positions — the client paints the colors onto the keypoints it already holds.
Not persisted.
_Avoid_: frame (no image is sent), update

**Coaching Message**:
The ~1/sec websocket message carrying the Form Score and suggestions, consumed
by on-screen text and spoken aloud via TTS. Shares its scoring pass with the
per-second persistence write.
_Avoid_: feedback, tip

**Feature Metric**:
One of the session-level quality measures (body alignment, joint consistency,
load control, speed control, overall stability) computed from features and
persisted at session end.

**Heatmap**:
A body overlay coloring each tracked joint by form-deviation severity — green
(within the correct range), orange (slight deviation), red (severe deviation).
It visualizes existing form analysis; it is not an injury-risk prediction.
Rendered on the client.
_Avoid_: skeleton, overlay (the heatmap is one kind of overlay)

**Risk Band**:
The discrete form-deviation classification of a joint: green, orange, or red.
Reflects distance from the correct range, not a clinical injury probability.
_Avoid_: zone, level, injury level

**Correct Range**:
The `[lo, hi]` span a joint angle should stay within for a given exercise.
Inside is green; a tolerance past an edge is orange; beyond is red. Defined
per exercise in the angle-range table, which is the single source of truth for
the heatmap, the form score, and suggestions.
_Avoid_: ideal angle, target (a range, not a single value)
