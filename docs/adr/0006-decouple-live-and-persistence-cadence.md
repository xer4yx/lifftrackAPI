# Decouple live heatmap cadence from persistence cadence

**Status:** accepted

## Context

Analysis historically ran once per second (every 30th frame) because three
server ML models had to run. With pose moved on-device and server ML removed,
scoring is now pure angle math (microseconds), so the once-per-second rate is
no longer a compute constraint. The heatmap needs to feel live; persistence
does not need to be high-frequency.

## Decision

Two consumers, two rates:

- **Live bands** — the client streams keypoints at ~10–15/sec; the server
  scores each and returns a lightweight per-joint bands message. Not persisted.
  The client may interpolate colors between ticks for perceived smoothness.
- **Persistence** — the existing once-per-second aggregation continues to write
  `exercise_data` to Firebase RTDB, and end-of-session `feature_metrics` are
  written as before.

## Consequences

- The Firebase write path and schema are unchanged; only a new, non-persisted
  bands message is added.
- The wire protocol now has two outbound message types at two cadences; the
  send rate is a tunable trading smoothness against battery/bandwidth.
