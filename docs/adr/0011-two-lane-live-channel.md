# Two-lane live channel; positions never returned; table stays server-side

**Status:** accepted

## Context

With pose on-device ([0001](0001-client-side-pose-estimation.md)) and cadence
decoupled from persistence ([0006](0006-decouple-live-and-persistence-cadence.md)),
we needed the concrete live websocket contract. Two facts shaped it: the client
already holds the keypoint positions it produced, and the app speaks feedback
aloud via TTS, which cannot consume a 15/sec stream.

## Decision

The live socket carries **two message types at two cadences**:

- **Bands** (~10–15/sec): `{ "type": "bands", "t": <echoed client timestamp>,
  "bands": { joint_name: "green"|"orange"|"red" } }`. Colors only — **no
  positions**. The client paints the colors onto the keypoint positions it
  already has locally. `t` lets the client drop stale bands and align them.
- **Coaching** (~1/sec): `accuracy` + `suggestions[]`, consumed by on-screen
  text and TTS. Computed in the same 1/sec scoring pass that writes to Firebase.

Bands are **server-authoritative**; the client paints what it receives and may
interpolate colors between ticks for smoothness. The per-exercise angle-range
table stays in **one place (the server)** — it is not duplicated into Dart.

## Consequences

- The live-render payload is a handful of enums per tick; no image and no
  coordinates cross the wire, and the heatmap stays glued to the live camera
  because positions are always the client's own latest.
- If server round-trip latency makes 15/sec feel rubbery on cellular, the
  fallback is on-device provisional bands — which requires porting the
  angle-range table to Dart and is deferred to avoid a second copy to keep in
  sync (see `ROADMAP.md`).
- The coaching lane and the persistence write share one scoring pass.
