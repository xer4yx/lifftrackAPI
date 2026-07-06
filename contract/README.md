# LiftTrack live wire contract

The versioned contract for the live exercise-tracking websocket. It is
**canonical in this repo (`lifftrackAPI`)** and mirrored verbatim into the app
repo (`liftttrack`). Both sides validate against these schemas so the protocol
cannot drift silently (ADR 0012).

## Messages

- **`keypoints`** (client → server, ~15/sec) — 17 COCO keypoints, normalized to
  `[0,1]` image coordinates, with `confidence`, plus a client timestamp `t`
  (ADR 0001, 0003).
- **`bands`** (server → client, ~10–15/sec) — per-joint form-deviation colors
  keyed by vertex keypoint (`green`/`orange`/`red`), and the echoed `t`. Colors
  only — no positions; the client paints them onto its own keypoints
  (ADR 0011, 0004).
- **`coaching`** (server → client, ~1/sec) — `accuracy` + `suggestions[]`, for
  on-screen text and TTS, from the same scoring pass that persists to Firebase
  (ADR 0011).

## Files

- `PROTOCOL_VERSION` — semantic version of the contract (currently `1.0.0`).
- `schemas/*.schema.json` — JSON Schema (2020-12) for each message.
- `examples/*.example.json` — golden messages, validated in CI on both repos.

## Changing the contract

Any change is a coordinated cross-repo event:

1. Edit the schema(s) and bump `PROTOCOL_VERSION`.
2. Update `EXPECTED_CONTRACT_CHECKSUM` in `tests/contract/test_contract.py`
   (the test prints the new value on failure).
3. Copy the whole `contract/` dir into the app repo in the same change.

The checksum is over the **canonical JSON** of the schemas (formatting- and
line-ending-independent), so an accidental divergence between the two repos
fails the contract test on the lagging side.
