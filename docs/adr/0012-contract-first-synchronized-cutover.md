# Contract-first, single synchronized protocol cutover across repos

**Status:** accepted

## Context

The migration to client-side pose changes the wire protocol on both the API
and the Flutter app. With a hard cutover (no deployed users — API repo ADR
[0009](0009-consolidate-to-one-pipeline-execution-order.md)), the two changes
must land together: there is no working end-to-end path if one repo flips weeks
before the other.

## Decision

Coordinate the two repos **contract-first**:

1. Write the wire contract down as a shared, versioned spec — the keypoint
   message (17 COCO, normalized [0,1], likelihood), the bands message, and the
   coaching message (ADRs [0003](0003-keypoint-contract-17-coco.md) and
   [0011](0011-two-lane-live-channel.md) define the content).
2. Both repos do independent cleanup in parallel first (neither touches the
   contract): API deletes legacy + fixes bugs + builds the angle-range table;
   app deletes legacy `lib/` duplicates and integrates ML Kit behind a flag on
   the old path.
3. **One synchronized flip:** the API keypoint protocol and the app livestream
   rewrite merge and deploy together — a single coordinated switch, no
   migration window, no compatibility shim.
4. Then heatmap on both sides against the live contract.

The Core/Websocket app split (ADR
[0007](0007-split-core-and-websocket-apps.md)) is near-invisible to the client —
it just points REST at the Core base URL and the websocket at the Websocket
base URL.

## Consequences

- The protocol change is treated as one change spanning two repos, not two
  independent backlogs.
- A compatibility shim (server accepting both old frames and new keypoints) was
  explicitly rejected — unnecessary without deployed users, and it would add
  throwaway code.
