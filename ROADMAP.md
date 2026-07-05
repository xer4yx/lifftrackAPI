# LiftTrack Roadmap

Forward-looking goals. Decisions already made live in `docs/adr/`; this file
tracks capabilities we have deliberately deferred and the direction we intend
to take. Items are not commitments to a date — they are the agreed target shape
so that today's cuts don't quietly become permanent.

## Now (in progress)

- **Client-side pose estimation.** On-device pose model; client streams
  keypoints, server scores and persists, client renders the heatmap.
  (ADR [0001](docs/adr/0001-client-side-pose-estimation.md))
- **Drop server pixel models from the live path.** No MoveNet, Roboflow, or
  3D-CNN on the websocket. (ADR [0002](docs/adr/0002-drop-server-object-and-action-models.md))
- **Body heatmap with green/orange/red risk bands**, rendered live on the
  client from server-provided per-region scores.

## Deferred — targeted for later

### Automatic exercise classification (was: 3D-CNN)
Today the lifter selects the exercise before a session and we trust that value.
Target: a **skeleton-based classifier that runs on the streamed keypoints**
(e.g. ST-GCN / a small temporal model over the keypoint sequence), not a
pixel CNN. It confirms or corrects the selected exercise without needing frames
and without a heavy server model. Only worth building once keypoint quality and
the keypoint contract are stable.

### Equipment-aware coaching + load control (was: Roboflow object detection)
Today form cues that depend on equipment type (e.g. bench press barbell vs
dumbbell) are dormant, and the `load_control` metric is removed. Target: detect
equipment and track load steadiness **on-device** (client-side object
detection), streaming an equipment label + load-position signal alongside
keypoints. Reintroduce `load_control` only once its scoring is rebuilt
correctly (the previous implementation always scored 0).

### Pose-model upgrade
The on-device model is already ML Kit BlazePose (33 landmarks), downselected to
17 COCO for the wire contract. The upgrade is to **widen the contract** to send
more of the 33 landmarks (e.g. heels, foot index, per-landmark visibility) when
injury-relevant detail is needed — no model swap required, just contract and
scoring extensions.

### Minimum-supported-version gate (pre-launch)
Today the app only shows a soft "update available" notice. Before public launch,
add a `min_supported_version` to the app config the client reads at
`/app-update`, with a hard "update required" block below it, so a future
breaking protocol change can never brick installed clients. Not needed now
(no deployed users); required before launch.

### Client attestation ("only our app")
Enforce that only the official mobile app can call the API via Play Integrity
(Android) and App Attest (iOS), verified server-side, on top of JWT + TLS +
rate limiting. CORS is not a control for a mobile-only client.
(ADR [0008](docs/adr/0008-client-attestation-not-cors.md))

### Scale-invariant features
Move displacement / speed / stability off raw pixel units onto a body-scale
normalization (e.g. torso or shoulder-width units) so thresholds are consistent
across devices, camera distances, and resolutions.

## Guardrails for deferred work

- Nothing deferred here should reintroduce a **heavy pixel model on the server
  live path** — that is the specific cost ADR 0002 exists to avoid.
- Any reintroduced metric must ship with a test proving it produces non-trivial
  scores on real motion, so we do not repeat the silently-broken `load_control`.
