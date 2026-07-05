# Consolidate to one pipeline; fixed execution order; hard cutover

**Status:** accepted

## Context

The codebase carries three generations of the exercise pipeline mounted at once
(`routers/WebsocketRouter`, `routers/v2/WebsocketRouter`, `interface/ws` v3),
with feature/form logic duplicated between the legacy `lifttrack/v2/comvis`
package and the clean-architecture `core/service`. Most correctness bugs the
code review found exist in both copies. The app has never been deployed to real
users — it was a thesis project — so there are no production clients to migrate.

## Decision

The **clean-architecture path** (`interface/ws` v3 + `core/service` +
`core/usecase` + `infrastructure`) is canonical. Everything else is legacy and
gets deleted. Because there are no deployed clients, the protocol change is a
**hard cutover** — no strangler, no parallel endpoints.

Work proceeds in this order (delete before fix, fix before build, consolidate
before split):

1. Delete legacy; collapse to the single clean-architecture path (removes
   `routers/` v1+v2, the deprecated `/inference` route, and the server ML
   services — likely TensorFlow entirely).
2. Fix the surviving path's correctness bugs (coordinates, angles, body
   alignment, exercise-name vocabulary, broken metrics).
3. Introduce the per-exercise angle-range table (ADR
   [0005](0005-per-exercise-angle-range-table.md)).
4. Migrate the wire protocol to keypoints-in / bands-out (ADR
   [0001](0001-client-side-pose-estimation.md)).
5. Build the heatmap.
6. Split into Core and Websocket apps (ADR
   [0007](0007-split-core-and-websocket-apps.md)).
7. Pre-launch security: client attestation + minimum-supported-version gate.

## Consequences

- The heatmap is deliberately held until after steps 1–3 so it renders correct
  form data, not the inverted/false-positive signals the review found.
- A minimum-supported-version gate must exist before public launch so a future
  breaking change never bricks real clients (the current app has only a soft
  "update available" notice). Tracked in `ROADMAP.md`.
