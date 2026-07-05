# Split into Core API and Websocket API (two apps, one codebase)

**Status:** accepted

## Context

The system was a single FastAPI monolith mounting REST routers and websocket
routers together. A surge of long-lived websocket connections can starve the
shared event loop and thread pool, degrading or taking down the REST endpoints
(login, history) in the same process. Stateful live-session traffic and
stateless REST traffic have different resource and scaling profiles.

The original instinct was a three-app split (core, auth, websocket) partly for
security. That security rationale does not hold (see
[0008](0008-client-attestation-not-cors.md)): auth is stateless JWT validation
that every app can do independently, so it needs no separate process.

## Decision

Split into **two deployables over one shared codebase**:

- **Core API** — auth endpoints, users, progress, metrics. Stateless; scales by
  request rate.
- **Websocket API** — live exercise tracking. Stateful, long-lived connections;
  scales by concurrent connections.

Both apps import the same `core/` / `infrastructure/` / `interface/` layers and
differ only in which routers they mount (separate entrypoints, e.g.
`main_core.py` and `main_ws.py`). Auth stays in Core; both apps validate the
same JWTs with a shared signing key and read the shared token blacklist from
Firebase. CORS is configured per app, but is not a security boundary here.

## Consequences

- A websocket surge can only affect the websocket tier; auth and history stay up.
- No third auth app; no token-introspection coupling.
- Two entrypoints must be kept in sync on shared middleware/config; the shared
  layers must stay genuinely shared (reinforces the maintainability work).
- This split is only worthwhile once the eager server-side model loading is
  removed; otherwise both apps inherit the slow startup.
