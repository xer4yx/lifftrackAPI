# "Only our app" is enforced by client attestation, not CORS

**Status:** accepted

## Context

A hard requirement is that only the official LiftTrack app may call the API.
The initial plan leaned on CORS and an isolated auth app for this. The client
is mobile-only (Flutter iOS/Android).

CORS is enforced by browsers, not servers; a native mobile client and any
direct caller (e.g. `curl`) ignore CORS headers entirely. CORS therefore
provides essentially none of the "only our app" protection for a mobile-only
product.

## Decision

Enforce "only our app" with **client attestation** — Play Integrity (Android)
and App Attest (iOS) — verified server-side, layered on top of the existing
controls: JWT auth, TLS, rate limiting, and the token blacklist. CORS is
retained only as ordinary browser hygiene (near-irrelevant while mobile-only)
and is not treated as a security control.

Implementation is tracked in `ROADMAP.md`; this ADR records that attestation —
not CORS and not an auth-service split — is the chosen mechanism.

## Consequences

- Security reviews must not credit CORS with access control.
- The API gains an attestation-verification step; unattested clients are
  rejected regardless of a valid JWT.
- If a web client is ever added, CORS becomes a real concern for that surface
  and needs its own scoping.
