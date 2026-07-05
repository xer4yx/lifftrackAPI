# Configuration: pydantic-settings with env-file layering

**Status:** accepted

## Context

Configuration was spread across three overlapping mechanisms: a `config.ini`
(configparser, ~39 call sites, mostly model URLs/paths), ad-hoc `load_dotenv`
calls, and a partly-built set of pydantic-settings `BaseSettings` classes.
The team wanted ASP.NET-style environment-specific settings
(`appsettings.{Environment}.json` selected by `ASPNETCORE_ENVIRONMENT`).

## Decision

Consolidate all configuration onto **pydantic-settings** as the single typed,
validated mechanism. Retire `config.ini` and ad-hoc `load_dotenv`. Most of
`config.ini` disappears anyway when the server ML models are removed in the
teardown.

Environment-specific config uses **env-file layering** (not literal JSON
appsettings files). Precedence, lowest to highest:

1. Field defaults in the `BaseSettings` classes
2. `.env.{APP_ENV}` file (non-secret, environment-shaped values)
3. Real process environment variables (highest — deploy overrides and secrets)

`APP_ENV` selects the environment (the `ASPNETCORE_ENVIRONMENT` analogue), one
of **`development`, `test`, `production`**. A small settings factory reads
`APP_ENV` once and builds the settings with the right `.env.{APP_ENV}`.

**Secrets** (Firebase credentials, JWT signing key, Roboflow key) live only in
environment variables or a secrets manager — never in a committed file. This is
the main reason literal `appsettings.{env}.json` was rejected: it invites
committed secrets.

**Domain data** (the per-exercise angle-range table, thresholds) is not app
settings; it lives in its own versioned, pydantic-validated data file.

## Consequences

- One config mechanism instead of three; validated at startup, so bad config
  fails the boot instead of surfacing as a runtime `None`.
- Both the Core and Websocket apps share the settings package and the same
  `APP_ENV` selection, each instantiating the subsets it needs.
- Committed `.env.{env}` files must be kept free of secrets by discipline and
  review.
