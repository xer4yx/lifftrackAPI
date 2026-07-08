# Configuration & local run (manual testing)

The API needs two git-ignored config files to run. Templates are checked in;
copy and fill them with real values.

## 1. `config.ini` (required to boot)

Read at import time (`lifttrack/__init__.py`) with no fallbacks — the server
**won't start** without it.

```sh
cp config.ini.example config.ini
```

Fill:
- `[Authentication]` — `SECRET_KEY` (generate: `python -c "import secrets; print(secrets.token_hex(32))"`), `ALGORITHM` (`HS256`), `TTL` (minutes).
- `[Firebase]` — `FIREBASE_DEV_DB` (RTDB URL), `FIREBASE_AUTH_UID`.

## 2. `.env` (Firebase runtime: login, session persistence)

Loaded by pydantic settings (`utils/*`).

```sh
cp .env.example .env
```

Fill the `FIREBASE_*` values (same DB URL/UID as `config.ini`, plus a
`FIREBASE_ADMIN_SDK` path to your service-account JSON and `FIREBASE_AUTH_TOKEN`).
`APP_*`/`CORS_*` are optional; `ROBOFLOW_*`/`MONGO_*` aren't needed to boot.

Download the service-account JSON from Firebase Console → Project settings →
Service accounts → *Generate new private key*, and point `FIREBASE_ADMIN_SDK`
at it. Keep it out of git.

## 3. Run the API

```sh
.venv/Scripts/python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` so a phone on the same LAN can reach it. Sanity check:
`GET http://<PC-LAN-IP>:8000/ping` → `{"status":"ok"}`.

## 4. Point the app at this API (manual keypoints-in test)

The app takes **two** build-time defines (both default to the production
`proxmox.lift-track.com`): `API_BASE_URL` for HTTP (login/register/profile/
progress) and `LIVE_WS_BASE_URL` for the live websocket. For a **local** dev API
you must set **both** to your PC's LAN IP — over plain `http://`/`ws://` — or the
app will authenticate against production and never reach your server.

Serve on the LAN IP (not localhost) so the phone can reach it, then build:

```sh
# API (repo root, venv) — host/port must match the app's defines
.venv/Scripts/python.exe -m uvicorn main:app --host <PC-LAN-IP> --port <PORT>

# App (Z:/liftttrack) — same host/port
flutter build apk --debug \
  --dart-define=API_BASE_URL=http://<PC-LAN-IP>:<PORT> \
  --dart-define=LIVE_WS_BASE_URL=ws://<PC-LAN-IP>:<PORT>
```

The client authenticates over HTTP, then connects to
`ws://<PC-LAN-IP>:<PORT>/v2/exercise-tracking?...` (the `/v2` prefix is the
**v3 keypoints-in** handler), streams `keypoints`, and consumes `bands`/
`coaching`. The phone must be on the same Wi-Fi/LAN, and Windows Firewall must
allow inbound on `<PORT>`. Use the in-app floating **DEBUG** overlay (with the
`pipeline 2s: …` counters) to trace runtime issues on the device; press
**Start** to begin streaming.
