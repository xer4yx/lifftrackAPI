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

The app reads the websocket base URL from a build-time define (`LiveConfig`,
default `wss://proxmox.lift-track.com`). For a **local** dev API, build the
debug APK against your PC's LAN IP over plain `ws://`:

```sh
# in the app repo (Z:/liftttrack)
flutter build apk --debug --dart-define=LIVE_WS_BASE_URL=ws://<PC-LAN-IP>:8000
```

The client then connects to `ws://<PC-LAN-IP>:8000/v2/exercise-tracking?...`,
streams `keypoints`, and consumes `bands`/`coaching`. Use the in-app floating
**DEBUG** overlay to trace any runtime errors on the device.
