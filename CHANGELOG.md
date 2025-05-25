# CHANGELOG


## v2.0.0 (2025-05-25)

### Bug Fixes

- Update inference package and Python version [skip ci]
  ([`f3d67b9`](https://github.com/xer4yx/lifftrackAPI/commit/f3d67b9359d9726be9c5a4b0dfce37e2a42ee5d7))

- Update inference package and Python version [skip ci]
  ([`7d52277`](https://github.com/xer4yx/lifftrackAPI/commit/7d52277871f14b7c1fa18b7db292a345a75ffbe6))

- **actions**: Fixed pre-commit action [skip ci]
  ([`41b7e96`](https://github.com/xer4yx/lifftrackAPI/commit/41b7e96be49a2c88907b4743531e327f487f2fe1))

- **requirements**: Fixed dependency compatibility in numpy [skip ci]
  ([`67518dc`](https://github.com/xer4yx/lifftrackAPI/commit/67518dccae704f53aef48276f65668e66e9e4312))

- **requirements**: Fixed dependency issues [skip ci]
  ([`9fabdfd`](https://github.com/xer4yx/lifftrackAPI/commit/9fabdfd3f2a189612db1d4b463fbc1f4626366a0))

- **requirements**: Fixed issues in inference-sdk compatibility [skip ci]
  ([`9f2e560`](https://github.com/xer4yx/lifftrackAPI/commit/9f2e560250f670a9594f6c07c3a55b9c47b1e475))

- **requirements**: Fixed package dependency conflicts [skip ci]
  ([`cb71618`](https://github.com/xer4yx/lifftrackAPI/commit/cb7161843f831fa5d9e6f3369fece4c8206694e0))

### Chores

- Added CHANGELOG, LICENSE, and workflows
  ([`6569647`](https://github.com/xer4yx/lifftrackAPI/commit/6569647c719fb80902b81ee16c963ce204a83d82))

- Updated README and requirement
  ([`9ed4050`](https://github.com/xer4yx/lifftrackAPI/commit/9ed40505efee18b7ac05c813c32685cc306e41ce))

- **.github**: Added and updated github actions
  ([`726e679`](https://github.com/xer4yx/lifftrackAPI/commit/726e679fadae6c7bf227a23b486112c7cc06b104))

### Features

- Major refactor of router components and API architecture
  ([`3930e45`](https://github.com/xer4yx/lifftrackAPI/commit/3930e45e79a51387bae64742a0f79d3e443f90fc))


## v1.10.4 (2025-05-10)

### Features

- Added database repository and di
  ([`8389730`](https://github.com/xer4yx/lifftrackAPI/commit/8389730553add1de4ce0be0acb6f94078ff26dce))


## v1.9.4 (2025-05-10)

### Features

- **core**: Added core layer
  ([`6bc7c20`](https://github.com/xer4yx/lifftrackAPI/commit/6bc7c2018fc5f3bc4b7a123833858a447f6ff23a))


## v1.8.4 (2025-05-10)

### Documentation

- Updated contents on README
  ([`987665a`](https://github.com/xer4yx/lifftrackAPI/commit/987665a08c15c98f918be2b761c17687fbacf05d))

### Features

- Added app config and resting state to features
  ([`98a31a4`](https://github.com/xer4yx/lifftrackAPI/commit/98a31a4395c03124250e4414351744f0dcb17925))

- **settings**: Added .env settings config
  ([`b8766ce`](https://github.com/xer4yx/lifftrackAPI/commit/b8766cea2aa46949a8a2eec555f103e9f2536c28))


## v1.7.4 (2025-04-09)

### Chores

- **progress**: Changed suggestion phrases
  ([`1b5d452`](https://github.com/xer4yx/lifftrackAPI/commit/1b5d45268781832cb7739fa43ca8150a5186d580))

## Changes: - Rephrased the suggestion into more concise and informative manner


## v1.7.3 (2025-04-08)

### Features

- **websocket**: Added parallelism to inference
  ([`f2aa599`](https://github.com/xer4yx/lifftrackAPI/commit/f2aa59959b5638bdacb98bab2f0cdb541fdc8f6a))


## v1.6.3 (2025-04-08)

### Chores

- **schema**: Separated schemas
  ([`f5383b5`](https://github.com/xer4yx/lifftrackAPI/commit/f5383b5490fb5f203f11bd50caa0a73484a13c20))

Separated schemas in each file

- **tests**: Added tests for comvis module
  ([`428b285`](https://github.com/xer4yx/lifftrackAPI/commit/428b285f00e8d40554761ec1db31fca5e7aebb52))

Added unit tests for comvis module

### Features

- **features**: Added additional calculations
  ([`7850a57`](https://github.com/xer4yx/lifftrackAPI/commit/7850a578fbd5d406f65bdf48f43c316216178ad4))


## v1.5.3 (2025-03-07)

### Features

- **ws**: Added livestream inference
  ([`b5ae360`](https://github.com/xer4yx/lifftrackAPI/commit/b5ae36001ef9ab8185035ed2e08ef45986408e4b))

Changes: - Added token validation when connecting to the websocket - Added bytes to yuv420 frame
  converter for livestream - Added router for websocket livestream


## v1.4.3 (2025-02-24)

### Features

- **ws**: Switch timeout connection to ack message
  ([`a460e10`](https://github.com/xer4yx/lifftrackAPI/commit/a460e103ec8b4524b3a8d20de05348489812247c))

Changes:

- Removed timeout connection

- Switched from acknowledgement message when closing websocket connection

- Removed debugging data on inference

This resolves #42 and #36


## v1.3.3 (2025-02-15)

### Features

- **http**: Added http connection pooling
  ([`e290755`](https://github.com/xer4yx/lifftrackAPI/commit/e290755b5ee5b5b39603acba4d7fdb1eddc0ca02))

This resolves #47


## v1.2.3 (2025-02-12)

### Bug Fixes

- **inference**: Fixed loading inference to basemodels
  ([`e7fb329`](https://github.com/xer4yx/lifftrackAPI/commit/e7fb3298158bf1e28ac058b09c19920c27bb9556))

- **object**: Fixed loading objects to basemodel
  ([`c3b77ae`](https://github.com/xer4yx/lifftrackAPI/commit/c3b77ae2cc24e0fc813a80cd6775203b86d7edb9))

### Refactoring

- **ws**: Migrated inference, and db operations
  ([`f14ef84`](https://github.com/xer4yx/lifftrackAPI/commit/f14ef849b232c1704b1ec17479ff8a6fa7bccfa6))


## v1.2.2 (2025-02-05)

### Features

- **http**: Added http endpoint for inference
  ([`dfb4604`](https://github.com/xer4yx/lifftrackAPI/commit/dfb4604f07403665d8a2c407e612f704cc0aa838))

BREAKING CHANGE: This feature may cause status code of 500

### Testing

- **ws**: Tested websocket connection
  ([`ea25fdb`](https://github.com/xer4yx/lifftrackAPI/commit/ea25fdba6c0211bf4ca45a1df1d39b4f0b7cc6af))

### Breaking Changes

- **http**: This feature may cause status code of 500


## v1.1.2 (2025-02-01)

### Features

- **ws**: Added socket disconnection safeguard
  ([`0c7e0a7`](https://github.com/xer4yx/lifftrackAPI/commit/0c7e0a745c6eeb2126ab1bc3a6172365039864f7))

Added a safeguard for socket disconnecting when the socket is disconnected abruptly while inference


## v1.0.2 (2025-01-31)

### Bug Fixes

- **progress**: Fixed exercise name query for progress endpoint
  ([`36d87d6`](https://github.com/xer4yx/lifftrackAPI/commit/36d87d681f98195f43ed5142265c71632c56ffc4))


## v1.0.1 (2025-01-30)

### Bug Fixes

- **router**: Fixed communication to client
  ([`3d166e3`](https://github.com/xer4yx/lifftrackAPI/commit/3d166e366646fd79c88ef11e258795d88ed1e868))


## v1.0.0 (2024-12-31)
