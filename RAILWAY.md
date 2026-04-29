# Deploying Console Wars to Railway

Terminal-based multiplayer Pudge Wars game. Players connect via telnet; the server streams ANSI frames at 15Hz. Includes AI bots (Double DQN) that can optionally learn online during gameplay.

## Prerequisites

- Railway account ([railway.app](https://railway.app))
- GitHub repo connected to Railway

## Deployment Steps

1. **Create a new project** in Railway dashboard. Connect the GitHub repo.

2. **Add a persistent volume:**
   - Service Settings > Mounts
   - Mount path: `/app/data`
   - Persists the bot's trained neural network across deploys

3. **Configure networking:**
   - Settings > Networking
   - Add a **TCP Proxy** (not HTTP — this is a raw telnet server)
   - Railway assigns a public host + port (e.g., `roundhouse.proxy.rlwy.net:12345`)

4. **Set environment variables** (optional — defaults are in Dockerfile):

   | Variable | Description | Default |
   |----------|-------------|---------|
   | `BOT_COUNT` | Number of AI bots | `2` (set `0` for none) |
   | `NO_TRAIN` | Disable learning (inference-only) | `0` (learning enabled) |

   `PORT`, `BASE_DIR`, `DATA_DIR` are auto-configured. Don't change them.

5. **Deploy** — Railway auto-detects the Dockerfile and builds.

## Connecting

```
telnet <railway-tcp-host> <railway-tcp-port>
```

Example: `telnet roundhouse.proxy.rlwy.net 12345`

Host and port are shown in the Railway Networking tab after enabling TCP proxy.

## Deployment Modes

| Mode | Config | Behavior |
|------|--------|----------|
| With learning | `BOT_COUNT=2`, `NO_TRAIN=0` | Bots play and improve over time. Brain persisted to volume. |
| Without learning | `BOT_COUNT=2`, `NO_TRAIN=1` | Bots use pre-trained brain, no writes to disk. Volume optional. |
| No bots | `BOT_COUNT=0` | Human-only server. No volume needed. |

## Persistence

- Bot brain (~1.5MB) saved to `/app/data/bot_brain.bin` on the Railway volume.
- Saves automatically on bot death and graceful shutdown (SIGTERM).
- On first deploy (empty volume), the pre-trained brain bundled in the Docker image seeds the file.
- To reset training: delete the file from the volume and redeploy.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Can't connect | Enable TCP proxy (not HTTP) in Railway networking |
| Bots not learning | Check that `NO_TRAIN` is not set to `1` |
| Brain lost after redeploy | Ensure persistent volume is mounted at `/app/data` |
| Build fails | Requires Alpine 3.19+ with g++, cmake, make (handled by Dockerfile) |

## Local Docker Testing

```sh
docker build -t console-wars .
docker run -p 7777:7777 -e PORT=7777 -v $(pwd)/data:/app/data console-wars
telnet localhost 7777
```
