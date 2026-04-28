# AI Bot System

Console Wars ships with a built-in AI opponent that learns to play Pudge Wars against you. It starts out clueless — moving randomly, throwing hooks at nothing — but over hundreds of games it teaches itself to dodge mines, land hooks, and set up kill combos. No external libraries, no pre-trained models. The bot learns from scratch every time it plays.

---

## How to Use

Add the `--bot` or `--bots` flag when starting the server.

```bash
# Start a game with one AI bot
./console-wars 7777 --bot

# Start a game with three AI bots
./console-wars 7777 --bots 3

# No bots (default)
./console-wars 7777
```

Then connect as usual with `telnet localhost 7777`. The bots are already in the game as regular players — they move, hook, and place mines just like a human would.

---

## How It Works

The bot uses **Double DQN** (Deep Q-Network), a reinforcement learning algorithm. Here is the short version:

1. **Observe.** Every game tick (15 times per second), the bot reads the game state — where everyone is, what is on cooldown, where the mines are, whether a hook shot lines up.

2. **Decide.** A small neural network scores every possible action (move in four directions, hook in four directions, place a mine, or wait). The bot picks the highest-scoring action, unless it is exploring — early on, it intentionally tries random actions to discover what works.

3. **Learn.** After the tick resolves, the bot checks what happened. Did it get a kill? Walk into a mine? Land a hook? Each outcome carries a reward signal. The bot stores this experience (what it saw, what it did, what happened) in a replay buffer.

4. **Replay.** Periodically, the bot samples a batch of past experiences and trains its neural network to better predict which actions lead to good outcomes. This is the same core idea behind game-playing AIs like those that mastered Atari games.

5. **Two networks.** The "Double" in Double DQN means there are two copies of the neural network: an **online network** that makes decisions, and a **target network** that provides stable value estimates during training. The target network is a delayed copy of the online network, updated less frequently. This prevents the bot from chasing its own tail — overestimating the value of actions based on its own shifting predictions.

---

## What the Bot Sees

Each tick, the bot converts the raw game state into a **47-dimensional observation vector** — a compact numerical snapshot of everything relevant. All values are normalized to the range [-1, 1] or [0, 1].

**Self state (8 features)**
Position, alive/dead status, and whether movement, mine placement, and hook abilities are off cooldown. Also tracks whether the bot's own hook is currently extending or retracting.

**Nearest enemy (10 features)**
Relative position (dx, dy), alive/dead status, cooldown readiness, hook state, distance, and whether the enemy is currently being pulled by a hook.

**Own mines (9 features)**
Up to 3 nearest mines placed by the bot. For each: whether it exists and its relative position. Sorted by distance.

**Enemy mines (9 features)**
Same layout for up to 3 nearest mines placed by enemies.

**Directional danger (8 features)**
Four flags indicating whether an enemy mine is within 3 tiles in each cardinal direction. Four flags for wall adjacency — useful for avoiding getting cornered.

**Tactical signals (3 features)**
- Is an enemy lined up for a hook shot (same row or column, within range, no walls blocking)?
- Is an enemy near one of the bot's own mines (potential combo kill)?
- How many mines does the bot currently have deployed?

The "enemy in hook line" feature is especially important — it gives the network a direct signal for *"a hook would connect right now,"* which dramatically speeds up learning to hook.

---

## Reward System

The bot receives a reward signal after each tick. Rewards are clipped to [-1.0, +1.0].

| Event | Reward | What It Teaches |
|---|---|---|
| Kill an enemy | +1.0 | The ultimate goal |
| Die | -1.0 | Stay alive |
| Hook lands near own mine | +0.5 | Pull enemies into mines (the key combo) |
| Hook lands on enemy | +0.3 | Hooking is good even without a mine setup |
| Wasted hook (hits nothing) | -0.05 | Don't spam hooks blindly |
| Survive a tick | +0.001 | Mild incentive to stay alive between big events |
| Enter enemy mine danger zone | -0.02 | Steer clear of enemy mines |

The discount factor (gamma) is 0.97. This means the bot values a reward 30 ticks in the future at about 40% of its immediate value — long enough to credit a hook for a kill that happens after the pull completes, but not so long that every action gets vague credit for distant events.

---

## Learning Timeline

The bot's learning is gradual. Here is roughly what to expect:

**Games 1--20: Random exploration.**
The bot moves erratically, throws hooks in random directions, and places mines with no purpose. It is filling up its replay buffer with raw experience. Do not expect competence.

**Games 20--50: Basic avoidance.**
The bot starts learning that walking into enemy mines is bad and that staying alive is good. Movement becomes less random. It may begin drifting toward enemies rather than wandering aimlessly.

**Games 50--150: Learns to hook.**
The bot starts firing hooks when an enemy is lined up on the same row or column. Accuracy improves steadily. It begins to associate hooking with positive outcomes.

**Games 150--300: Combos emerge.**
The bot starts placing mines strategically and pulling enemies into them. This is the hook-into-mine combo — the core skill of Pudge Wars. The behavior is inconsistent at first but becomes more deliberate.

**Games 300--500: Competitive play.**
The bot plays at a level that challenges a practiced human. It dodges mines, lines up hook shots, and executes combos with some regularity. It still holds a 5% exploration rate, so it will occasionally do something odd — this is intentional and prevents it from getting stuck in predictable patterns.

---

## Persistence

The bot saves its learned state to **`data/bot_brain.bin`**. This file is written:

- At the end of every game round (a death-and-respawn cycle)
- On server shutdown (SIGINT / Ctrl+C)

When the server restarts with `--bot`, the bot loads its saved brain and picks up where it left off — same skill level, same exploration rate, same learning progress.

**To reset the bot's learning entirely**, delete the file:

```bash
rm data/bot_brain.bin
```

The next time the server starts with `--bot`, the bot will begin from scratch.

The replay buffer (past experiences) is *not* saved — it rebuilds naturally during play. The file itself is roughly 121 KB.

---

## Technical Details

This section is for developers who want to understand or modify the bot internals.

**Neural network architecture**

```
Input (47) --> Hidden (128, ReLU) --> Hidden (64, ReLU) --> Output (11, linear)
```

The 11 outputs correspond to the 11 possible actions: None, MoveUp, MoveDown, MoveLeft, MoveRight, HookUp, HookDown, HookLeft, HookRight, PlaceMine, and Quit (never selected by the bot). Total parameters: ~15,115 floats (~60 KB). Forward pass takes roughly 15 microseconds — well within the 66ms tick budget.

**Training**

| Parameter | Value |
|---|---|
| Replay buffer size | 50,000 transitions |
| Mini-batch size | 32 |
| Training frequency | Every 4 ticks |
| Warm-up period | 5,000 transitions (~1 game) before training begins |
| Target network sync | Every 1,000 training steps |
| Discount factor (gamma) | 0.97 |
| Learning rate | 0.0005, halved every 150 games, minimum 0.00005 |
| Exploration (epsilon) | 1.0 to 0.05 linearly over 200 games |
| Gradient clipping | Global norm 1.0 |
| Weight initialization | He normal (scale = sqrt(2 / fan_in)) |

**Action masking.** Invalid actions are excluded from both random and greedy selection. The bot will not try to fire a hook while it is on cooldown, place a mine when at the maximum count, or act while being pulled.

**Source files**

```
src/bot/
  bot_player.hpp/cpp       Orchestrator: owns a player ID and brain, runs the pre/post tick cycle
  bot_brain.hpp            Abstract interface (decide, learn, save, load)
  bot_observation.hpp/cpp  Feature extraction: game state to 47-float vector
  bot_reward.hpp/cpp       Reward computation from consecutive observations
  neural_net.hpp/cpp       Feedforward network with forward pass and backpropagation
  replay_buffer.hpp        Ring buffer of (state, action, reward, next state) transitions
  dqn_brain.hpp/cpp        Double DQN implementation wiring all components together
```
