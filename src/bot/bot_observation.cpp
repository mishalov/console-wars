#include "bot_observation.hpp"
#include "../game_state.hpp"
#include "../bonus.hpp"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int manhattan(Vec2 a, Vec2 b) {
    return std::abs(a.x - b.x) + std::abs(a.y - b.y);
}

static int chebyshev(Vec2 a, Vec2 b) {
    return std::max(std::abs(a.x - b.x), std::abs(a.y - b.y));
}

/// Return a pointer to the nearest alive enemy, or nullptr.
static const Pudge* nearest_alive_enemy(const GameState& state, PlayerId self_id, Vec2 self_pos) {
    const Pudge* best = nullptr;
    int best_dist = INT_MAX;
    for (const auto& p : state.pudges()) {
        if (p.id == self_id || !p.alive) continue;
        int d = manhattan(self_pos, p.pos);
        if (d < best_dist) {
            best_dist = d;
            best = &p;
        }
    }
    return best;
}

/// Return a pointer to the 2nd nearest alive enemy, or nullptr.
static const Pudge* second_nearest_alive_enemy(const GameState& state, PlayerId self_id,
                                                Vec2 self_pos, PlayerId exclude_id) {
    const Pudge* best = nullptr;
    int best_dist = INT_MAX;
    for (const auto& p : state.pudges()) {
        if (p.id == self_id || p.id == exclude_id || !p.alive) continue;
        int d = manhattan(self_pos, p.pos);
        if (d < best_dist) {
            best_dist = d;
            best = &p;
        }
    }
    return best;
}

/// Check whether there is a clear line of sight along a single axis
/// (row or column) between two positions, within a maximum distance.
static bool clear_line(const GameState& state, Vec2 from, Vec2 to, int max_dist) {
    if (from.x == to.x) {
        // Same column — walk vertically
        int dist = std::abs(to.y - from.y);
        if (dist > max_dist) return false;
        int step = (to.y > from.y) ? 1 : -1;
        for (int y = from.y + step; y != to.y; y += step) {
            if (state.tile_at(from.x, y) == TileType::Wall) return false;
        }
        return true;
    }
    if (from.y == to.y) {
        // Same row — walk horizontally
        int dist = std::abs(to.x - from.x);
        if (dist > max_dist) return false;
        int step = (to.x > from.x) ? 1 : -1;
        for (int x = from.x + step; x != to.x; x += step) {
            if (state.tile_at(x, from.y) == TileType::Wall) return false;
        }
        return true;
    }
    return false;  // not on same row or column
}

// ---------------------------------------------------------------------------
// observe()
// ---------------------------------------------------------------------------

BotObservation observe(const GameState& state, PlayerId bot_id,
                       Vec2 prev_self_pos, Vec2 prev_enemy_pos,
                       PlayerId prev_enemy_id) {
    BotObservation obs{};

    // Feature normalization assumes the default 40x20 map
    assert(state.width() == 40 && state.height() == 20 &&
           "bot observation assumes 40x20 map");

    const Pudge* self = nullptr;
    for (const auto& p : state.pudges()) {
        if (p.id == bot_id) { self = &p; break; }
    }
    if (!self) return obs;  // bot not found — return zeroed observation

    auto& f = obs.features;

    // Store previous positions in the observation for callers
    obs.prev_self_pos  = prev_self_pos;
    obs.prev_enemy_pos = prev_enemy_pos;
    obs.prev_enemy_id  = prev_enemy_id;

    // -----------------------------------------------------------------------
    // Snapshot copies for reward computation
    // -----------------------------------------------------------------------
    obs.kills        = self->score.kills;
    obs.deaths       = self->score.deaths;
    obs.hooks_landed = self->score.hooks_landed;
    obs.alive        = self->alive;
    obs.hook_extending  = (self->hook.state == HookState::Extending);
    obs.hook_has_target = (self->hook.target_id != INVALID_PLAYER);
    obs.pos          = self->pos;
    obs.bonus_picked_up = self->picked_up_bonus;

    // -----------------------------------------------------------------------
    // Index 0-7: Self state (fractional cooldowns)
    // -----------------------------------------------------------------------
    f[0] = static_cast<float>(self->pos.x) / 39.0f;
    f[1] = static_cast<float>(self->pos.y) / 19.0f;
    f[2] = self->alive ? 1.0f : 0.0f;
    f[3] = 1.0f - static_cast<float>(self->move_cooldown) / static_cast<float>(std::max(1, self->move_cooldown_max));
    f[4] = 1.0f - static_cast<float>(self->mine_cooldown) / static_cast<float>(std::max(1, self->mine_cooldown_max));
    f[5] = (self->hook.state == HookState::Ready)
               ? (1.0f - static_cast<float>(self->hook.cooldown) / static_cast<float>(std::max(1, self->hook.cooldown_max)))
               : 0.0f;
    f[6] = (self->hook.state == HookState::Extending)  ? 1.0f : 0.0f;
    f[7] = (self->hook.state == HookState::Retracting) ? 1.0f : 0.0f;

    // -----------------------------------------------------------------------
    // Index 8-17: Nearest alive enemy (relative to self)
    // -----------------------------------------------------------------------
    const Pudge* enemy = nearest_alive_enemy(state, bot_id, self->pos);

    // Track nearest enemy ID for velocity feature correctness
    obs.nearest_enemy_id = enemy ? enemy->id : INVALID_PLAYER;

    auto encode_enemy = [&](const Pudge* e, std::size_t base) {
        if (!e) return;
        f[base + 0u] = static_cast<float>(e->pos.x - self->pos.x) / 39.0f;
        f[base + 1u] = static_cast<float>(e->pos.y - self->pos.y) / 19.0f;
        f[base + 2u] = e->alive ? 1.0f : 0.0f;
        f[base + 3u] = 1.0f - static_cast<float>(e->move_cooldown) / static_cast<float>(std::max(1, e->move_cooldown_max));
        f[base + 4u] = 1.0f - static_cast<float>(e->mine_cooldown) / static_cast<float>(std::max(1, e->mine_cooldown_max));
        f[base + 5u] = (e->hook.state == HookState::Ready)
                          ? (1.0f - static_cast<float>(e->hook.cooldown) / static_cast<float>(std::max(1, e->hook.cooldown_max)))
                          : 0.0f;
        f[base + 6u] = (e->hook.state == HookState::Extending)  ? 1.0f : 0.0f;
        f[base + 7u] = (e->hook.state == HookState::Retracting) ? 1.0f : 0.0f;
        f[base + 8u] = static_cast<float>(manhattan(self->pos, e->pos)) / 58.0f;
        f[base + 9u] = e->being_pulled ? 1.0f : 0.0f;
    };

    encode_enemy(enemy, 8);

    // -----------------------------------------------------------------------
    // Index 18-27: 2nd nearest alive enemy
    // -----------------------------------------------------------------------
    const Pudge* enemy2 = enemy
        ? second_nearest_alive_enemy(state, bot_id, self->pos, enemy->id)
        : nullptr;
    encode_enemy(enemy2, 18);

    // -----------------------------------------------------------------------
    // Index 28-36: Own mines (3 slots, sorted by Chebyshev distance to self)
    // Index 37-45: Enemy mines (3 slots, sorted by Chebyshev distance to self)
    // -----------------------------------------------------------------------
    struct MineDist {
        const Mine* mine;
        int dist;
    };
    auto fill_mine_slots = [&](int base_index, bool own) {
        std::vector<MineDist> candidates;
        for (const auto& m : state.mines()) {
            if (!m.active) continue;
            bool is_own = (m.owner_id == bot_id);
            if (is_own != own) continue;
            candidates.push_back({&m, chebyshev(self->pos, m.pos)});
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const MineDist& a, const MineDist& b) { return a.dist < b.dist; });

        for (size_t i = 0; i < 3 && i < candidates.size(); ++i) {
            auto idx = static_cast<size_t>(base_index) + i * 3;
            f[idx + 0] = 1.0f;  // exists
            f[idx + 1] = static_cast<float>(candidates[i].mine->pos.x - self->pos.x) / 39.0f;
            f[idx + 2] = static_cast<float>(candidates[i].mine->pos.y - self->pos.y) / 19.0f;
        }
    };

    fill_mine_slots(28, true);   // own mines
    fill_mine_slots(37, false);  // enemy mines

    // -----------------------------------------------------------------------
    // Index 46-53: Directional danger / wall signals
    // -----------------------------------------------------------------------
    // Directions in canonical order: Up, Down, Left, Right
    static constexpr Direction dirs[4] = {
        Direction::Up, Direction::Down, Direction::Left, Direction::Right
    };

    for (int d = 0; d < 4; ++d) {
        Vec2 step = dir_to_vec(dirs[d]);

        // 46-49: enemy mine within 3 tiles in this direction
        for (int dist = 1; dist <= 3; ++dist) {
            Vec2 check = {self->pos.x + step.x * dist, self->pos.y + step.y * dist};
            // Stop scanning if we hit a wall or go out of bounds
            if (check.x < 0 || check.x >= state.width() ||
                check.y < 0 || check.y >= state.height()) break;
            if (state.tile_at(check.x, check.y) == TileType::Wall) break;

            for (const auto& m : state.mines()) {
                if (m.active && m.owner_id != bot_id && m.pos == check) {
                    f[static_cast<size_t>(46 + d)] = 1.0f;
                    goto next_danger_dir;  // found one, stop scanning this direction
                }
            }
        }
        next_danger_dir:

        // 50-53: wall adjacent (1 tile in this direction)
        {
            Vec2 adj = self->pos + step;
            if (adj.x < 0 || adj.x >= state.width() ||
                adj.y < 0 || adj.y >= state.height() ||
                state.tile_at(adj.x, adj.y) == TileType::Wall) {
                f[static_cast<size_t>(50 + d)] = 1.0f;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Index 54-57: Hook direction ray-cast — dist to enemy in line (4 dirs)
    // Index 58-61: Hook direction ray-cast — pull through own mine zone (4 dirs)
    // -----------------------------------------------------------------------
    for (int d = 0; d < 4; ++d) {
        Vec2 step = dir_to_vec(dirs[d]);
        float enemy_dist_norm = 0.0f;
        float through_mine = 0.0f;

        for (int dist = 1; dist <= 10; ++dist) {
            Vec2 tile = {self->pos.x + step.x * dist, self->pos.y + step.y * dist};

            // Stop on out-of-bounds or wall
            if (tile.x < 0 || tile.x >= state.width() ||
                tile.y < 0 || tile.y >= state.height()) break;
            if (state.tile_at(tile.x, tile.y) == TileType::Wall) break;

            // 54-57: Check if an alive enemy is on this tile
            if (enemy_dist_norm == 0.0f) {
                for (const auto& p : state.pudges()) {
                    if (p.id == bot_id || !p.alive) continue;
                    if (p.pos == tile) {
                        enemy_dist_norm = static_cast<float>(dist) / 10.0f;
                        break;
                    }
                }
            }

            // 58-61: Check if this tile on the pull path is within Chebyshev
            // distance 1 of any own mine (i.e., pulling through mine zone)
            if (through_mine == 0.0f) {
                for (const auto& m : state.mines()) {
                    if (!m.active || m.owner_id != bot_id) continue;
                    if (chebyshev(tile, m.pos) <= 1) {
                        through_mine = 1.0f;
                        break;
                    }
                }
            }
        }

        f[static_cast<size_t>(54 + d)] = enemy_dist_norm;
        f[static_cast<size_t>(58 + d)] = through_mine;
    }

    // -----------------------------------------------------------------------
    // Index 88-91: Has hookable target per direction (binary)
    //   1.0 if enemy in ray OR mine-pull opportunity exists in that direction
    // -----------------------------------------------------------------------
    for (int d = 0; d < 4; ++d) {
        f[static_cast<size_t>(88 + d)] = (f[static_cast<size_t>(54 + d)] > 0.0f || f[static_cast<size_t>(58 + d)] > 0.0f) ? 1.0f : 0.0f;
    }

    // -----------------------------------------------------------------------
    // Index 62: Hook line clear — nearest enemy on same axis, within 10, no walls
    // -----------------------------------------------------------------------
    if (enemy) {
        f[62] = clear_line(state, self->pos, enemy->pos, 10) ? 1.0f : 0.0f;
    }

    // -----------------------------------------------------------------------
    // Index 63-65: Mine-enemy geometry
    //   63: min chebyshev(own_mine, nearest_enemy) / 10.0
    //   64: mine-to-enemy dx / 39.0
    //   65: mine-to-enemy dy / 19.0
    // -----------------------------------------------------------------------
    if (enemy) {
        int best_cheby = INT_MAX;
        const Mine* closest_mine = nullptr;
        for (const auto& m : state.mines()) {
            if (!m.active || m.owner_id != bot_id) continue;
            int cd = chebyshev(enemy->pos, m.pos);
            if (cd < best_cheby) {
                best_cheby = cd;
                closest_mine = &m;
            }
        }
        if (closest_mine) {
            f[63] = static_cast<float>(best_cheby) / 10.0f;
            f[64] = static_cast<float>(enemy->pos.x - closest_mine->pos.x) / 39.0f;
            f[65] = static_cast<float>(enemy->pos.y - closest_mine->pos.y) / 19.0f;
        }
    }

    // -----------------------------------------------------------------------
    // Index 66: Mine count
    // -----------------------------------------------------------------------
    f[66] = std::min(1.0f, static_cast<float>(state.mines_owned_by(bot_id)) / 3.0f);

    // -----------------------------------------------------------------------
    // Index 67-70: Velocity features (delta positions from previous tick)
    //   Enemy velocity only computed when the same enemy is nearest as last tick.
    // -----------------------------------------------------------------------
    f[67] = static_cast<float>(self->pos.x - prev_self_pos.x) / 39.0f;
    f[68] = static_cast<float>(self->pos.y - prev_self_pos.y) / 19.0f;
    bool same_enemy = enemy && (enemy->id == prev_enemy_id);
    f[69] = same_enemy ? static_cast<float>(enemy->pos.x - prev_enemy_pos.x) / 39.0f : 0.0f;
    f[70] = same_enemy ? static_cast<float>(enemy->pos.y - prev_enemy_pos.y) / 19.0f : 0.0f;

    // -----------------------------------------------------------------------
    // Index 71-73: Hook head position (when extending)
    // -----------------------------------------------------------------------
    if (self->hook.state == HookState::Extending) {
        f[71] = static_cast<float>(self->hook.head.x - self->pos.x) / 10.0f;
        f[72] = static_cast<float>(self->hook.head.y - self->pos.y) / 10.0f;
        f[73] = 1.0f - static_cast<float>(self->hook.chain.size()) / static_cast<float>(self->hook.max_range);
    }

    // -----------------------------------------------------------------------
    // Index 74: Enemy count (alive enemies / max possible enemies)
    // -----------------------------------------------------------------------
    int alive_enemies = 0;
    for (const auto& p : state.pudges()) {
        if (p.id != bot_id && p.alive) ++alive_enemies;
    }
    f[74] = static_cast<float>(alive_enemies) / 15.0f;  // max 16 players - 1

    // -----------------------------------------------------------------------
    // Index 75-82: Ground bonus info
    // -----------------------------------------------------------------------
    const auto& bonus = state.bonus();
    if (bonus.active) {
        f[75] = 1.0f;
        f[76] = static_cast<float>(bonus.pos.x - self->pos.x) / 39.0f;
        f[77] = static_cast<float>(bonus.pos.y - self->pos.y) / 19.0f;
        f[78] = static_cast<float>(manhattan(self->pos, bonus.pos)) / 58.0f;
        auto type_idx = static_cast<size_t>(bonus.type);
        if (type_idx < 4) {
            f[79 + type_idx] = 1.0f;
        }
    }

    // -----------------------------------------------------------------------
    // Index 83-86: Self bonus state
    // -----------------------------------------------------------------------
    if (self->active_bonus.remaining > 0) {
        f[83] = 1.0f;
        bool offensive = (self->active_bonus.type == BonusType::MultiHook ||
                          self->active_bonus.type == BonusType::SuperHook);
        f[84] = offensive ? 1.0f : 0.0f;
        f[85] = (self->active_bonus.type == BonusType::Immunity) ? 1.0f : 0.0f;
        f[86] = static_cast<float>(self->active_bonus.remaining) / static_cast<float>(BONUS_DURATION);
        obs.has_active_bonus = true;
    }

    // -----------------------------------------------------------------------
    // Index 87: Nearest enemy immune
    // -----------------------------------------------------------------------
    if (enemy && enemy->active_bonus.type == BonusType::Immunity &&
        enemy->active_bonus.remaining > 0) {
        f[87] = 1.0f;
    }

    return obs;
}
