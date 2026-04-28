#include "bot_observation.hpp"
#include "../game_state.hpp"

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

BotObservation observe(const GameState& state, PlayerId bot_id) {
    BotObservation obs{};

    // Feature normalization assumes the default 40x20 map
    assert(state.width() == 40 && state.height() == 20 &&
           "bot observation assumes 40x20 map");

    const Pudge* self = nullptr;
    // GameState::get_pudge is non-const in the interface, so we iterate the
    // const vector instead.
    for (const auto& p : state.pudges()) {
        if (p.id == bot_id) { self = &p; break; }
    }
    if (!self) return obs;  // bot not found — return zeroed observation

    auto& f = obs.features;

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

    // -----------------------------------------------------------------------
    // Index 0-7: Self state
    // -----------------------------------------------------------------------
    f[0] = static_cast<float>(self->pos.x) / 39.0f;
    f[1] = static_cast<float>(self->pos.y) / 19.0f;
    f[2] = self->alive ? 1.0f : 0.0f;
    f[3] = (self->move_cooldown == 0) ? 1.0f : 0.0f;
    f[4] = (self->mine_cooldown == 0) ? 1.0f : 0.0f;
    f[5] = (self->hook.cooldown == 0 && self->hook.state == HookState::Ready) ? 1.0f : 0.0f;
    f[6] = (self->hook.state == HookState::Extending)  ? 1.0f : 0.0f;
    f[7] = (self->hook.state == HookState::Retracting) ? 1.0f : 0.0f;

    // -----------------------------------------------------------------------
    // Index 8-17: Nearest alive enemy (relative to self)
    // -----------------------------------------------------------------------
    const Pudge* enemy = nearest_alive_enemy(state, bot_id, self->pos);
    if (enemy) {
        f[8]  = static_cast<float>(enemy->pos.x - self->pos.x) / 39.0f;
        f[9]  = static_cast<float>(enemy->pos.y - self->pos.y) / 19.0f;
        f[10] = enemy->alive ? 1.0f : 0.0f;
        f[11] = (enemy->move_cooldown == 0) ? 1.0f : 0.0f;
        f[12] = (enemy->mine_cooldown == 0) ? 1.0f : 0.0f;
        f[13] = (enemy->hook.cooldown == 0 && enemy->hook.state == HookState::Ready) ? 1.0f : 0.0f;
        f[14] = (enemy->hook.state == HookState::Extending)  ? 1.0f : 0.0f;
        f[15] = (enemy->hook.state == HookState::Retracting) ? 1.0f : 0.0f;
        f[16] = static_cast<float>(manhattan(self->pos, enemy->pos)) / 58.0f;
        f[17] = enemy->being_pulled ? 1.0f : 0.0f;
    }
    // else: f[8..17] remain 0.0 (default)

    // -----------------------------------------------------------------------
    // Index 18-26: Own mines (3 slots, sorted by distance to self)
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
            candidates.push_back({&m, manhattan(self->pos, m.pos)});
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

    fill_mine_slots(18, true);   // own mines
    fill_mine_slots(27, false);  // enemy mines

    // -----------------------------------------------------------------------
    // Index 36-43: Directional danger / wall signals
    // -----------------------------------------------------------------------
    // Directions in canonical order: Up, Down, Left, Right
    static constexpr Direction dirs[4] = {
        Direction::Up, Direction::Down, Direction::Left, Direction::Right
    };

    for (int d = 0; d < 4; ++d) {
        Vec2 step = dir_to_vec(dirs[d]);

        // 36-39: enemy mine within 3 tiles in this direction
        for (int dist = 1; dist <= 3; ++dist) {
            Vec2 check = {self->pos.x + step.x * dist, self->pos.y + step.y * dist};
            // Stop scanning if we hit a wall or go out of bounds
            if (check.x < 0 || check.x >= state.width() ||
                check.y < 0 || check.y >= state.height()) break;
            if (state.tile_at(check.x, check.y) == TileType::Wall) break;

            for (const auto& m : state.mines()) {
                if (m.active && m.owner_id != bot_id && m.pos == check) {
                    f[static_cast<size_t>(36 + d)] = 1.0f;
                    goto next_danger_dir;  // found one, stop scanning this direction
                }
            }
        }
        next_danger_dir:

        // 40-43: wall adjacent (1 tile in this direction)
        {
            Vec2 adj = self->pos + step;
            if (adj.x < 0 || adj.x >= state.width() ||
                adj.y < 0 || adj.y >= state.height() ||
                state.tile_at(adj.x, adj.y) == TileType::Wall) {
                f[static_cast<size_t>(40 + d)] = 1.0f;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Index 44-46: Tactical features
    // -----------------------------------------------------------------------

    // 44: enemy_in_hook_line — nearest enemy on same row or column, within
    //     10 tiles, with no walls between.
    if (enemy) {
        f[44] = clear_line(state, self->pos, enemy->pos, 10) ? 1.0f : 0.0f;
    }

    // 45: enemy_near_own_mine — nearest enemy within Chebyshev distance 3
    //     of any of bot's mines.
    if (enemy) {
        for (const auto& m : state.mines()) {
            if (m.active && m.owner_id == bot_id &&
                chebyshev(enemy->pos, m.pos) <= 3) {
                f[45] = 1.0f;
                break;
            }
        }
    }

    // 46: mines_owned_by(bot_id) / 3.0
    f[46] = static_cast<float>(state.mines_owned_by(bot_id)) / 3.0f;

    return obs;
}
