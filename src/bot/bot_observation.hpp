#pragma once
#include "../types.hpp"
#include <array>
#include <cstddef>

class GameState;

struct BotObservation {
    // -----------------------------------------------------------------------
    // Feature layout (92 dimensions):
    //   0- 7:  Self state (fractional cooldowns)
    //   8-17:  Nearest enemy
    //  18-27:  2nd nearest enemy (zeros if <2 enemies)
    //  28-36:  Own mines 3x(exists, rel_x, rel_y)
    //  37-45:  Enemy mines 3x(exists, rel_x, rel_y)
    //  46-49:  Directional danger (enemy mine within 3)
    //  50-53:  Wall adjacency
    //  54-57:  Hook ray-cast: dist to enemy in line (4 dirs)
    //  58-61:  Hook ray-cast: pull through own mine zone (4 dirs)
    //  62:     Hook line clear
    //  63-65:  Mine-enemy geometry: chebyshev, dx, dy
    //  66:     Mine count
    //  67-70:  Velocity: delta_self_x/y, delta_enemy_x/y
    //  71-73:  Hook head: rel_x, rel_y, remaining_range
    //  74:     Enemy count
    //  75-78:  Bonus: exists, rel_x, rel_y, manhattan_dist
    //  79-82:  Bonus type (one-hot: MultiHook, Immunity, SuperHook, MineField)
    //  83-86:  Self bonus: has_bonus, offensive, defensive, remaining
    //  87:     Nearest enemy immune
    //  88-91:  Has hookable target (enemy or mine-pull) per direction
    // -----------------------------------------------------------------------
    static constexpr int SIZE = 92;
    std::array<float, SIZE> features{};

    // Copies of key state for reward computation
    int kills = 0;
    int deaths = 0;
    int hooks_landed = 0;
    bool alive = true;
    bool hook_extending = false;
    bool hook_has_target = false;
    Vec2 pos{};

    // Bonus state for reward
    bool bonus_picked_up = false;
    bool has_active_bonus = false;

    // Previous-tick positions for velocity tracking (set by caller)
    Vec2 prev_self_pos{};
    Vec2 prev_enemy_pos{};
    PlayerId prev_enemy_id = INVALID_PLAYER;  // ID of nearest enemy last tick
    PlayerId nearest_enemy_id = INVALID_PLAYER;  // ID of nearest enemy this tick

    // Named feature indices for external consumers (e.g. reward function)
    static constexpr size_t kAlive               = 2;
    static constexpr size_t kDangerUpStart       = 46;
    static constexpr size_t kDangerRightEnd      = 49;
    static constexpr size_t kEnemyNearOwnMine    = 63;  // now chebyshev distance, not binary
    static constexpr size_t kMineCount           = 66;
    static constexpr size_t kHookLine            = 62;
    static constexpr size_t kHasTargetUp         = 88;
    static constexpr size_t kHasTargetDown       = 89;
    static constexpr size_t kHasTargetLeft       = 90;
    static constexpr size_t kHasTargetRight      = 91;
};

/// Extract a 92-dimensional feature vector from the game state for the given
/// bot player.  See bot_observation.cpp for the detailed feature layout.
BotObservation observe(const GameState& state, PlayerId bot_id,
                       Vec2 prev_self_pos = {0, 0}, Vec2 prev_enemy_pos = {0, 0},
                       PlayerId prev_enemy_id = INVALID_PLAYER);
