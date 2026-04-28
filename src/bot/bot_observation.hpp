#pragma once
#include "../types.hpp"
#include <array>
#include <cstddef>

class GameState;

struct BotObservation {
    static constexpr int SIZE = 47;
    std::array<float, SIZE> features{};

    // Copies of key state for reward computation
    int kills = 0;
    int deaths = 0;
    int hooks_landed = 0;
    bool alive = true;
    bool hook_extending = false;
    bool hook_has_target = false;
    Vec2 pos{};

    // Named feature indices for external consumers (e.g. reward function)
    static constexpr size_t kAlive               = 2;
    static constexpr size_t kDangerUpStart       = 36;
    static constexpr size_t kDangerRightEnd      = 39;
    static constexpr size_t kEnemyNearOwnMine    = 45;
};

/// Extract a 47-dimensional feature vector from the game state for the given
/// bot player.  See bot_observation.cpp for the detailed feature layout.
BotObservation observe(const GameState& state, PlayerId bot_id);
