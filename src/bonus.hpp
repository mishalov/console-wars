#pragma once
#include "types.hpp"

inline constexpr int BONUS_SPAWN_INTERVAL = 150;  // 10 seconds at 15Hz
inline constexpr int BONUS_DURATION = 90;          // 6 seconds at 15Hz
inline constexpr int RESPAWN_IMMUNITY_TICKS = 30;  // ~2s at 66ms ticks
inline constexpr int SUPER_HOOK_RANGE = 30;
inline constexpr int SUPER_HOOK_SPEED = 3;

enum class BonusType : uint8_t {
    MultiHook = 0,
    Immunity  = 1,
    SuperHook = 2,
    MineField = 3,
    Count     = 4,  // sentinel for random selection
};

// Ground pickup entity (one at a time on the map)
struct Bonus {
    BonusType type = BonusType::MultiHook;
    Vec2 pos = {0, 0};
    bool active = false;  // is there a bonus on the map?
};

// Active effect on a pudge
struct ActiveBonus {
    BonusType type = BonusType::MultiHook;
    int remaining = 0;  // 0 = no active bonus
    bool consumed = false;  // for MultiHook/SuperHook: used up on first fire
};
