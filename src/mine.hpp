#pragma once
#include "types.hpp"

inline constexpr int MAX_MINES_PER_PLAYER = 3;
inline constexpr int MINE_PLACE_COOLDOWN = 30;
inline constexpr int MINE_EXPLOSION_TICKS = 3;

struct Mine {
    int id = -1;
    PlayerId owner_id = INVALID_PLAYER;
    Vec2 pos = {0, 0};
    int explosion_timer = -1;   // -1 means not exploding; counts down from MINE_EXPLOSION_TICKS
    bool active = true;
    bool being_hooked = false;
    PlayerId hooked_by = INVALID_PLAYER;  // pudge currently dragging this mine via hook
};
