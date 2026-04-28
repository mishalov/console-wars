#pragma once
#include "types.hpp"
#include "hook.hpp"
#include "score.hpp"

struct Pudge {
    PlayerId id = INVALID_PLAYER;
    Vec2 pos;
    Direction facing = Direction::Right;
    int move_cooldown = 0;
    int move_cooldown_max = 2;  // ticks between moves (~133ms at 15Hz)
    bool alive = true;
    bool being_pulled = false;  // for Phase 1d hook
    Hook hook;
    Score score;
    int mine_cooldown = 0;
    int mine_cooldown_max = 30;
    int respawn_timer = -1;     // -1 = not respawning; counts down to 0

    Vec2 desired_position(Direction dir) const;
    void tick();
};
