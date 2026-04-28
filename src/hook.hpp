#pragma once
#include "types.hpp"
#include <vector>

struct Hook {
    HookState state = HookState::Ready;
    Direction direction = Direction::None;
    Vec2 origin = {0, 0};       // where the hook was fired from
    Vec2 head = {0, 0};         // current hook tip position
    std::vector<Vec2> chain;    // tiles the chain occupies (from origin outward)
    int max_range = 10;
    PlayerId target_id = INVALID_PLAYER;
    int hooked_mine_id = -1;
    int cooldown = 0;           // ticks until can fire again
    int cooldown_max = 15;      // ~1 sec at 15Hz tick rate

    void fire(Direction dir, Vec2 from);
    Vec2 advance();             // move head 1 tile forward, append to chain, return new head
    void start_retract();
    bool retract_tick();        // pop chain tail, update head. returns true when chain empty
    void reset();               // back to Ready, start cooldown timer
    void tick_cooldown();       // decrement cooldown when Ready
};
