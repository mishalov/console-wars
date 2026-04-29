#include "hook.hpp"

void Hook::fire(Direction dir, Vec2 from) {
    state = HookState::Extending;
    direction = dir;
    origin = from;
    head = from;
    chain.clear();
    target_id = INVALID_PLAYER;
    hooked_mine_id = -1;
}

Vec2 Hook::advance() {
    head += dir_to_vec(direction);
    chain.push_back(head);
    return head;
}

void Hook::start_retract() {
    state = HookState::Retracting;
}

bool Hook::retract_tick() {
    if (chain.empty()) {
        return true;
    }
    chain.pop_back();
    head = chain.empty() ? origin : chain.back();
    return chain.empty();
}

void Hook::reset() {
    state = HookState::Ready;
    direction = Direction::None;
    chain.clear();
    target_id = INVALID_PLAYER;
    hooked_mine_id = -1;
    cooldown = cooldown_max;
    max_range = 10;
    speed = 1;
}

void Hook::tick_cooldown() {
    if (cooldown > 0) {
        --cooldown;
    }
}
