#include "pudge.hpp"

Vec2 Pudge::desired_position(Direction dir) const {
    return pos + dir_to_vec(dir);
}

void Pudge::tick() {
    if (move_cooldown > 0) {
        --move_cooldown;
    }
    if (mine_cooldown > 0) {
        --mine_cooldown;
    }
    if (respawn_timer > 0) {
        --respawn_timer;
    }
    if (hook.state == HookState::Ready) {
        hook.tick_cooldown();
    }
}
