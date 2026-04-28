#include "game_state.hpp"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>

static constexpr Vec2 kSpawnPoints[] = {
    {5, 5}, {34, 5}, {5, 14}, {34, 14},
    {10, 10}, {29, 10}, {20, 3}, {20, 16}
};
static constexpr int kNumSpawnPoints = 8;

bool GameState::load_map(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open map: " << path << "\n";
        return false;
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        // Strip \r if present
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) continue;
        lines.push_back(line);
    }

    if (lines.empty()) {
        std::cerr << "Map file is empty\n";
        return false;
    }

    width_ = static_cast<int>(lines[0].size());
    height_ = static_cast<int>(lines.size());

    grid_.resize(static_cast<size_t>(width_ * height_), TileType::Empty);

    for (int y = 0; y < height_; ++y) {
        if (static_cast<int>(lines[static_cast<size_t>(y)].size()) != width_) {
            std::cerr << "Map line " << y << " has inconsistent width\n";
            return false;
        }
        for (int x = 0; x < width_; ++x) {
            char c = lines[static_cast<size_t>(y)][static_cast<size_t>(x)];
            grid_[static_cast<size_t>(y * width_ + x)] =
                (c == '#') ? TileType::Wall : TileType::Empty;
        }
    }

    std::cout << "Loaded map: " << width_ << "x" << height_ << "\n";
    return true;
}

TileType GameState::tile_at(int x, int y) const {
    if (x < 0 || x >= width_ || y < 0 || y >= height_) {
        return TileType::Wall;
    }
    return grid_[static_cast<size_t>(y * width_ + x)];
}

Vec2 GameState::next_spawn_point() {
    // Try to find an unoccupied spawn point away from mines
    for (int i = 0; i < kNumSpawnPoints; ++i) {
        Vec2 sp = kSpawnPoints[(next_id_ + i) % kNumSpawnPoints];
        if (!is_walkable(sp) || is_occupied(sp)) continue;

        // Check no active mine within Chebyshev distance 1
        bool mine_nearby = false;
        for (const auto& m : mines_) {
            if (!m.active || m.explosion_timer >= 0) continue;
            int dx = std::abs(sp.x - m.pos.x);
            int dy = std::abs(sp.y - m.pos.y);
            if (std::max(dx, dy) <= 1) {
                mine_nearby = true;
                break;
            }
        }
        if (mine_nearby) continue;

        return sp;
    }
    // Fallback: all occupied, just cycle
    return kSpawnPoints[next_id_ % kNumSpawnPoints];
}

PlayerId GameState::add_pudge(Vec2 spawn_pos) {
    Pudge p;
    p.id = next_id_++;
    p.pos = spawn_pos;
    p.facing = Direction::Right;
    p.alive = true;
    pudges_.push_back(p);
    std::cout << "Pudge " << p.id << " spawned at (" << spawn_pos.x << "," << spawn_pos.y << ")\n";
    return p.id;
}

void GameState::remove_pudge(PlayerId id) {
    pudges_.erase(
        std::remove_if(pudges_.begin(), pudges_.end(),
            [id](const Pudge& p) { return p.id == id; }),
        pudges_.end()
    );
}

Pudge* GameState::get_pudge(PlayerId id) {
    for (auto& p : pudges_) {
        if (p.id == id) return &p;
    }
    return nullptr;
}

const std::vector<Pudge>& GameState::pudges() const {
    return pudges_;
}

void GameState::handle_input(PlayerId id, InputAction action) {
    Pudge* pudge = get_pudge(id);
    if (!pudge || !pudge->alive) return;

    // PlaceMine action (space)
    if (action == InputAction::PlaceMine) {
        if (pudge->mine_cooldown > 0) return;
        if (mines_owned_by(pudge->id) >= MAX_MINES_PER_PLAYER) return;
        place_mine(pudge->id, pudge->pos);
        pudge->mine_cooldown = pudge->mine_cooldown_max;
        return;
    }

    // Hook actions (IJKL)
    Direction hook_dir = Direction::None;
    switch (action) {
        case InputAction::HookUp:    hook_dir = Direction::Up; break;
        case InputAction::HookDown:  hook_dir = Direction::Down; break;
        case InputAction::HookLeft:  hook_dir = Direction::Left; break;
        case InputAction::HookRight: hook_dir = Direction::Right; break;
        default: break;
    }

    if (hook_dir != Direction::None) {
        if (pudge->hook.state == HookState::Ready &&
            pudge->hook.cooldown == 0 &&
            !pudge->being_pulled) {
            pudge->hook.fire(hook_dir, pudge->pos);
        }
        return;
    }

    // Movement actions (WASD / arrows)
    Direction dir = Direction::None;
    switch (action) {
        case InputAction::MoveUp:    dir = Direction::Up; break;
        case InputAction::MoveDown:  dir = Direction::Down; break;
        case InputAction::MoveLeft:  dir = Direction::Left; break;
        case InputAction::MoveRight: dir = Direction::Right; break;
        default: return;
    }

    // Always update facing regardless of whether move succeeds
    pudge->facing = dir;

    // Can't move while hook is active or being pulled
    if (pudge->hook.state != HookState::Ready) return;
    if (pudge->being_pulled) return;

    // Cooldown check
    if (pudge->move_cooldown > 0) return;

    Vec2 target = pudge->desired_position(dir);

    if (!is_walkable(target)) return;
    if (is_occupied(target, pudge->id)) return;

    pudge->pos = target;
    pudge->move_cooldown = pudge->move_cooldown_max;
}

void GameState::tick() {
    ++tick_count_;
    for (auto& p : pudges_) {
        p.tick();
    }

    // Hook state machine for each pudge
    for (auto& p : pudges_) {
        if (!p.alive) continue;

        switch (p.hook.state) {
            case HookState::Extending: {
                Vec2 new_head = p.hook.advance();

                if (!is_walkable(new_head)) {
                    // Hit a wall: remove the invalid position from chain
                    p.hook.chain.pop_back();
                    p.hook.head = p.hook.chain.empty() ? p.hook.origin : p.hook.chain.back();
                    p.hook.start_retract();
                } else {
                    // Check if hook hit another pudge
                    bool hit_target = false;
                    for (auto& other : pudges_) {
                        if (other.id == p.id) continue;
                        if (!other.alive) continue;
                        if (other.being_pulled) continue;
                        if (other.pos == new_head) {
                            p.hook.target_id = other.id;
                            other.being_pulled = true;
                            p.hook.start_retract();
                            p.score.hooks_landed++;
                            hit_target = true;
                            break;
                        }
                    }

                    // Check if hook hit a mine (only if didn't hit a pudge)
                    if (!hit_target && p.hook.state == HookState::Extending) {
                        for (auto& mine : mines_) {
                            if (!mine.active) continue;
                            if (mine.explosion_timer >= 0) continue;  // already exploding
                            if (mine.being_hooked) continue;
                            if (mine.pos == new_head) {
                                p.hook.hooked_mine_id = mine.id;
                                mine.being_hooked = true;
                                p.hook.start_retract();
                                hit_target = true;
                                break;
                            }
                        }
                    }

                    // Check max range (only if still extending)
                    if (!hit_target && p.hook.state == HookState::Extending) {
                        if (static_cast<int>(p.hook.chain.size()) >= p.hook.max_range) {
                            p.hook.start_retract();
                        }
                    }
                }
                break;
            }

            case HookState::Retracting: {
                bool done = p.hook.retract_tick();

                // Handle hooked pudge
                if (p.hook.target_id != INVALID_PLAYER) {
                    Pudge* target = get_pudge(p.hook.target_id);
                    if (target && target->alive) {
                        if (done) {
                            // Place target adjacent to caster in hook direction
                            Vec2 adj = p.hook.origin + dir_to_vec(p.hook.direction);
                            if (is_walkable(adj) && !is_occupied(adj, target->id)) {
                                target->pos = adj;
                            } else if (!is_occupied(p.hook.origin, target->id)) {
                                target->pos = p.hook.origin;  // stack on caster as last resort
                            }
                        } else {
                            target->pos = p.hook.head;
                        }
                    }
                }

                // Handle hooked mine
                if (p.hook.hooked_mine_id >= 0) {
                    Mine* hooked_mine = nullptr;
                    for (auto& m : mines_) {
                        if (m.id == p.hook.hooked_mine_id) {
                            hooked_mine = &m;
                            break;
                        }
                    }
                    if (hooked_mine && hooked_mine->active) {
                        if (done) {
                            // Place mine adjacent to caster
                            Vec2 adj = p.hook.origin + dir_to_vec(p.hook.direction);
                            if (is_walkable(adj)) {
                                hooked_mine->pos = adj;
                            } else {
                                hooked_mine->pos = p.hook.origin;
                            }
                            // Transfer ownership
                            hooked_mine->owner_id = p.id;
                            hooked_mine->being_hooked = false;
                        } else {
                            hooked_mine->pos = p.hook.head;
                        }
                    }
                }

                if (done) {
                    // Clear being_pulled on target
                    if (p.hook.target_id != INVALID_PLAYER) {
                        Pudge* target = get_pudge(p.hook.target_id);
                        if (target) {
                            target->being_pulled = false;
                        }
                    }
                    p.hook.reset();
                }
                break;
            }

            case HookState::Ready:
                // Cooldown is handled in Pudge::tick()
                break;
        }
    }

    // Mine proximity checks and explosions
    check_mine_proximity();
    update_mine_explosions();

    // Respawn check
    for (auto& p : pudges_) {
        if (!p.alive && p.respawn_timer == 0) {
            respawn_pudge(p.id);
        }
    }
}

uint32_t GameState::tick_count() const {
    return tick_count_;
}

bool GameState::is_walkable(Vec2 pos) const {
    return tile_at(pos.x, pos.y) == TileType::Empty;
}

bool GameState::is_occupied(Vec2 pos, PlayerId exclude) const {
    for (const auto& p : pudges_) {
        if (p.id == exclude) continue;
        if (!p.alive) continue;
        if (p.pos == pos) return true;
    }
    return false;
}

// --- Mine management ---

void GameState::place_mine(PlayerId owner_id, Vec2 pos) {
    Mine m;
    m.id = next_mine_id_++;
    m.owner_id = owner_id;
    m.pos = pos;
    m.active = true;
    m.being_hooked = false;
    m.explosion_timer = -1;
    mines_.push_back(m);
}

const std::vector<Mine>& GameState::mines() const {
    return mines_;
}

int GameState::mines_owned_by(PlayerId owner_id) const {
    int count = 0;
    for (const auto& m : mines_) {
        if (m.active && m.owner_id == owner_id) {
            ++count;
        }
    }
    return count;
}

void GameState::remove_mines_by_owner(PlayerId owner_id) {
    mines_.erase(
        std::remove_if(mines_.begin(), mines_.end(),
            [owner_id](const Mine& m) { return m.active && m.owner_id == owner_id; }),
        mines_.end()
    );
}

void GameState::check_mine_proximity() {
    for (auto& mine : mines_) {
        if (!mine.active) continue;
        if (mine.explosion_timer >= 0) continue;  // already detonating
        if (mine.being_hooked) continue;

        for (auto& pudge : pudges_) {
            if (!pudge.alive) continue;
            if (pudge.id == mine.owner_id) continue;

            // Chebyshev distance <= 1
            int dx = std::abs(pudge.pos.x - mine.pos.x);
            int dy = std::abs(pudge.pos.y - mine.pos.y);
            if (dx <= 1 && dy <= 1) {
                // Detonate
                mine.explosion_timer = MINE_EXPLOSION_TICKS;
                kill_pudge(pudge.id, mine.owner_id);
            }
        }
    }
}

void GameState::update_mine_explosions() {
    for (auto& mine : mines_) {
        if (!mine.active) continue;
        if (mine.explosion_timer < 0) continue;

        --mine.explosion_timer;
        if (mine.explosion_timer < 0) {
            mine.active = false;
        }
    }

    // Erase inactive mines
    mines_.erase(
        std::remove_if(mines_.begin(), mines_.end(),
            [](const Mine& m) { return !m.active; }),
        mines_.end()
    );
}

void GameState::kill_pudge(PlayerId victim_id, PlayerId killer_id) {
    Pudge* victim = get_pudge(victim_id);
    if (!victim || !victim->alive) return;

    // Release victim's hook target before resetting (C1: prevents permanent soft-lock)
    if (victim->hook.target_id != INVALID_PLAYER) {
        Pudge* pull_target = get_pudge(victim->hook.target_id);
        if (pull_target) pull_target->being_pulled = false;
    }

    // Clear being_hooked on victim's dragged mine (C2: prevents invulnerable mine)
    if (victim->hook.hooked_mine_id >= 0) {
        for (auto& m : mines_) {
            if (m.id == victim->hook.hooked_mine_id) {
                m.being_hooked = false;
                break;
            }
        }
    }

    victim->alive = false;
    victim->respawn_timer = 45;
    victim->being_pulled = false;
    victim->hook = Hook{};
    victim->hook.cooldown = 0;
    victim->score.deaths++;

    Pudge* killer = get_pudge(killer_id);
    if (killer) {
        killer->score.kills++;
    }
}

void GameState::respawn_pudge(PlayerId id) {
    Pudge* p = get_pudge(id);
    if (!p) return;

    p->alive = true;
    p->respawn_timer = -1;
    p->pos = next_spawn_point();
    p->hook = Hook{};
    p->move_cooldown = 0;
    p->mine_cooldown = 0;
    p->being_pulled = false;
}
