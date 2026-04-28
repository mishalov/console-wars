#pragma once
#include "types.hpp"
#include "pudge.hpp"
#include "mine.hpp"
#include "input.hpp"
#include <string>
#include <vector>

class GameState {
public:
    bool load_map(const std::string& path);

    int width() const { return width_; }
    int height() const { return height_; }
    TileType tile_at(int x, int y) const;

    // Pudge management
    PlayerId add_pudge(Vec2 spawn_pos, bool is_bot = false);
    void remove_pudge(PlayerId id);
    Pudge* get_pudge(PlayerId id);
    const std::vector<Pudge>& pudges() const;
    void handle_input(PlayerId id, InputAction action);
    void tick();
    uint32_t tick_count() const;

    // Spawn point cycling
    Vec2 next_spawn_point();

    // Mine management
    void place_mine(PlayerId owner_id, Vec2 pos);
    const std::vector<Mine>& mines() const;
    void remove_mines_by_owner(PlayerId owner_id);
    int mines_owned_by(PlayerId owner_id) const;

    // Kill / respawn
    void kill_pudge(PlayerId victim_id, PlayerId killer_id);
    void respawn_pudge(PlayerId id);

private:
    int width_ = 0;
    int height_ = 0;
    std::vector<TileType> grid_;

    std::vector<Pudge> pudges_;
    PlayerId next_id_ = 0;
    uint32_t tick_count_ = 0;

    std::vector<Mine> mines_;
    int next_mine_id_ = 0;

    bool is_walkable(Vec2 pos) const;
    bool is_occupied(Vec2 pos, PlayerId exclude = INVALID_PLAYER) const;
    void check_mine_proximity();
    void update_mine_explosions();
};
