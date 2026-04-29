#pragma once
#include "types.hpp"
#include "pudge.hpp"
#include "mine.hpp"
#include "bonus.hpp"
#include "input.hpp"
#include <random>
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

    // Bonus system
    const Bonus& bonus() const { return bonus_; }

    // Extra hooks (MultiHook bonus)
    struct ExtraHook {
        Hook hook;
        PlayerId owner_id = INVALID_PLAYER;
    };
    const std::vector<ExtraHook>& extra_hooks() const { return extra_hooks_; }

private:
    int width_ = 0;
    int height_ = 0;
    std::vector<TileType> grid_;

    std::vector<Pudge> pudges_;
    PlayerId next_id_ = 0;
    uint32_t tick_count_ = 0;

    std::vector<Mine> mines_;
    int next_mine_id_ = 0;

    // Bonus system
    Bonus bonus_;
    int bonus_spawn_timer_ = BONUS_SPAWN_INTERVAL;
    std::vector<ExtraHook> extra_hooks_;
    std::mt19937 rng_{std::random_device{}()};

    bool is_walkable(Vec2 pos) const;
    bool is_occupied(Vec2 pos, PlayerId exclude = INVALID_PLAYER) const;
    void check_mine_proximity();
    void update_mine_explosions();

    // Bonus helpers
    void update_bonus_spawn();
    void check_bonus_pickup();
    void update_bonus_timers();
    void apply_bonus(PlayerId id, BonusType type);
    void tick_extra_hooks();
    Vec2 random_empty_tile();
};
