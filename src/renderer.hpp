#pragma once
#include "game_state.hpp"
#include "types.hpp"
#include <string>

class Renderer {
public:
    // Renders the full frame for a specific viewer (includes per-player HUD)
    std::string render_full(const GameState& state, PlayerId viewer_id) const;

private:
    void append_cursor_home(std::string& buf) const;
    void append_hide_cursor(std::string& buf) const;
    void append_mines(std::string& buf, const GameState& state) const;
    void append_hooks(std::string& buf, const GameState& state) const;
    void append_extra_hooks(std::string& buf, const GameState& state) const;
    void append_bonus(std::string& buf, const GameState& state) const;
    void append_pudges(std::string& buf, const GameState& state) const;
    void append_hud(std::string& buf, const GameState& state, PlayerId viewer_id) const;
    void append_roster(std::string& buf, const GameState& state, PlayerId viewer_id, int start_row) const;
};
