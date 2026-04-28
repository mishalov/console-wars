#include "renderer.hpp"
#include "mine.hpp"
#include <cstdio>

// ANSI 256-color palette indices for pudge colors
static constexpr int kPudgeColors[] = {196, 46, 33, 226, 201, 51, 208, 82};
static constexpr const char* kColorNames[] = {
    "Red", "Green", "Blue", "Yellow", "Magenta", "Cyan", "Orange", "Lime"
};
static constexpr int kNumColors = 8;

void Renderer::append_cursor_home(std::string& buf) const {
    buf.append("\x1b[H");
}

void Renderer::append_hide_cursor(std::string& buf) const {
    buf.append("\x1b[?25l");
}

void Renderer::append_mines(std::string& buf, const GameState& state) const {
    char seq[64];
    for (const auto& mine : state.mines()) {
        if (!mine.active) continue;

        int row = mine.pos.y + 1;
        int col = mine.pos.x * 2 + 1;

        if (mine.explosion_timer >= 0) {
            // Exploding: white on red background
            int n = std::snprintf(seq, sizeof(seq),
                "\x1b[%d;%dH\x1b[97;41m**\x1b[0m", row, col);
            buf.append(seq, static_cast<size_t>(n));
        } else {
            // Normal mine: <> in owner's color
            int color = kPudgeColors[mine.owner_id % kNumColors];
            int n = std::snprintf(seq, sizeof(seq),
                "\x1b[%d;%dH\x1b[38;5;%dm<>\x1b[0m", row, col, color);
            buf.append(seq, static_cast<size_t>(n));
        }
    }
}

void Renderer::append_pudges(std::string& buf, const GameState& state) const {
    char seq[48];
    for (const auto& pudge : state.pudges()) {
        if (!pudge.alive) continue;

        // Position cursor: row = pos.y + 1, col = pos.x * 2 + 1 (1-indexed)
        int row = pudge.pos.y + 1;
        int col = pudge.pos.x * 2 + 1;
        int color = kPudgeColors[pudge.id % kNumColors];

        int n = std::snprintf(seq, sizeof(seq), "\x1b[%d;%dH\x1b[38;5;%dm@@\x1b[0m",
                              row, col, color);
        buf.append(seq, static_cast<size_t>(n));
    }
}

void Renderer::append_hooks(std::string& buf, const GameState& state) const {
    char seq[48];
    for (const auto& pudge : state.pudges()) {
        if (!pudge.alive) continue;
        if (pudge.hook.state == HookState::Ready) continue;
        if (pudge.hook.chain.empty()) continue;

        int color = kPudgeColors[pudge.id % kNumColors];

        // Determine chain body and tip characters based on direction
        const char* chain_body = nullptr;
        const char* hook_tip = nullptr;
        switch (pudge.hook.direction) {
            case Direction::Up:
                chain_body = "||";
                hook_tip = "^^";
                break;
            case Direction::Down:
                chain_body = "||";
                hook_tip = "vv";
                break;
            case Direction::Left:
                chain_body = "==";
                hook_tip = "<<";
                break;
            case Direction::Right:
                chain_body = "==";
                hook_tip = ">>";
                break;
            default:
                chain_body = "..";
                hook_tip = "..";
                break;
        }

        // Render chain segments (all except the last, which is the tip)
        for (size_t i = 0; i + 1 < pudge.hook.chain.size(); ++i) {
            Vec2 pos = pudge.hook.chain[i];
            int row = pos.y + 1;
            int col = pos.x * 2 + 1;

            int n = std::snprintf(seq, sizeof(seq), "\x1b[%d;%dH\x1b[38;5;%dm%s\x1b[0m",
                                  row, col, color, chain_body);
            buf.append(seq, static_cast<size_t>(n));
        }

        // Render the tip (last element in chain)
        Vec2 tip_pos = pudge.hook.chain.back();
        int tip_row = tip_pos.y + 1;
        int tip_col = tip_pos.x * 2 + 1;

        int n = std::snprintf(seq, sizeof(seq), "\x1b[%d;%dH\x1b[1;38;5;%dm%s\x1b[0m",
                              tip_row, tip_col, color, hook_tip);
        buf.append(seq, static_cast<size_t>(n));
    }
}

void Renderer::append_hud(std::string& buf, const GameState& state, PlayerId viewer_id) const {
    // Move cursor to row below the map
    int hud_row = state.height() + 1;
    char seq[256];

    int color_idx = viewer_id % kNumColors;
    int player_count = static_cast<int>(state.pudges().size());

    // Find the viewer's pudge for score/status info
    const Pudge* viewer = nullptr;
    for (const auto& p : state.pudges()) {
        if (p.id == viewer_id) {
            viewer = &p;
            break;
        }
    }

    if (!viewer) {
        // Viewer not found (disconnected?), minimal HUD
        int n = std::snprintf(seq, sizeof(seq),
            "\x1b[%d;1H\x1b[KPlayers: %d | Spectating",
            hud_row, player_count);
        buf.append(seq, static_cast<size_t>(n));
        append_roster(buf, state, viewer_id, hud_row + 2);
        return;
    }

    // Check if dead
    if (!viewer->alive) {
        // Respawn timer display
        double respawn_sec = static_cast<double>(viewer->respawn_timer) / 15.0;
        int n = std::snprintf(seq, sizeof(seq),
            "\x1b[%d;1H\x1b[K*** RESPAWNING in %.1fs ***",
            hud_row, respawn_sec);
        buf.append(seq, static_cast<size_t>(n));
        // Clear second line
        int n2 = std::snprintf(seq, sizeof(seq), "\x1b[%d;1H\x1b[K", hud_row + 1);
        buf.append(seq, static_cast<size_t>(n2));
        append_roster(buf, state, viewer_id, hud_row + 3);
        return;
    }

    // --- Line 1: Players: N | You: Player X (Color) | K:3 D:1 | Hook: READY ---
    const char* hook_status = "";
    char hook_buf[32] = {};
    if (viewer->hook.state == HookState::Extending) {
        hook_status = ">>..";
    } else if (viewer->hook.state == HookState::Retracting) {
        hook_status = "<<..";
    } else if (viewer->hook.cooldown > 0) {
        double cd_sec = static_cast<double>(viewer->hook.cooldown) / 15.0;
        std::snprintf(hook_buf, sizeof(hook_buf), "CD %.1fs", cd_sec);
        hook_status = hook_buf;
    } else {
        hook_status = "READY";
    }

    int n = std::snprintf(seq, sizeof(seq),
        "\x1b[%d;1H\x1b[KPlayers: %d | You: Player %d (%s) | K:%d D:%d | Hook: %s",
        hud_row, player_count, viewer_id, kColorNames[color_idx],
        viewer->score.kills, viewer->score.deaths, hook_status);
    buf.append(seq, static_cast<size_t>(n));

    // --- Line 2: Mines: [*][*][ ] | Mine CD: ready ---
    int mines_placed = 0;
    for (const auto& m : state.mines()) {
        if (m.active && m.owner_id == viewer_id) {
            ++mines_placed;
        }
    }

    // Build mine slots string
    char mine_slots[32] = {};
    int slot_pos = 0;
    for (int i = 0; i < MAX_MINES_PER_PLAYER; ++i) {
        if (i < mines_placed) {
            slot_pos += std::snprintf(mine_slots + slot_pos,
                sizeof(mine_slots) - static_cast<size_t>(slot_pos), "[*]");
        } else {
            slot_pos += std::snprintf(mine_slots + slot_pos,
                sizeof(mine_slots) - static_cast<size_t>(slot_pos), "[ ]");
        }
    }

    const char* mine_cd_str = "";
    char mine_cd_buf[32] = {};
    if (viewer->mine_cooldown > 0) {
        double cd_sec = static_cast<double>(viewer->mine_cooldown) / 15.0;
        std::snprintf(mine_cd_buf, sizeof(mine_cd_buf), "%.1fs", cd_sec);
        mine_cd_str = mine_cd_buf;
    } else {
        mine_cd_str = "ready";
    }

    n = std::snprintf(seq, sizeof(seq),
        "\x1b[%d;1H\x1b[KMines: %s | Mine CD: %s",
        hud_row + 1, mine_slots, mine_cd_str);
    buf.append(seq, static_cast<size_t>(n));

    // --- Line 3: Controls ---
    n = std::snprintf(seq, sizeof(seq),
        "\x1b[%d;1H\x1b[K\x1b[90mWASD:move  IJKL:hook  SPACE:mine  Q:quit\x1b[0m",
        hud_row + 2);
    buf.append(seq, static_cast<size_t>(n));

    // Clear gap line between controls and roster
    n = std::snprintf(seq, sizeof(seq), "\x1b[%d;1H\x1b[K", hud_row + 3);
    buf.append(seq, static_cast<size_t>(n));

    // --- Roster below controls ---
    append_roster(buf, state, viewer_id, hud_row + 4);
}

void Renderer::append_roster(std::string& buf, const GameState& state, PlayerId viewer_id, int start_row) const {
    char seq[192];

    // Header
    int n = std::snprintf(seq, sizeof(seq),
        "\x1b[%d;1H\x1b[K\x1b[1;97m--- Players ---\x1b[0m", start_row);
    buf.append(seq, static_cast<size_t>(n));

    int row = start_row + 1;
    for (const auto& pudge : state.pudges()) {
        int color = kPudgeColors[pudge.id % kNumColors];
        const char* color_name = kColorNames[pudge.id % kNumColors];
        const char* type_label = pudge.is_bot ? "Bot" : "Player";
        bool is_viewer = (pudge.id == viewer_id);

        // Format: @@ ColorName  Type (You)  K:X D:Y
        if (is_viewer) {
            n = std::snprintf(seq, sizeof(seq),
                "\x1b[%d;1H\x1b[K \x1b[38;5;%dm@@\x1b[0m %-8s %-6s \x1b[1;97m(You)\x1b[0m  K:%d D:%d",
                row, color, color_name, type_label,
                pudge.score.kills, pudge.score.deaths);
        } else {
            n = std::snprintf(seq, sizeof(seq),
                "\x1b[%d;1H\x1b[K \x1b[38;5;%dm@@\x1b[0m %-8s %-6s        K:%d D:%d",
                row, color, color_name, type_label,
                pudge.score.kills, pudge.score.deaths);
        }
        buf.append(seq, static_cast<size_t>(n));
        ++row;
    }

    // Clear next line to remove stale entries from disconnected players
    n = std::snprintf(seq, sizeof(seq), "\x1b[%d;1H\x1b[K", row);
    buf.append(seq, static_cast<size_t>(n));
}

std::string Renderer::render_full(const GameState& state, PlayerId viewer_id) const {
    std::string buf;
    // Reserve rough estimate: 2 chars per tile + ANSI codes + pudges + HUD
    buf.reserve(static_cast<size_t>(state.width() * state.height() * 12 + 512));

    append_hide_cursor(buf);
    append_cursor_home(buf);

    // Tiles
    for (int y = 0; y < state.height(); ++y) {
        for (int x = 0; x < state.width(); ++x) {
            TileType tile = state.tile_at(x, y);
            if (tile == TileType::Wall) {
                buf.append("\x1b[100m  \x1b[0m");  // dark gray background
            } else {
                buf.append("  ");  // empty space (2 chars per tile)
            }
        }
        buf.append("\r\n");
    }

    // Order: tiles -> hooks -> mines -> pudges -> HUD
    append_hooks(buf, state);
    append_mines(buf, state);
    append_pudges(buf, state);
    append_hud(buf, state, viewer_id);

    return buf;
}
