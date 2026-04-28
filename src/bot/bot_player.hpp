#pragma once

#include "bot_brain.hpp"
#include "bot_observation.hpp"
#include "bot_reward.hpp"
#include "../game_state.hpp"
#include "../input.hpp"
#include "../types.hpp"

#include <memory>
#include <string>
#include <vector>

/// High-level orchestrator that drives a bot within the game loop.
///
/// Usage:
///   BotPlayer bot(id, "data/bot");
///   // game loop:
///   bot.pre_tick(state);   // observe, decide, inject action
///   state.tick();
///   bot.post_tick(state);  // detect death / respawn transitions
///   // on shutdown:
///   bot.save();
class BotPlayer {
public:
    BotPlayer(PlayerId id, const std::string& data_dir);

    /// Called *before* GameState::tick().
    /// Observes the world, feeds reward to the brain, decides an action,
    /// and injects it into the game state via handle_input().
    void pre_tick(GameState& state);

    /// Called *after* GameState::tick().
    /// Detects death and respawn transitions so the brain can be notified
    /// and internal bookkeeping can be reset.
    void post_tick(const GameState& state);

    /// Persist the brain to disk.
    void save() const;

    PlayerId player_id() const noexcept { return id_; }

private:
    PlayerId                   id_;
    std::string                data_dir_;
    std::unique_ptr<BotBrain>  brain_;

    BotObservation             prev_obs_{};
    InputAction                last_action_ = InputAction::None;
    bool                       has_prev_    = false;
    bool                       was_dead_    = false;

    /// Build the list of actions the bot is allowed to take right now.
    std::vector<InputAction> get_valid_actions(const GameState& state) const;
};
