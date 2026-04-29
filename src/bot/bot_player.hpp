#pragma once

#include "bot_brain.hpp"
#include "bot_observation.hpp"
#include "bot_reward.hpp"
#include "../game_state.hpp"
#include "../input.hpp"
#include "../types.hpp"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
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
    BotPlayer(PlayerId id, const std::string& data_dir, bool inference_mode = false);
    ~BotPlayer();

    // Non-copyable, non-movable (owns mutex + thread)
    BotPlayer(const BotPlayer&) = delete;
    BotPlayer& operator=(const BotPlayer&) = delete;
    BotPlayer(BotPlayer&&) = delete;
    BotPlayer& operator=(BotPlayer&&) = delete;

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
    bool                       inference_mode_ = false;
    std::unique_ptr<BotBrain>  brain_;

    BotObservation             prev_obs_{};
    InputAction                last_action_ = InputAction::None;
    bool                       has_prev_    = false;
    bool                       was_dead_    = false;

    // Velocity tracking: previous tick positions for observation deltas
    Vec2                       prev_self_pos_{};
    Vec2                       prev_enemy_pos_{};
    PlayerId                   prev_enemy_id_ = INVALID_PLAYER;

    // N-step return accumulation
    struct PendingStep {
        BotObservation obs;
        InputAction action = InputAction::None;
        float reward = 0.0f;
    };
    std::deque<PendingStep> nstep_buffer_;
    static constexpr int   kNSteps = 5;
    static constexpr float kGamma  = 0.99f;  // must match DqnBrain::kGamma

    /// Flush the oldest n-step transition from the buffer.
    void flush_nstep_transition(const BotObservation& next_obs);
    /// Flush all remaining n-step transitions (called at episode end).
    void flush_all_nstep(const BotObservation& terminal_obs);

    /// Build the list of actions the bot is allowed to take right now.
    std::vector<InputAction> get_valid_actions(const GameState& state) const;

    // --- Async save infrastructure ---
    static constexpr int kSaveEveryNGames = 10;  // save every Nth death
    int death_count_ = 0;

    // Background writer thread
    mutable std::mutex       save_mutex_;
    mutable std::condition_variable save_cv_;
    mutable std::string      save_pending_;  // serialized data awaiting write
    mutable std::atomic<bool> save_shutdown_{false};
    mutable std::thread       save_thread_;

    void save_async() const;
    void save_thread_func() const;
};
