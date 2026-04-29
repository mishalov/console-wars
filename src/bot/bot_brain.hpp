#pragma once

#include "bot_observation.hpp"
#include "../input.hpp"

#include <iosfwd>
#include <string>
#include <vector>

/// Abstract interface for a bot decision-making module.
/// Implementations receive observations and produce actions,
/// learn from outcomes, and support persistence.
class BotBrain {
public:
    virtual ~BotBrain() = default;

    /// Choose an action given the current observation and the set of
    /// currently valid actions.  Must return one element of valid_actions.
    virtual InputAction decide(const BotObservation& obs,
                               const std::vector<InputAction>& valid_actions) = 0;

    /// Called after each game tick with the previous observation, the action
    /// taken, the resulting observation, and the scalar reward.
    /// n_steps indicates how many environment steps this compound reward spans
    /// (used for n-step returns; defaults to 1 for standard 1-step TD).
    virtual void on_outcome(const BotObservation& prev, InputAction action,
                            const BotObservation& curr, float reward,
                            int n_steps = 1) = 0;

    /// Called when a round ends (the bot died and will respawn).
    virtual void on_game_end() = 0;

    /// Persist learned weights / state to the given path.
    virtual void save(const std::string& path) const = 0;

    /// Persist learned weights / state to an output stream.
    virtual void save(std::ostream& os) const = 0;

    /// Restore learned weights / state from the given path.
    virtual void load(const std::string& path) = 0;
};
