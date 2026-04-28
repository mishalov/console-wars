#include "bot/dqn_brain.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

// ============================================================================
// File-format constants
// ============================================================================

static constexpr char     kMagic[4] = {'C', 'W', 'B', 'T'};
static constexpr uint32_t kVersion  = 1;

// ============================================================================
// Construction
// ============================================================================

DqnBrain::DqnBrain()
    : online_({kInputDim, 128, 64, kOutputDim})
    , target_({kInputDim, 128, 64, kOutputDim})
    , buffer_(50000)
{
    sync_target();
}

// ============================================================================
// Action <-> index mapping
// ============================================================================

int DqnBrain::action_to_index(InputAction a)
{
    switch (a) {
        case InputAction::None:      return 0;
        case InputAction::MoveUp:    return 1;
        case InputAction::MoveDown:  return 2;
        case InputAction::MoveLeft:  return 3;
        case InputAction::MoveRight: return 4;
        case InputAction::HookUp:    return 5;
        case InputAction::HookDown:  return 6;
        case InputAction::HookLeft:  return 7;
        case InputAction::HookRight: return 8;
        case InputAction::PlaceMine: return 9;
        case InputAction::Quit:      return 0;  // should never be called with Quit
    }
    return 0;  // unreachable, but silences compiler warning
}

InputAction DqnBrain::index_to_action(int i)
{
    switch (i) {
        case  0: return InputAction::None;
        case  1: return InputAction::MoveUp;
        case  2: return InputAction::MoveDown;
        case  3: return InputAction::MoveLeft;
        case  4: return InputAction::MoveRight;
        case  5: return InputAction::HookUp;
        case  6: return InputAction::HookDown;
        case  7: return InputAction::HookLeft;
        case  8: return InputAction::HookRight;
        case  9: return InputAction::PlaceMine;
        default: return InputAction::None;
    }
}

// ============================================================================
// decide()  --  epsilon-greedy over valid_actions
// ============================================================================

InputAction DqnBrain::decide(const BotObservation& obs,
                              const std::vector<InputAction>& valid_actions)
{
    assert(!valid_actions.empty());

    // Epsilon-greedy: with probability epsilon_, choose uniformly at random
    // among valid_actions.
    {
        auto dist = std::uniform_real_distribution<float>(0.0f, 1.0f);
        if (dist(rng_) < epsilon_) {
            auto idx_dist = std::uniform_int_distribution<int>(
                0, static_cast<int>(valid_actions.size()) - 1);
            return valid_actions[static_cast<std::size_t>(idx_dist(rng_))];
        }
    }

    // Greedy: forward pass on the online network, pick the valid action
    // with the highest Q-value.
    std::vector<float> input(obs.features.begin(), obs.features.end());
    std::vector<float> q_values = online_.forward(input);

    InputAction best_action  = valid_actions[0];
    float       best_q       = -std::numeric_limits<float>::infinity();

    for (const auto& a : valid_actions) {
        float q = q_values[static_cast<std::size_t>(action_to_index(a))];
        if (q > best_q) {
            best_q      = q;
            best_action = a;
        }
    }

    return best_action;
}

// ============================================================================
// on_outcome()
// ============================================================================

void DqnBrain::on_outcome(const BotObservation& prev, InputAction action,
                           const BotObservation& curr, float reward)
{
    // Build a Transition.
    bot::Transition t;
    t.state.assign(prev.features.begin(), prev.features.end());
    t.action     = action_to_index(action);
    t.reward     = reward;
    t.next_state.assign(curr.features.begin(), curr.features.end());
    t.done       = !curr.alive;

    buffer_.add(std::move(t));
    ++total_steps_;

    // Train every kTrainInterval steps once the buffer is warm.
    if (total_steps_ % kTrainInterval == 0 &&
        buffer_.size() >= static_cast<std::size_t>(kMinBufferSize))
    {
        train_batch();
    }
}

// ============================================================================
// train_batch()  --  one Double-DQN mini-batch SGD step
// ============================================================================

void DqnBrain::train_batch()
{
    auto batch = buffer_.sample(static_cast<std::size_t>(kBatchSize));

    for (const auto& t : batch) {
        // --- Double DQN value estimation ---
        // Online network selects the best next action.
        std::vector<float> online_next = online_.forward(t.next_state);
        int best_next_action = 0;
        {
            float best_val = online_next[0];
            for (int i = 1; i < kOutputDim; ++i) {
                if (online_next[static_cast<std::size_t>(i)] > best_val) {
                    best_val = online_next[static_cast<std::size_t>(i)];
                    best_next_action = i;
                }
            }
        }

        // Target network evaluates Q at that action.
        std::vector<float> target_next = target_.forward(t.next_state);
        float q_target = target_next[static_cast<std::size_t>(best_next_action)];

        // TD target.
        float y = t.reward;
        if (!t.done) {
            y += kGamma * q_target;
        }

        // --- Update online network ---
        // Forward current state to get the full Q-vector, then replace the
        // taken action's entry with the TD target.
        std::vector<float> current_q = online_.forward(t.state);
        current_q[static_cast<std::size_t>(t.action)] = y;

        online_.backprop(t.state, current_q, learning_rate_);
    }

    ++train_steps_;

    // Periodic target-network sync.
    if (train_steps_ % kTargetSyncEvery == 0) {
        sync_target();
    }
}

// ============================================================================
// Helpers
// ============================================================================

void DqnBrain::sync_target()
{
    target_.copy_weights_from(online_);
}

// ============================================================================
// on_game_end()
// ============================================================================

void DqnBrain::on_game_end()
{
    ++games_played_;

    // Epsilon decay: linearly anneal from 1.0 to kEpsilonMin over 200
    // "games" (note: a "game" here is one death-respawn cycle, not a full
    // session — this means epsilon decays quickly with frequent deaths).
    epsilon_ = std::max(kEpsilonMin,
                        1.0f - static_cast<float>(games_played_) / 200.0f);

    // Learning-rate half-life decay: halve every 150 games, floor at kLrMin.
    learning_rate_ = std::max(kLrMin,
        0.0005f * std::pow(0.5f, static_cast<float>(games_played_) / 150.0f));
}

// ============================================================================
// Persistence
// ============================================================================

void DqnBrain::save(const std::string& path) const
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("DqnBrain::save: cannot open " + path);
    }

    // Magic + version
    ofs.write(kMagic, sizeof(kMagic));
    ofs.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));

    // Scalar state
    ofs.write(reinterpret_cast<const char*>(&games_played_),  sizeof(games_played_));
    ofs.write(reinterpret_cast<const char*>(&total_steps_),   sizeof(total_steps_));
    ofs.write(reinterpret_cast<const char*>(&train_steps_),   sizeof(train_steps_));
    ofs.write(reinterpret_cast<const char*>(&epsilon_),       sizeof(epsilon_));
    ofs.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(learning_rate_));

    // Networks
    online_.save(ofs);
    target_.save(ofs);

    if (!ofs) {
        throw std::runtime_error("DqnBrain::save: write error");
    }
}

void DqnBrain::load(const std::string& path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("DqnBrain::load: cannot open " + path);
    }

    // Magic
    char magic[4]{};
    ifs.read(magic, sizeof(magic));
    if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("DqnBrain::load: bad magic header");
    }

    // Version
    uint32_t version = 0;
    ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != kVersion) {
        throw std::runtime_error("DqnBrain::load: unsupported version");
    }

    // Scalar state
    ifs.read(reinterpret_cast<char*>(&games_played_),  sizeof(games_played_));
    ifs.read(reinterpret_cast<char*>(&total_steps_),   sizeof(total_steps_));
    ifs.read(reinterpret_cast<char*>(&train_steps_),   sizeof(train_steps_));
    ifs.read(reinterpret_cast<char*>(&epsilon_),       sizeof(epsilon_));
    ifs.read(reinterpret_cast<char*>(&learning_rate_), sizeof(learning_rate_));

    // Networks
    online_.load(ifs);
    target_.load(ifs);

    if (!ifs) {
        throw std::runtime_error("DqnBrain::load: stream error during read");
    }
}
