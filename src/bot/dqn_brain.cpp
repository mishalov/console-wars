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
static constexpr uint32_t kVersion  = 3;

// ============================================================================
// Construction
// ============================================================================

DqnBrain::DqnBrain(bool inference_mode)
    : online_({kInputDim, 256, 128, 64, kOutputDim})
    , target_({kInputDim, 256, 128, 64, kOutputDim})
    , buffer_(inference_mode ? 1 : 100000)
    , inference_mode_(inference_mode)
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

    // Inference mode: small epsilon for stochasticity, use stable target network.
    float effective_epsilon = inference_mode_ ? 0.01f : epsilon_;

    // Epsilon-greedy: with probability epsilon_, choose uniformly at random
    // among valid_actions.
    {
        auto dist = std::uniform_real_distribution<float>(0.0f, 1.0f);
        if (dist(rng_) < effective_epsilon) {
            auto idx_dist = std::uniform_int_distribution<int>(
                0, static_cast<int>(valid_actions.size()) - 1);
            return valid_actions[static_cast<std::size_t>(idx_dist(rng_))];
        }
    }

    // Greedy: use target network in inference (more stable Polyak-averaged Q-estimates),
    // online network in training.
    const auto& net = inference_mode_ ? target_ : online_;
    std::vector<float> q_values = net.forward(obs.features.data(), kInputDim);

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
                           const BotObservation& curr, float reward,
                           int n_steps)
{
    if (inference_mode_) return;

    // Build a Transition using std::array (no heap allocation).
    bot::Transition t;
    std::copy(prev.features.begin(), prev.features.end(), t.state.begin());
    t.action     = action_to_index(action);
    t.reward     = reward;
    std::copy(curr.features.begin(), curr.features.end(), t.next_state.begin());
    t.done       = !curr.alive;
    t.n_steps    = n_steps;

    buffer_.add(std::move(t), std::abs(reward) + 0.01f);
    ++total_steps_;

    // Train every kTrainInterval steps once the buffer is warm.
    if (total_steps_ % kTrainInterval == 0 &&
        buffer_.size() >= static_cast<std::size_t>(kMinBufferSize))
    {
        train_batch();
    }
}

// ============================================================================
// train_batch()  --  Optimized Double-DQN mini-batch step with n-step returns
//
// Key optimizations vs. original:
//   1. Cache current_q from first loop, reuse in second loop (eliminates 32 forward passes)
//   2. Use backprop_from_cached() with pre-computed activations (eliminates 32 more)
//   3. Use precomputed kGammaPow[] lookup table instead of std::pow()
//   4. Sort batch by buffer index for better cache locality
//   5. forward() uses pre-allocated workspace buffers (zero heap allocs)
// ============================================================================

void DqnBrain::train_batch()
{
    // Beta annealing for importance-sampling correction.
    float beta = kBetaStart + (kBetaEnd - kBetaStart) *
        std::min(1.0f, static_cast<float>(train_steps_) / static_cast<float>(kBetaAnnealSteps));

    auto batch = buffer_.sample_prioritized(
        static_cast<std::size_t>(kBatchSize), beta);

    // Sort batch by buffer index for cache locality when accessing transition data.
    std::sort(batch.begin(), batch.end(),
              [](const bot::ReplayBuffer::SampledTransition& a,
                 const bot::ReplayBuffer::SampledTransition& b) {
                  return a.index < b.index;
              });

    const auto batch_sz = batch.size();

    // Pre-compute all TD targets, errors, and cache current Q-values + activations.
    // This avoids the redundant second forward pass on t.state in the update loop.
    std::vector<float> td_targets(batch_sz);
    std::vector<float> td_errors(batch_sz);

    // Cache: current_q values and their activations for backprop_from_cached.
    std::vector<std::vector<float>> cached_current_q(batch_sz);
    std::vector<std::vector<std::vector<float>>> cached_activations(batch_sz);

    for (std::size_t i = 0; i < batch_sz; ++i) {
        const auto& t = *batch[i].transition;

        // Double DQN: online selects best next action.
        std::vector<float> online_next = online_.forward(t.next_state.data(), kInputDim);
        int best_next_action = 0;
        {
            float best_val = online_next[0];
            for (int a = 1; a < kOutputDim; ++a) {
                if (online_next[static_cast<std::size_t>(a)] > best_val) {
                    best_val = online_next[static_cast<std::size_t>(a)];
                    best_next_action = a;
                }
            }
        }

        // Target evaluates at that action.
        std::vector<float> target_next = target_.forward(t.next_state.data(), kInputDim);
        float q_target = target_next[static_cast<std::size_t>(best_next_action)];

        // N-step TD target: y = G_n + gamma^n * Q_target(s', a*)
        // Use precomputed gamma power LUT instead of std::pow.
        float y = t.reward;
        if (!t.done) {
            int ns = t.n_steps;
            float gamma_n = (ns >= 0 && ns <= kMaxNSteps)
                          ? kGammaPow[static_cast<std::size_t>(ns)]
                          : std::pow(kGamma, static_cast<float>(ns));
            y += gamma_n * q_target;
        }
        td_targets[i] = y;

        // Forward pass on current state -- cache the result AND the internal activations.
        cached_current_q[i] = online_.forward(t.state.data(), kInputDim);
        cached_activations[i] = online_.cached_activations();

        td_errors[i] = y - cached_current_q[i][static_cast<std::size_t>(t.action)];
    }

    // Apply updates with importance-sampling weights.
    // Uses cached Q-values (no redundant forward pass) and backprop_from_cached
    // (avoids the internal forward pass inside backprop).
    for (std::size_t i = 0; i < batch_sz; ++i) {
        const auto& t = *batch[i].transition;
        float is_weight = batch[i].weight;

        // Reuse cached current_q (eliminates pass #4 from original).
        std::vector<float>& current_q = cached_current_q[i];

        // Scale the gradient by the IS weight:
        // target[action] = pred + is_weight * (td_target - pred)
        float pred = current_q[static_cast<std::size_t>(t.action)];
        current_q[static_cast<std::size_t>(t.action)] = pred + is_weight * (td_targets[i] - pred);

        // Use backprop_from_cached to avoid the redundant forward pass inside backprop
        // (eliminates pass #5 from original).
        online_.backprop_from_cached(cached_activations[i], current_q, learning_rate_);

        // Update priority in the sum-tree.
        buffer_.update_priority(batch[i].index, td_errors[i]);
    }

    ++train_steps_;

    // Soft target update every training step (Polyak averaging).
    target_.soft_update(online_, kTau);

    // Epsilon decay: exponential anneal based on training steps.
    epsilon_ = std::max(kEpsilonMin,
                        0.5f * std::exp(-static_cast<float>(train_steps_) / 15000.0f));
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
    if (inference_mode_) return;
    ++games_played_;
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
    save(ofs);
    if (!ofs) {
        throw std::runtime_error("DqnBrain::save: write error");
    }
}

void DqnBrain::save(std::ostream& os) const
{
    // Magic + version
    os.write(kMagic, sizeof(kMagic));
    os.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));

    // Scalar state
    os.write(reinterpret_cast<const char*>(&games_played_),  sizeof(games_played_));
    os.write(reinterpret_cast<const char*>(&total_steps_),   sizeof(total_steps_));
    os.write(reinterpret_cast<const char*>(&train_steps_),   sizeof(train_steps_));
    os.write(reinterpret_cast<const char*>(&epsilon_),       sizeof(epsilon_));
    os.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(learning_rate_));

    // Networks
    online_.save(os);
    target_.save(os);
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
