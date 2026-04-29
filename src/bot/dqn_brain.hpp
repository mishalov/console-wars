#pragma once

#include "bot_brain.hpp"
#include "neural_net.hpp"
#include "replay_buffer.hpp"

#include <array>
#include <cstdint>
#include <iosfwd>
#include <random>
#include <string>
#include <vector>

/// Double-DQN implementation of BotBrain.
///
/// Uses two NeuralNet instances (online + target) and an experience replay
/// buffer.  The online network selects actions via epsilon-greedy; the
/// target network is used for value estimation (Double DQN).
///
/// Performance optimizations:
/// - Pre-allocated neural net workspace buffers (zero heap allocs per forward/backprop)
/// - Cached forward pass results to eliminate redundant computations in train_batch
/// - std::array-based Transition to eliminate replay buffer heap fragmentation
/// - Precomputed gamma power lookup table
/// - Batch sorted by buffer index for cache locality
class DqnBrain : public BotBrain {
public:
    explicit DqnBrain(bool inference_mode = false);

    InputAction decide(const BotObservation& obs,
                       const std::vector<InputAction>& valid_actions) override;

    void on_outcome(const BotObservation& prev, InputAction action,
                    const BotObservation& curr, float reward,
                    int n_steps) override;

    void on_game_end() override;

    void save(const std::string& path) const override;
    void save(std::ostream& os) const override;  // stream overload for async serialization
    void load(const std::string& path) override;

    // --- Accessors (useful for diagnostics / tests) -------------------------
    float epsilon()      const noexcept { return epsilon_; }
    float learning_rate() const noexcept { return learning_rate_; }
    uint32_t games_played() const noexcept { return games_played_; }
    uint32_t total_steps()  const noexcept { return total_steps_; }
    uint32_t train_steps()  const noexcept { return train_steps_; }

private:
    // --- Topology -----------------------------------------------------------
    static constexpr int kInputDim  = BotObservation::SIZE;  // 92
    static constexpr int kOutputDim = 10;                    // InputActions (excl. Quit)

    // --- Networks / buffer --------------------------------------------------
    bot::NeuralNet   online_;
    bot::NeuralNet   target_;
    bot::ReplayBuffer buffer_;

    // --- Hyperparameters ----------------------------------------------------
    float    epsilon_       = 0.5f;
    float    learning_rate_ = 0.0003f;
    uint32_t games_played_  = 0;
    uint32_t total_steps_   = 0;
    uint32_t train_steps_   = 0;

    static constexpr float    kGamma           = 0.99f;
    static constexpr int      kBatchSize       = 32;
    static constexpr uint32_t kMinBufferSize   = 1000;
    static constexpr uint32_t kTrainInterval   = 4;
    static constexpr float    kTau             = 0.005f;
    static constexpr float    kEpsilonMin      = 0.02f;

    // PER beta annealing: IS correction grows from kBetaStart to kBetaEnd
    static constexpr float    kBetaStart       = 0.4f;
    static constexpr float    kBetaEnd         = 1.0f;
    static constexpr uint32_t kBetaAnnealSteps = 100000;

    // N-step return: max n_steps value (from bot_player.hpp kNSteps = 5)
    static constexpr int kMaxNSteps = 5;

    // Precomputed gamma^n for n = 0..kMaxNSteps (avoids std::pow per sample).
    static constexpr std::array<float, kMaxNSteps + 1> kGammaPow = {{
        1.0f,                                   // gamma^0
        kGamma,                                 // gamma^1
        kGamma * kGamma,                        // gamma^2
        kGamma * kGamma * kGamma,               // gamma^3
        kGamma * kGamma * kGamma * kGamma,      // gamma^4
        kGamma * kGamma * kGamma * kGamma * kGamma  // gamma^5
    }};

    // --- RNG ----------------------------------------------------------------
    const bool   inference_mode_ = false;
    std::mt19937 rng_{std::random_device{}()};

    // --- Helpers ------------------------------------------------------------
    void train_batch();
    void sync_target();

    static int         action_to_index(InputAction a);
    static InputAction index_to_action(int i);
};
