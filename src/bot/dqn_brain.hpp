#pragma once

#include "bot_brain.hpp"
#include "neural_net.hpp"
#include "replay_buffer.hpp"

#include <cstdint>
#include <random>
#include <string>
#include <vector>

/// Double-DQN implementation of BotBrain.
///
/// Uses two NeuralNet instances (online + target) and an experience replay
/// buffer.  The online network selects actions via epsilon-greedy; the
/// target network is used for value estimation (Double DQN).
class DqnBrain : public BotBrain {
public:
    DqnBrain();

    InputAction decide(const BotObservation& obs,
                       const std::vector<InputAction>& valid_actions) override;

    void on_outcome(const BotObservation& prev, InputAction action,
                    const BotObservation& curr, float reward) override;

    void on_game_end() override;

    void save(const std::string& path) const override;
    void load(const std::string& path) override;

    // --- Accessors (useful for diagnostics / tests) -------------------------
    float epsilon()      const noexcept { return epsilon_; }
    float learning_rate() const noexcept { return learning_rate_; }
    uint32_t games_played() const noexcept { return games_played_; }
    uint32_t total_steps()  const noexcept { return total_steps_; }
    uint32_t train_steps()  const noexcept { return train_steps_; }

private:
    // --- Topology -----------------------------------------------------------
    static constexpr int kInputDim  = BotObservation::SIZE;  // 47
    static constexpr int kOutputDim = 10;                    // InputActions (excl. Quit)

    // --- Networks / buffer --------------------------------------------------
    bot::NeuralNet   online_;
    bot::NeuralNet   target_;
    bot::ReplayBuffer buffer_;

    // --- Hyperparameters ----------------------------------------------------
    float    epsilon_       = 1.0f;
    float    learning_rate_ = 0.0005f;
    uint32_t games_played_  = 0;
    uint32_t total_steps_   = 0;
    uint32_t train_steps_   = 0;

    static constexpr float    kGamma           = 0.97f;
    static constexpr int      kBatchSize       = 32;
    static constexpr uint32_t kMinBufferSize   = 5000;
    static constexpr uint32_t kTrainInterval   = 4;
    static constexpr uint32_t kTargetSyncEvery = 1000;
    static constexpr float    kEpsilonMin      = 0.05f;
    static constexpr float    kLrMin           = 0.00005f;

    // --- RNG ----------------------------------------------------------------
    std::mt19937 rng_{std::random_device{}()};

    // --- Helpers ------------------------------------------------------------
    void train_batch();
    void sync_target();

    static int         action_to_index(InputAction a);
    static InputAction index_to_action(int i);
};
