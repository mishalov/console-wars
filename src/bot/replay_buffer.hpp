#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

#include "bot_observation.hpp"

namespace bot {

// State dimension fixed at BotObservation::SIZE (92).
// Using std::array eliminates ~200K heap allocations in the replay buffer.
static constexpr int kStateDim = BotObservation::SIZE;

struct Transition {
    std::array<float, kStateDim> state{};       // observation vector (92 floats)
    int action = 0;                              // action index (0-9)
    float reward = 0.0f;
    std::array<float, kStateDim> next_state{};  // next observation vector (92 floats)
    bool done = false;                           // episode terminal?
    int n_steps = 1;                             // number of steps in this n-step return
};

// Sum-tree for O(log N) proportional sampling and O(log N) priority update.
// Internal array layout: tree_[1] = root, tree_[cap..2*cap-1] = leaves.
// Capacity is rounded up to the next power of 2 for proper binary-tree alignment.
class SumTree {
public:
    explicit SumTree(std::size_t capacity)
        : capacity_(next_power_of_two(capacity))
        , tree_(2 * capacity_, 0.0f)
    {}

    // Set priority for leaf at data_index.
    void update(std::size_t data_index, float priority) {
        std::size_t tree_index = data_index + capacity_;
        float delta = priority - tree_[tree_index];
        tree_[tree_index] = priority;
        // Propagate up to root.
        while (tree_index > 1) {
            tree_index /= 2;
            tree_[tree_index] += delta;
        }
    }

    // Sample a leaf proportionally.  Returns the data index.
    [[nodiscard]] std::size_t sample(float value) const {
        std::size_t idx = 1;  // root
        while (idx < capacity_) {
            std::size_t left = 2 * idx;
            if (value <= tree_[left]) {
                idx = left;
            } else {
                value -= tree_[left];
                idx = left + 1;
            }
        }
        return idx - capacity_;
    }

    [[nodiscard]] float total() const { return tree_[1]; }

    [[nodiscard]] float priority_at(std::size_t data_index) const {
        return tree_[data_index + capacity_];
    }

    [[nodiscard]] std::size_t capacity() const { return capacity_; }

private:
    std::size_t capacity_;
    std::vector<float> tree_;

    // Round up to nearest power of 2.
    static std::size_t next_power_of_two(std::size_t n) {
        if (n == 0) return 1;
        std::size_t p = 1;
        while (p < n) p <<= 1;
        return p;
    }
};

// Prioritized Experience Replay buffer.
// Samples transitions proportionally to their TD-error priority.
class ReplayBuffer {
public:
    explicit ReplayBuffer(std::size_t capacity = 100000)
        : capacity_(capacity)
        , tree_(capacity)
    {
        data_.reserve(capacity_);
    }

    // Add a transition with an initial priority.
    // Overwrites the oldest entry when the buffer is full.
    void add(Transition t, float priority = 1.0f) {
        float p = std::pow(std::max(priority, min_priority_), alpha_);

        if (data_.size() < capacity_) {
            data_.push_back(std::move(t));
            tree_.update(data_.size() - 1, p);
        } else {
            data_[write_idx_] = std::move(t);
            tree_.update(write_idx_, p);
        }

        if (p > max_priority_) max_priority_ = p;
        write_idx_ = (write_idx_ + 1) % capacity_;
    }

    // A sampled transition together with its buffer index and IS weight.
    struct SampledTransition {
        const Transition* transition;
        std::size_t index;
        float weight;  // importance-sampling weight
    };

    // Sample a batch with proportional priorities.
    // beta controls importance-sampling correction (0 = none, 1 = full).
    [[nodiscard]] std::vector<SampledTransition> sample_prioritized(
        std::size_t batch_size, float beta)
    {
        const auto n = size();
        if (batch_size > n) {
            throw std::invalid_argument("batch_size exceeds buffer size");
        }

        std::vector<SampledTransition> batch;
        batch.reserve(batch_size);

        float total = tree_.total();
        if (total <= 0.0f) total = 1.0f;

        float segment = total / static_cast<float>(batch_size);

        // Use the constant priority floor for IS weight normalisation.
        // All priorities are >= pow(min_priority_, alpha_), so this is a safe lower bound.
        float min_prob_bound = std::pow(min_priority_, alpha_) / total;
        float max_weight = std::pow(static_cast<float>(n) * min_prob_bound, -beta);

        for (std::size_t i = 0; i < batch_size; ++i) {
            float lo = segment * static_cast<float>(i);
            float hi = segment * static_cast<float>(i + 1);
            auto dist = std::uniform_real_distribution<float>(lo, hi);
            float value = dist(rng_);

            std::size_t idx = tree_.sample(value);
            if (idx >= n) idx = n - 1;  // safety clamp

            float prob = tree_.priority_at(idx) / total;
            if (prob <= 0.0f) prob = min_priority_ / total;

            float weight = std::pow(static_cast<float>(n) * prob, -beta) / max_weight;

            batch.push_back({&data_[idx], idx, weight});
        }

        return batch;
    }

    // Update priority for a specific index using the TD error magnitude.
    void update_priority(std::size_t index, float td_error) {
        float p = std::pow(std::max(std::abs(td_error) + min_priority_, min_priority_), alpha_);
        tree_.update(index, p);
        if (p > max_priority_) max_priority_ = p;
    }

    [[nodiscard]] std::size_t size()     const noexcept { return data_.size(); }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] bool        empty()    const noexcept { return data_.empty(); }

    // Seed the internal RNG for reproducible sampling.
    void seed(unsigned s) { rng_.seed(s); }

private:
    std::size_t capacity_;
    std::size_t write_idx_ = 0;
    std::vector<Transition> data_;
    SumTree tree_;
    std::mt19937 rng_{std::random_device{}()};

    static constexpr float alpha_       = 0.6f;   // prioritisation exponent
    static constexpr float min_priority_ = 0.01f;  // floor to avoid zero priority
    float max_priority_ = 1.0f;                    // tracks max for new transitions
};

}  // namespace bot
