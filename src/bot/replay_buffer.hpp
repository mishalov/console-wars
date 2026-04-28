#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

namespace bot {

struct Transition {
    std::vector<float> state;       // observation vector
    int action = 0;                 // action index (0-10)
    float reward = 0.0f;
    std::vector<float> next_state;  // next observation vector
    bool done = false;              // episode terminal?
};

// Fixed-capacity ring buffer for experience replay.
// Uniform random sampling without replacement.
class ReplayBuffer {
public:
    explicit ReplayBuffer(std::size_t capacity = 50000)
        : capacity_(capacity)
    {
        buffer_.reserve(capacity_);
    }

    // Add a transition; overwrites the oldest entry when full.
    void add(Transition t) {
        if (buffer_.size() < capacity_) {
            buffer_.push_back(std::move(t));
        } else {
            buffer_[write_idx_] = std::move(t);
        }
        write_idx_ = (write_idx_ + 1) % capacity_;
    }

    // Uniform random sample *without* replacement.
    // batch_size must be <= size().
    [[nodiscard]] std::vector<Transition> sample(std::size_t batch_size) {
        const auto n = size();
        if (batch_size > n) {
            throw std::invalid_argument(
                "ReplayBuffer::sample: batch_size exceeds buffer size");
        }

        // Fisher-Yates partial shuffle on an index array to pick
        // batch_size unique indices.
        std::vector<std::size_t> indices(n);
        for (std::size_t i = 0; i < n; ++i) {
            indices[i] = i;
        }

        for (std::size_t i = 0; i < batch_size; ++i) {
            auto dist = std::uniform_int_distribution<std::size_t>(i, n - 1);
            std::size_t j = dist(rng_);
            std::swap(indices[i], indices[j]);
        }

        std::vector<Transition> batch;
        batch.reserve(batch_size);
        for (std::size_t i = 0; i < batch_size; ++i) {
            batch.push_back(buffer_[indices[i]]);
        }
        return batch;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return buffer_.size();
    }

    [[nodiscard]] std::size_t capacity() const noexcept {
        return capacity_;
    }

    [[nodiscard]] bool empty() const noexcept {
        return buffer_.empty();
    }

    // Seed the internal RNG for reproducible sampling.
    void seed(unsigned s) {
        rng_.seed(s);
    }

private:
    std::size_t capacity_;
    std::size_t write_idx_ = 0;
    std::vector<Transition> buffer_;
    std::mt19937 rng_{std::random_device{}()};
};

}  // namespace bot
