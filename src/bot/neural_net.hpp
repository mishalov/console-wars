#pragma once

#include <cstddef>
#include <iosfwd>
#include <vector>

namespace bot {

// Minimal feedforward neural network with pre-allocated workspace buffers.
// Default use-case: 92 -> 256 -> 128 -> 64 -> 10.
// He-initialised weights, Adam optimizer with gradient clipping (global norm 1.0).
//
// Performance: forward() and backprop() use mutable pre-allocated buffers to
// avoid per-call heap allocations. The forward() signature remains const via
// mutable workspace members.
class NeuralNet {
public:
    // layer_sizes: e.g. {92, 256, 128, 64, 10}.
    // The first element is the input dimension; the last is the output dimension.
    // Hidden layers use ReLU; the output layer is linear.
    explicit NeuralNet(std::vector<int> layer_sizes);

    // Forward pass from a vector.  input.size() must equal layer_sizes[0].
    // Returns the output-layer activations (size == layer_sizes.back()).
    // Uses pre-allocated workspace buffers internally (no heap allocations).
    [[nodiscard]] std::vector<float> forward(const std::vector<float>& input) const;

    // Forward pass from a raw pointer. Avoids vector copy overhead.
    // data must point to at least `size` floats, where size == layer_sizes[0].
    [[nodiscard]] std::vector<float> forward(const float* data, int size) const;

    // Forward pass that writes output into a caller-supplied buffer.
    // out_buf must have space for at least layer_sizes.back() floats.
    // Returns number of output floats written.
    int forward_to(const float* data, int size, float* out_buf) const;

    // Single-sample update via backpropagation (Adam optimizer).
    // target.size() must equal output dimension.
    // Uses MSE loss and clips gradients to global norm <= 1.0.
    void backprop(const std::vector<float>& input,
                  const std::vector<float>& target,
                  float learning_rate);

    // Backprop from pre-computed activations (avoids redundant forward pass).
    // `cached_activations` must contain (num_layers) vectors:
    //   [0] = input, [1] = hidden1 output, ..., [N-1] = network output.
    // `target` is the desired output (same as in backprop()).
    // NOTE: This overload reads pre-activations from the internal ws_pre_acts_ cache.
    // Only correct if no other forward pass has been issued since the forward() call
    // that produced these cached_activations.
    void backprop_from_cached(const std::vector<std::vector<float>>& cached_activations,
                              const std::vector<float>& target,
                              float learning_rate);

    // Backprop from pre-computed activations AND explicit pre-activations.
    // Use this overload when multiple forward passes have been interleaved and the
    // internal ws_pre_acts_ cache no longer corresponds to `cached_activations`.
    // `pre_activations` must contain (num_layers - 1) vectors (one per connection).
    void backprop_from_cached(const std::vector<std::vector<float>>& cached_activations,
                              const std::vector<std::vector<float>>& pre_activations,
                              const std::vector<float>& target,
                              float learning_rate);

    // Deep-copy all weights and biases from another network with identical
    // topology.  Used for target-network sync in DQN.
    void copy_weights_from(const NeuralNet& other);

    // Polyak-averaging (soft) target update:
    //   this = tau * source + (1 - tau) * this
    // Used for smooth target-network updates in DQN.
    void soft_update(const NeuralNet& source, float tau);

    // Binary serialisation (all weights + biases, prefixed with topology).
    void save(std::ostream& os) const;
    void load(std::istream& is);

    [[nodiscard]] const std::vector<int>& topology() const noexcept { return layer_sizes_; }

    // Access cached activations from the most recent forward pass.
    // Valid only after calling forward() or forward_to().
    // Contains (num_layers) entries: [0]=input copy, [1..N-1]=layer outputs.
    [[nodiscard]] const std::vector<std::vector<float>>& cached_activations() const noexcept {
        return ws_activations_;
    }

    // Access cached pre-activations (z values before activation function) from
    // the most recent forward pass. Valid only after calling forward() or forward_to().
    // Contains (num_layers - 1) entries: ws_pre_acts_[i] = z for connection i.
    [[nodiscard]] const std::vector<std::vector<float>>& cached_pre_activations() const noexcept {
        return ws_pre_acts_;
    }

private:
    std::vector<int> layer_sizes_;  // e.g. {92, 256, 128, 64, 10}

    // Per-layer parameters (index i corresponds to the connection between
    // layer i and layer i+1).  There are (num_layers - 1) parameter sets.
    struct LayerParams {
        std::vector<float> weights;  // rows = out, cols = in  (row-major)
        std::vector<float> biases;   // size = out
        int fan_in  = 0;
        int fan_out = 0;

        // Adam optimizer state
        std::vector<float> m_weights;  // 1st moment (momentum)
        std::vector<float> m_biases;
        std::vector<float> v_weights;  // 2nd moment (RMS)
        std::vector<float> v_biases;
    };
    std::vector<LayerParams> params_;

    int adam_t_ = 0;  // Adam timestep for bias correction

    // --- Pre-allocated workspace buffers (mutable for const forward()) -------

    // Forward pass: two alternating buffers sized to max layer dimension.
    mutable std::vector<float> ws_buf_a_;
    mutable std::vector<float> ws_buf_b_;

    // Cached activations from most recent forward pass (for backprop_from_cached).
    // activations_[0] = input, activations_[i+1] = output of layer i.
    mutable std::vector<std::vector<float>> ws_activations_;

    // Cached pre-activations (z values before activation function) from most
    // recent forward pass.  Used by backprop_from_cached() to avoid recomputing
    // GEMV products.  ws_pre_acts_[i] = z values for connection i.
    mutable std::vector<std::vector<float>> ws_pre_acts_;

    // Backprop workspace (pre-allocated once, reused across calls).
    std::vector<std::vector<float>> bp_pre_acts_;  // pre-activation values per layer
    std::vector<float> bp_delta_;                   // current delta vector
    std::vector<float> bp_prev_delta_;              // previous layer delta
    struct LayerGrad {
        std::vector<float> dw;
        std::vector<float> db;
    };
    std::vector<LayerGrad> bp_grads_;              // per-layer gradients

    void init_weights();
    void allocate_workspace();

    // Internal forward pass implementation. Reads from ws_buf_a_ (input already copied there).
    // Writes output into ws_buf_a_. Caches activations if cache_activations is true.
    void forward_impl(const float* input, int input_size, bool cache_activations) const;

    // Internal backprop using pre-computed activations.
    void backprop_impl(const std::vector<std::vector<float>>& activations,
                       const std::vector<float>& target,
                       float learning_rate);
};

}  // namespace bot
