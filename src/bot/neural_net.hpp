#pragma once

#include <cstddef>
#include <iosfwd>
#include <vector>

namespace bot {

// Minimal feedforward neural network.
// Default use-case: 47 -> 128 (ReLU) -> 64 (ReLU) -> 11 (linear).
// He-initialised weights, SGD with gradient clipping (global norm 1.0).
class NeuralNet {
public:
    // layer_sizes: e.g. {47, 128, 64, 11}.
    // The first element is the input dimension; the last is the output dimension.
    // Hidden layers use ReLU; the output layer is linear.
    explicit NeuralNet(std::vector<int> layer_sizes);

    // Forward pass.  input.size() must equal layer_sizes[0].
    // Returns the output-layer activations (size == layer_sizes.back()).
    [[nodiscard]] std::vector<float> forward(const std::vector<float>& input) const;

    // Single-sample SGD update via backpropagation.
    // target.size() must equal output dimension.
    // Uses MSE loss and clips gradients to global norm <= 1.0.
    void backprop(const std::vector<float>& input,
                  const std::vector<float>& target,
                  float learning_rate);

    // Deep-copy all weights and biases from another network with identical
    // topology.  Used for target-network sync in DQN.
    void copy_weights_from(const NeuralNet& other);

    // Binary serialisation (all weights + biases, prefixed with topology).
    void save(std::ostream& os) const;
    void load(std::istream& is);

    [[nodiscard]] const std::vector<int>& topology() const noexcept { return layer_sizes_; }

private:
    std::vector<int> layer_sizes_;  // e.g. {47, 128, 64, 11}

    // Per-layer parameters (index i corresponds to the connection between
    // layer i and layer i+1).  There are (num_layers - 1) parameter sets.
    struct LayerParams {
        std::vector<float> weights;  // rows = out, cols = in  (row-major)
        std::vector<float> biases;   // size = out
        int fan_in  = 0;
        int fan_out = 0;
    };
    std::vector<LayerParams> params_;

    void init_weights();
};

}  // namespace bot
