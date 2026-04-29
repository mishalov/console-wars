#include "bot/neural_net.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>

namespace bot {

// ---------------------------------------------------------------------------
// Construction / initialisation
// ---------------------------------------------------------------------------

NeuralNet::NeuralNet(std::vector<int> layer_sizes)
    : layer_sizes_(std::move(layer_sizes))
{
    if (layer_sizes_.size() < 2) {
        throw std::invalid_argument("NeuralNet requires at least 2 layers");
    }
    for (auto s : layer_sizes_) {
        if (s <= 0) {
            throw std::invalid_argument("Layer size must be > 0");
        }
    }

    const auto num_connections = layer_sizes_.size() - 1;
    params_.resize(num_connections);

    for (std::size_t i = 0; i < num_connections; ++i) {
        auto& p   = params_[i];
        p.fan_in  = layer_sizes_[i];
        p.fan_out = layer_sizes_[i + 1];
        p.weights.resize(static_cast<std::size_t>(p.fan_out) *
                         static_cast<std::size_t>(p.fan_in));
        p.biases.assign(static_cast<std::size_t>(p.fan_out), 0.0f);

        // Zero-initialize Adam optimizer state
        p.m_weights.assign(p.weights.size(), 0.0f);
        p.m_biases.assign(p.biases.size(), 0.0f);
        p.v_weights.assign(p.weights.size(), 0.0f);
        p.v_biases.assign(p.biases.size(), 0.0f);
    }

    init_weights();
}

void NeuralNet::init_weights()
{
    std::mt19937 rng{42};  // deterministic seed for reproducibility

    for (auto& p : params_) {
        // He initialisation: N(0, sqrt(2 / fan_in))
        const float stddev =
            std::sqrt(2.0f / static_cast<float>(p.fan_in));
        std::normal_distribution<float> dist(0.0f, stddev);

        for (auto& w : p.weights) {
            w = dist(rng);
        }
        // biases stay at 0
    }
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

std::vector<float> NeuralNet::forward(const std::vector<float>& input) const
{
    assert(static_cast<int>(input.size()) == layer_sizes_.front());

    // Current activation vector -- starts as the input.
    std::vector<float> act = input;

    const auto num_layers = params_.size();
    for (std::size_t li = 0; li < num_layers; ++li) {
        const auto& p = params_[li];
        const auto out_sz = static_cast<std::size_t>(p.fan_out);
        const auto in_sz  = static_cast<std::size_t>(p.fan_in);

        std::vector<float> next(out_sz);

        for (std::size_t o = 0; o < out_sz; ++o) {
            float sum = p.biases[o];
            const float* row = p.weights.data() + o * in_sz;
            for (std::size_t j = 0; j < in_sz; ++j) {
                sum += row[j] * act[j];
            }
            next[o] = sum;
        }

        // Activation: ReLU for hidden layers, linear for the output layer.
        const bool is_output = (li == num_layers - 1);
        if (!is_output) {
            for (auto& v : next) {
                v = std::max(v, 0.0f);
            }
        }

        act = std::move(next);
    }

    return act;
}

// ---------------------------------------------------------------------------
// Backpropagation (single-sample SGD with gradient clipping)
// ---------------------------------------------------------------------------

void NeuralNet::backprop(const std::vector<float>& input,
                         const std::vector<float>& target,
                         float learning_rate)
{
    assert(static_cast<int>(input.size()) == layer_sizes_.front());
    assert(static_cast<int>(target.size()) == layer_sizes_.back());

    const auto num_connections = params_.size();

    // ------ Forward pass (cache pre-activations and activations) ----------

    // activations[0] = input,  activations[i+1] = output of layer i
    std::vector<std::vector<float>> activations(num_connections + 1);
    // pre_activations[i] = z for connection i (before activation)
    std::vector<std::vector<float>> pre_acts(num_connections);

    activations[0] = input;

    for (std::size_t li = 0; li < num_connections; ++li) {
        const auto& p = params_[li];
        const auto out_sz = static_cast<std::size_t>(p.fan_out);
        const auto in_sz  = static_cast<std::size_t>(p.fan_in);

        std::vector<float> z(out_sz);
        const auto& prev = activations[li];

        for (std::size_t o = 0; o < out_sz; ++o) {
            float sum = p.biases[o];
            const float* row = p.weights.data() + o * in_sz;
            for (std::size_t j = 0; j < in_sz; ++j) {
                sum += row[j] * prev[j];
            }
            z[o] = sum;
        }

        pre_acts[li] = z;

        // Apply activation.
        const bool is_output = (li == num_connections - 1);
        if (!is_output) {
            for (auto& v : z) {
                v = std::max(v, 0.0f);
            }
        }
        activations[li + 1] = std::move(z);
    }

    // ------ Backward pass -------------------------------------------------

    // Per-layer weight and bias gradients (same layout as params_).
    struct LayerGrad {
        std::vector<float> dw;  // same size as weights
        std::vector<float> db;  // same size as biases
    };
    std::vector<LayerGrad> grads(num_connections);

    // delta for current layer (propagated backwards).
    std::vector<float> delta;

    // Output layer delta: dL/dz = (output - target) for MSE loss.
    {
        const auto& output = activations.back();
        const auto sz = output.size();
        delta.resize(sz);
        for (std::size_t i = 0; i < sz; ++i) {
            delta[i] = output[i] - target[i];
        }
    }

    // Walk layers from output towards input.
    for (std::size_t li = num_connections; li-- > 0;) {
        const auto& p    = params_[li];
        const auto out_sz = static_cast<std::size_t>(p.fan_out);
        const auto in_sz  = static_cast<std::size_t>(p.fan_in);

        auto& g = grads[li];
        g.dw.resize(p.weights.size());
        g.db.resize(p.biases.size());

        const auto& act_prev = activations[li];

        // Compute weight and bias gradients for this layer.
        for (std::size_t o = 0; o < out_sz; ++o) {
            g.db[o] = delta[o];
            float* dw_row = g.dw.data() + o * in_sz;
            for (std::size_t j = 0; j < in_sz; ++j) {
                dw_row[j] = delta[o] * act_prev[j];
            }
        }

        // Propagate delta to previous layer (skip if we reached the input).
        if (li > 0) {
            std::vector<float> prev_delta(in_sz, 0.0f);
            for (std::size_t o = 0; o < out_sz; ++o) {
                const float* w_row = p.weights.data() + o * in_sz;
                for (std::size_t j = 0; j < in_sz; ++j) {
                    prev_delta[j] += w_row[j] * delta[o];
                }
            }
            // Element-wise multiply by ReLU derivative of the pre-activation.
            const auto& z = pre_acts[li - 1];
            for (std::size_t j = 0; j < in_sz; ++j) {
                prev_delta[j] *= (z[j] > 0.0f) ? 1.0f : 0.0f;
            }
            delta = std::move(prev_delta);
        }
    }

    // ------ Gradient clipping (global L2 norm, max 1.0) -------------------

    float global_norm_sq = 0.0f;
    for (const auto& g : grads) {
        for (float v : g.dw) global_norm_sq += v * v;
        for (float v : g.db) global_norm_sq += v * v;
    }
    const float global_norm = std::sqrt(global_norm_sq);
    constexpr float kMaxNorm = 1.0f;

    float clip_coeff = 1.0f;
    if (global_norm > kMaxNorm) {
        clip_coeff = kMaxNorm / global_norm;
    }

    // ------ Parameter update (Adam) -----------------------------------------

    ++adam_t_;
    constexpr float beta1 = 0.9f;
    constexpr float beta2 = 0.999f;
    constexpr float eps   = 1e-8f;

    float bc1 = 1.0f - std::pow(beta1, static_cast<float>(adam_t_));
    float bc2 = 1.0f - std::pow(beta2, static_cast<float>(adam_t_));

    for (std::size_t li = 0; li < num_connections; ++li) {
        auto& p = params_[li];
        const auto& g = grads[li];

        for (std::size_t i = 0; i < p.weights.size(); ++i) {
            float grad = clip_coeff * g.dw[i];
            p.m_weights[i] = beta1 * p.m_weights[i] + (1.0f - beta1) * grad;
            p.v_weights[i] = beta2 * p.v_weights[i] + (1.0f - beta2) * grad * grad;
            float m_hat = p.m_weights[i] / bc1;
            float v_hat = p.v_weights[i] / bc2;
            p.weights[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
        }

        for (std::size_t i = 0; i < p.biases.size(); ++i) {
            float grad = clip_coeff * g.db[i];
            p.m_biases[i] = beta1 * p.m_biases[i] + (1.0f - beta1) * grad;
            p.v_biases[i] = beta2 * p.v_biases[i] + (1.0f - beta2) * grad * grad;
            float m_hat = p.m_biases[i] / bc1;
            float v_hat = p.v_biases[i] / bc2;
            p.biases[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
        }
    }
}

// ---------------------------------------------------------------------------
// Weight copying
// ---------------------------------------------------------------------------

void NeuralNet::copy_weights_from(const NeuralNet& other)
{
    if (layer_sizes_ != other.layer_sizes_) {
        throw std::invalid_argument(
            "copy_weights_from: topology mismatch");
    }
    for (std::size_t i = 0; i < params_.size(); ++i) {
        params_[i].weights = other.params_[i].weights;
        params_[i].biases  = other.params_[i].biases;
    }
    // Copy Adam timestep (for serialisation consistency) but NOT the moment
    // vectors -- the target network does not train.
    adam_t_ = other.adam_t_;
}

void NeuralNet::soft_update(const NeuralNet& source, float tau)
{
    if (layer_sizes_ != source.layer_sizes_) {
        throw std::invalid_argument("soft_update: topology mismatch");
    }
    for (std::size_t i = 0; i < params_.size(); ++i) {
        for (std::size_t j = 0; j < params_[i].weights.size(); ++j) {
            params_[i].weights[j] = tau * source.params_[i].weights[j]
                                  + (1.0f - tau) * params_[i].weights[j];
        }
        for (std::size_t j = 0; j < params_[i].biases.size(); ++j) {
            params_[i].biases[j] = tau * source.params_[i].biases[j]
                                 + (1.0f - tau) * params_[i].biases[j];
        }
    }
}

// ---------------------------------------------------------------------------
// Serialisation (binary)
// ---------------------------------------------------------------------------

void NeuralNet::save(std::ostream& os) const
{
    // Write topology.
    auto num_layers = static_cast<uint32_t>(layer_sizes_.size());
    os.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    for (auto s : layer_sizes_) {
        auto val = static_cast<int32_t>(s);
        os.write(reinterpret_cast<const char*>(&val), sizeof(val));
    }

    // Write weights, biases, and Adam state for each connection.
    for (const auto& p : params_) {
        os.write(reinterpret_cast<const char*>(p.weights.data()),
                 static_cast<std::streamsize>(p.weights.size() * sizeof(float)));
        os.write(reinterpret_cast<const char*>(p.biases.data()),
                 static_cast<std::streamsize>(p.biases.size() * sizeof(float)));
        os.write(reinterpret_cast<const char*>(p.m_weights.data()),
                 static_cast<std::streamsize>(p.m_weights.size() * sizeof(float)));
        os.write(reinterpret_cast<const char*>(p.m_biases.data()),
                 static_cast<std::streamsize>(p.m_biases.size() * sizeof(float)));
        os.write(reinterpret_cast<const char*>(p.v_weights.data()),
                 static_cast<std::streamsize>(p.v_weights.size() * sizeof(float)));
        os.write(reinterpret_cast<const char*>(p.v_biases.data()),
                 static_cast<std::streamsize>(p.v_biases.size() * sizeof(float)));
    }

    // Write Adam timestep.
    auto adam_t_val = static_cast<int32_t>(adam_t_);
    os.write(reinterpret_cast<const char*>(&adam_t_val), sizeof(adam_t_val));
}

void NeuralNet::load(std::istream& is)
{
    // Read and verify topology.
    uint32_t num_layers = 0;
    is.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

    std::vector<int> topo(num_layers);
    for (uint32_t i = 0; i < num_layers; ++i) {
        int32_t val = 0;
        is.read(reinterpret_cast<char*>(&val), sizeof(val));
        topo[i] = static_cast<int>(val);
    }

    if (topo != layer_sizes_) {
        throw std::runtime_error(
            "NeuralNet::load: saved topology does not match current topology");
    }

    // Read weights, biases, and Adam state.
    for (auto& p : params_) {
        is.read(reinterpret_cast<char*>(p.weights.data()),
                static_cast<std::streamsize>(p.weights.size() * sizeof(float)));
        is.read(reinterpret_cast<char*>(p.biases.data()),
                static_cast<std::streamsize>(p.biases.size() * sizeof(float)));
        is.read(reinterpret_cast<char*>(p.m_weights.data()),
                static_cast<std::streamsize>(p.m_weights.size() * sizeof(float)));
        is.read(reinterpret_cast<char*>(p.m_biases.data()),
                static_cast<std::streamsize>(p.m_biases.size() * sizeof(float)));
        is.read(reinterpret_cast<char*>(p.v_weights.data()),
                static_cast<std::streamsize>(p.v_weights.size() * sizeof(float)));
        is.read(reinterpret_cast<char*>(p.v_biases.data()),
                static_cast<std::streamsize>(p.v_biases.size() * sizeof(float)));
    }

    // Read Adam timestep.
    int32_t adam_t_val = 0;
    is.read(reinterpret_cast<char*>(&adam_t_val), sizeof(adam_t_val));
    adam_t_ = static_cast<int>(adam_t_val);

    if (!is) {
        throw std::runtime_error("NeuralNet::load: stream error during read");
    }
}

}  // namespace bot
