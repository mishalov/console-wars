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
    allocate_workspace();
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

void NeuralNet::allocate_workspace()
{
    // Find maximum layer dimension for alternating buffers.
    int max_dim = 0;
    for (auto s : layer_sizes_) {
        max_dim = std::max(max_dim, s);
    }
    ws_buf_a_.resize(static_cast<std::size_t>(max_dim));
    ws_buf_b_.resize(static_cast<std::size_t>(max_dim));

    // Pre-allocate activation cache (one vector per layer).
    const auto num_layers = layer_sizes_.size();
    ws_activations_.resize(num_layers);
    for (std::size_t i = 0; i < num_layers; ++i) {
        ws_activations_[i].resize(static_cast<std::size_t>(layer_sizes_[i]));
    }

    // Pre-allocate pre-activation cache (one vector per connection).
    const auto num_conn = num_layers - 1;
    ws_pre_acts_.resize(num_conn);
    for (std::size_t i = 0; i < num_conn; ++i) {
        ws_pre_acts_[i].resize(static_cast<std::size_t>(layer_sizes_[i + 1]));
    }

    // Pre-allocate backprop workspace.
    const auto num_connections = params_.size();
    bp_pre_acts_.resize(num_connections);
    for (std::size_t i = 0; i < num_connections; ++i) {
        bp_pre_acts_[i].resize(static_cast<std::size_t>(params_[i].fan_out));
    }

    bp_delta_.resize(static_cast<std::size_t>(max_dim));
    bp_prev_delta_.resize(static_cast<std::size_t>(max_dim));

    bp_grads_.resize(num_connections);
    for (std::size_t i = 0; i < num_connections; ++i) {
        bp_grads_[i].dw.resize(params_[i].weights.size());
        bp_grads_[i].db.resize(params_[i].biases.size());
    }
}

// ---------------------------------------------------------------------------
// Forward pass (internal implementation)
// ---------------------------------------------------------------------------

void NeuralNet::forward_impl(const float* input, int input_size, bool cache_activations) const
{
    assert(input_size == layer_sizes_.front());

    const auto num_connections = params_.size();

    // Copy input into workspace buffer A.
    std::memcpy(ws_buf_a_.data(), input, static_cast<std::size_t>(input_size) * sizeof(float));

    // Cache input activations if requested.
    if (cache_activations) {
        std::memcpy(ws_activations_[0].data(), input,
                    static_cast<std::size_t>(input_size) * sizeof(float));
    }

    // Alternate between buf_a (current input) and buf_b (current output).
    float* cur_in  = ws_buf_a_.data();
    float* cur_out = ws_buf_b_.data();

    for (std::size_t li = 0; li < num_connections; ++li) {
        const auto& p = params_[li];
        const auto out_sz = static_cast<std::size_t>(p.fan_out);
        const auto in_sz  = static_cast<std::size_t>(p.fan_in);

        for (std::size_t o = 0; o < out_sz; ++o) {
            float sum = p.biases[o];
            const float* row = p.weights.data() + o * in_sz;
            for (std::size_t j = 0; j < in_sz; ++j) {
                sum += row[j] * cur_in[j];
            }
            cur_out[o] = sum;
        }

        // Cache pre-activations (z values) before applying activation function.
        if (cache_activations) {
            std::memcpy(ws_pre_acts_[li].data(), cur_out, out_sz * sizeof(float));
        }

        // Activation: ReLU for hidden layers, linear for the output layer.
        const bool is_output = (li == num_connections - 1);
        if (!is_output) {
            for (std::size_t i = 0; i < out_sz; ++i) {
                cur_out[i] = std::max(cur_out[i], 0.0f);
            }
        }

        // Cache activations if requested.
        if (cache_activations) {
            std::memcpy(ws_activations_[li + 1].data(), cur_out, out_sz * sizeof(float));
        }

        // Swap buffers: output becomes input for next layer.
        std::swap(cur_in, cur_out);
    }

    // After the loop, cur_in points to the final output (due to swap after last layer).
    // If the number of layers is even, result is in buf_a; if odd, in buf_b.
    // We always want the result accessible from ws_buf_a_ for the return path.
    if (cur_in != ws_buf_a_.data()) {
        const auto out_sz = static_cast<std::size_t>(layer_sizes_.back());
        std::memcpy(ws_buf_a_.data(), cur_in, out_sz * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// Forward pass (public API)
// ---------------------------------------------------------------------------

std::vector<float> NeuralNet::forward(const std::vector<float>& input) const
{
    assert(static_cast<int>(input.size()) == layer_sizes_.front());
    forward_impl(input.data(), static_cast<int>(input.size()), true);

    const auto out_sz = static_cast<std::size_t>(layer_sizes_.back());
    return std::vector<float>(ws_buf_a_.data(), ws_buf_a_.data() + out_sz);
}

std::vector<float> NeuralNet::forward(const float* data, int size) const
{
    forward_impl(data, size, true);

    const auto out_sz = static_cast<std::size_t>(layer_sizes_.back());
    return std::vector<float>(ws_buf_a_.data(), ws_buf_a_.data() + out_sz);
}

int NeuralNet::forward_to(const float* data, int size, float* out_buf) const
{
    forward_impl(data, size, true);

    const int out_sz = layer_sizes_.back();
    std::memcpy(out_buf, ws_buf_a_.data(), static_cast<std::size_t>(out_sz) * sizeof(float));
    return out_sz;
}

// ---------------------------------------------------------------------------
// Backpropagation (single-sample, Adam optimizer, gradient clipping)
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
    // Reuse ws_activations_ for activations.
    std::memcpy(ws_activations_[0].data(), input.data(),
                input.size() * sizeof(float));

    for (std::size_t li = 0; li < num_connections; ++li) {
        const auto& p = params_[li];
        const auto out_sz = static_cast<std::size_t>(p.fan_out);
        const auto in_sz  = static_cast<std::size_t>(p.fan_in);

        const float* prev = ws_activations_[li].data();
        float* z = bp_pre_acts_[li].data();

        for (std::size_t o = 0; o < out_sz; ++o) {
            float sum = p.biases[o];
            const float* row = p.weights.data() + o * in_sz;
            for (std::size_t j = 0; j < in_sz; ++j) {
                sum += row[j] * prev[j];
            }
            z[o] = sum;
        }

        // Apply activation and store in ws_activations_[li+1].
        float* act_out = ws_activations_[li + 1].data();
        const bool is_output = (li == num_connections - 1);
        if (!is_output) {
            for (std::size_t i = 0; i < out_sz; ++i) {
                act_out[i] = std::max(z[i], 0.0f);
            }
        } else {
            std::memcpy(act_out, z, out_sz * sizeof(float));
        }
    }

    // Delegate to the shared backprop implementation.
    backprop_impl(ws_activations_, target, learning_rate);
}

void NeuralNet::backprop_from_cached(const std::vector<std::vector<float>>& cached_activations,
                                     const std::vector<float>& target,
                                     float learning_rate)
{
    assert(cached_activations.size() == layer_sizes_.size());
    assert(static_cast<int>(target.size()) == layer_sizes_.back());

    // Use the pre-activations that were cached during the forward pass that
    // produced these activations.  This eliminates the GEMV recomputation that
    // was the original bottleneck.
    const auto num_connections = params_.size();
    for (std::size_t li = 0; li < num_connections; ++li) {
        std::memcpy(bp_pre_acts_[li].data(), ws_pre_acts_[li].data(),
                    ws_pre_acts_[li].size() * sizeof(float));
    }

    backprop_impl(cached_activations, target, learning_rate);
}

void NeuralNet::backprop_from_cached(const std::vector<std::vector<float>>& cached_activations,
                                     const std::vector<std::vector<float>>& pre_activations,
                                     const std::vector<float>& target,
                                     float learning_rate)
{
    assert(cached_activations.size() == layer_sizes_.size());
    assert(pre_activations.size() == params_.size());
    assert(static_cast<int>(target.size()) == layer_sizes_.back());

    // Copy the caller-supplied pre-activations into bp_pre_acts_ for use by backprop_impl.
    const auto num_connections = params_.size();
    for (std::size_t li = 0; li < num_connections; ++li) {
        assert(pre_activations[li].size() == bp_pre_acts_[li].size());
        std::memcpy(bp_pre_acts_[li].data(), pre_activations[li].data(),
                    pre_activations[li].size() * sizeof(float));
    }

    backprop_impl(cached_activations, target, learning_rate);
}

void NeuralNet::backprop_impl(const std::vector<std::vector<float>>& activations,
                              const std::vector<float>& target,
                              float learning_rate)
{
    const auto num_connections = params_.size();

    // ------ Backward pass -------------------------------------------------

    // Output layer delta: dL/dz = (output - target) for MSE loss.
    const auto& output = activations.back();
    const auto out_dim = static_cast<std::size_t>(layer_sizes_.back());
    for (std::size_t i = 0; i < out_dim; ++i) {
        bp_delta_[i] = output[i] - target[i];
    }

    // Walk layers from output towards input.
    for (std::size_t li = num_connections; li-- > 0;) {
        const auto& p    = params_[li];
        const auto out_sz = static_cast<std::size_t>(p.fan_out);
        const auto in_sz  = static_cast<std::size_t>(p.fan_in);

        auto& g = bp_grads_[li];
        const float* act_prev = activations[li].data();

        // Compute weight and bias gradients for this layer.
        for (std::size_t o = 0; o < out_sz; ++o) {
            g.db[o] = bp_delta_[o];
            float* dw_row = g.dw.data() + o * in_sz;
            const float d = bp_delta_[o];
            for (std::size_t j = 0; j < in_sz; ++j) {
                dw_row[j] = d * act_prev[j];
            }
        }

        // Propagate delta to previous layer (skip if we reached the input).
        if (li > 0) {
            // Zero out prev_delta.
            std::memset(bp_prev_delta_.data(), 0, in_sz * sizeof(float));

            for (std::size_t o = 0; o < out_sz; ++o) {
                const float* w_row = p.weights.data() + o * in_sz;
                const float d = bp_delta_[o];
                for (std::size_t j = 0; j < in_sz; ++j) {
                    bp_prev_delta_[j] += w_row[j] * d;
                }
            }
            // Element-wise multiply by ReLU derivative of the pre-activation.
            const float* z = bp_pre_acts_[li - 1].data();
            for (std::size_t j = 0; j < in_sz; ++j) {
                bp_prev_delta_[j] *= (z[j] > 0.0f) ? 1.0f : 0.0f;
            }
            // Swap delta and prev_delta (avoid copy).
            std::swap(bp_delta_, bp_prev_delta_);
        }
    }

    // ------ Gradient clipping (global L2 norm, max 1.0) -------------------

    float global_norm_sq = 0.0f;
    for (std::size_t li = 0; li < num_connections; ++li) {
        const auto& g = bp_grads_[li];
        for (std::size_t i = 0; i < g.dw.size(); ++i) {
            global_norm_sq += g.dw[i] * g.dw[i];
        }
        for (std::size_t i = 0; i < g.db.size(); ++i) {
            global_norm_sq += g.db[i] * g.db[i];
        }
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
        const auto& g = bp_grads_[li];

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
// Serialisation (binary) -- format unchanged for compatibility
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
