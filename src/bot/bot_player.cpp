#include "bot/bot_player.hpp"
#include "bot/dqn_brain.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

// ============================================================================
// Construction / Destruction
// ============================================================================

BotPlayer::BotPlayer(PlayerId id, const std::string& data_dir, bool inference_mode)
    : id_(id)
    , data_dir_(data_dir)
    , inference_mode_(inference_mode)
    , brain_(std::make_unique<DqnBrain>(inference_mode))
    , save_thread_(inference_mode ? std::thread{} : std::thread(&BotPlayer::save_thread_func, this))
{
    // Attempt to restore a previously saved brain.
    try {
        brain_->load(data_dir_ + "/bot_brain.bin");
    } catch (const std::runtime_error& e) {
        if (inference_mode_) {
            throw std::runtime_error(
                "Cannot run in inference mode: " + std::string(e.what()));
        }
        std::string msg = e.what();
        if (msg.find("cannot open") == std::string::npos) {
            std::cerr << "Warning: bot brain load failed (" << msg
                      << ") — starting fresh\n";
        }
    }
}

BotPlayer::~BotPlayer()
{
    save_shutdown_.store(true, std::memory_order_release);
    save_cv_.notify_one();
    if (save_thread_.joinable()) {
        save_thread_.join();
    }
}

// ============================================================================
// pre_tick()
// ============================================================================

void BotPlayer::pre_tick(GameState& state)
{
    // Find our pudge.
    const Pudge* self = nullptr;
    for (const auto& p : state.pudges()) {
        if (p.id == id_) { self = &p; break; }
    }

    if (!self || !self->alive) {
        has_prev_ = false;
        prev_self_pos_  = {};
        prev_enemy_pos_ = {};
        prev_enemy_id_  = INVALID_PLAYER;
        return;  // dead or not yet spawned -- nothing to do
    }

    // Observe (pass previous positions and enemy ID for velocity features).
    BotObservation curr_obs = observe(state, id_, prev_self_pos_, prev_enemy_pos_, prev_enemy_id_);

    // If we have a previous observation, compute the reward and accumulate
    // in the n-step buffer. (Skipped in inference mode — no learning.)
    if (!inference_mode_ && has_prev_) {
        float reward = compute_reward(prev_obs_, curr_obs);
        nstep_buffer_.push_back({prev_obs_, last_action_, reward});

        if (static_cast<int>(nstep_buffer_.size()) >= kNSteps) {
            flush_nstep_transition(curr_obs);
        }
    }

    // Decide.
    std::vector<InputAction> valid = get_valid_actions(state);
    InputAction action = brain_->decide(curr_obs, valid);

    // Inject the action into the game state.
    state.handle_input(id_, action);

    // Update stored positions and enemy ID for next tick's velocity calculation.
    prev_self_pos_ = self->pos;
    prev_enemy_id_ = curr_obs.nearest_enemy_id;
    {
        const Pudge* nearest_enemy = nullptr;
        int best_dist = INT_MAX;
        for (const auto& p : state.pudges()) {
            if (p.id == id_ || !p.alive) continue;
            int d = std::abs(p.pos.x - self->pos.x) + std::abs(p.pos.y - self->pos.y);
            if (d < best_dist) { best_dist = d; nearest_enemy = &p; }
        }
        prev_enemy_pos_ = nearest_enemy ? nearest_enemy->pos : Vec2{0, 0};
    }

    // Book-keeping for next tick.
    prev_obs_    = curr_obs;
    last_action_ = action;
    has_prev_    = true;
}

// ============================================================================
// post_tick()
// ============================================================================

void BotPlayer::post_tick(const GameState& state)
{
    // Find our pudge.
    const Pudge* self = nullptr;
    for (const auto& p : state.pudges()) {
        if (p.id == id_) { self = &p; break; }
    }

    bool currently_dead = (!self || !self->alive);

    // Detect death transition: was alive (has_prev_ implies alive on prev
    // tick), and now dead.
    if (has_prev_ && currently_dead) {
        if (!inference_mode_) {
            // Get the actual death-state observation instead of faking it.
            BotObservation terminal_obs = observe(state, id_);
            if (terminal_obs.pos == Vec2{0,0} && !terminal_obs.alive) {
                // Pudge was removed from state, use prev obs as fallback
                terminal_obs = prev_obs_;
                terminal_obs.alive = false;
                terminal_obs.features[BotObservation::kAlive] = 0.0f;
            }
            float reward = compute_reward(prev_obs_, terminal_obs);

            // Add the terminal step and flush all remaining n-step transitions
            nstep_buffer_.push_back({prev_obs_, last_action_, reward});
            flush_all_nstep(terminal_obs);

            brain_->on_game_end();

            // Throttled async save: only every Nth death to avoid I/O spikes
            ++death_count_;
            if (death_count_ >= kSaveEveryNGames) {
                death_count_ = 0;
                save_async();
            }
        }

        has_prev_ = false;
        nstep_buffer_.clear();
        prev_self_pos_  = {};
        prev_enemy_pos_ = {};
        prev_enemy_id_  = INVALID_PLAYER;
    }

    // Detect respawn transition: was dead, now alive.
    if (was_dead_ && !currently_dead) {
        has_prev_ = false;  // fresh episode
        nstep_buffer_.clear();
        prev_self_pos_  = {};
        prev_enemy_pos_ = {};
        prev_enemy_id_  = INVALID_PLAYER;
    }

    was_dead_ = currently_dead;
}

// ============================================================================
// N-step return helpers
// ============================================================================

void BotPlayer::flush_nstep_transition(const BotObservation& next_obs)
{
    // Compute n-step discounted return from the oldest entry.
    float G = 0.0f;
    float gamma_pow = 1.0f;
    for (const auto& step : nstep_buffer_) {
        G += gamma_pow * step.reward;
        gamma_pow *= kGamma;
    }
    // The oldest transition's state is the "state"; next_obs is the "next_state".
    // The brain will use gamma^n for the bootstrap value.
    auto& oldest = nstep_buffer_.front();
    brain_->on_outcome(oldest.obs, oldest.action, next_obs, G,
                       static_cast<int>(nstep_buffer_.size()));
    nstep_buffer_.pop_front();
}

void BotPlayer::flush_all_nstep(const BotObservation& terminal_obs)
{
    while (!nstep_buffer_.empty()) {
        float G = 0.0f;
        float gamma_pow = 1.0f;
        for (const auto& step : nstep_buffer_) {
            G += gamma_pow * step.reward;
            gamma_pow *= kGamma;
        }
        auto& oldest = nstep_buffer_.front();
        brain_->on_outcome(oldest.obs, oldest.action, terminal_obs, G,
                           static_cast<int>(nstep_buffer_.size()));
        nstep_buffer_.pop_front();
    }
}

// ============================================================================
// get_valid_actions()
// ============================================================================

std::vector<InputAction> BotPlayer::get_valid_actions(const GameState& state) const
{
    // Movement and idle are always valid.
    std::vector<InputAction> actions = {
        InputAction::None,
        InputAction::MoveUp,
        InputAction::MoveDown,
        InputAction::MoveLeft,
        InputAction::MoveRight,
    };

    // Find our pudge (must exist and be alive -- callers guarantee this).
    const Pudge* self = nullptr;
    for (const auto& p : state.pudges()) {
        if (p.id == id_) { self = &p; break; }
    }
    if (!self) return actions;

    // Hook actions: available when the hook is Ready, off cooldown, and the
    // pudge is not currently being pulled by an enemy hook.
    if (self->hook.state == HookState::Ready &&
        self->hook.cooldown == 0 &&
        !self->being_pulled)
    {
        actions.push_back(InputAction::HookUp);
        actions.push_back(InputAction::HookDown);
        actions.push_back(InputAction::HookLeft);
        actions.push_back(InputAction::HookRight);
    }

    // PlaceMine: available when mine cooldown is zero and the player hasn't
    // hit the mine cap.
    if (self->mine_cooldown == 0 &&
        state.mines_owned_by(id_) < MAX_MINES_PER_PLAYER)
    {
        actions.push_back(InputAction::PlaceMine);
    }

    // Never include Quit.

    return actions;
}

// ============================================================================
// save() — synchronous (for graceful shutdown)
// ============================================================================

void BotPlayer::save() const
{
    if (inference_mode_) return;
    // Shut down the background writer and wait for it to finish any in-flight
    // write, so we don't race on the output file.
    save_shutdown_.store(true, std::memory_order_release);
    save_cv_.notify_one();
    if (save_thread_.joinable()) {
        save_thread_.join();
    }
    brain_->save(data_dir_ + "/bot_brain.bin");
}

// ============================================================================
// Async save infrastructure
// ============================================================================

void BotPlayer::save_async() const
{
    // Serialize into memory buffer (fast, no I/O)
    std::ostringstream oss(std::ios::binary);
    brain_->save(oss);

    // Hand off to background writer thread
    {
        std::lock_guard<std::mutex> lock(save_mutex_);
        save_pending_ = oss.str();
    }
    save_cv_.notify_one();
}

void BotPlayer::save_thread_func() const
{
    while (true) {
        std::string data;
        {
            std::unique_lock<std::mutex> lock(save_mutex_);
            save_cv_.wait(lock, [this] {
                return !save_pending_.empty() ||
                       save_shutdown_.load(std::memory_order_acquire);
            });

            if (save_shutdown_.load(std::memory_order_acquire) &&
                save_pending_.empty()) {
                return;
            }
            data = std::move(save_pending_);
            save_pending_.clear();
        }

        // Write to temp file + rename for atomicity
        std::string path = data_dir_ + "/bot_brain.bin";
        std::string tmp_path = path + ".tmp";

        std::ofstream ofs(tmp_path, std::ios::binary);
        if (ofs) {
            ofs.write(data.data(), static_cast<std::streamsize>(data.size()));
            ofs.close();
            if (ofs) {
                std::rename(tmp_path.c_str(), path.c_str());
            }
        }
    }
}
