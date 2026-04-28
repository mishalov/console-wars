#include "bot/bot_player.hpp"
#include "bot/dqn_brain.hpp"

#include <iostream>
#include <stdexcept>

// ============================================================================
// Construction
// ============================================================================

BotPlayer::BotPlayer(PlayerId id, const std::string& data_dir)
    : id_(id)
    , data_dir_(data_dir)
    , brain_(std::make_unique<DqnBrain>())
{
    // Attempt to restore a previously saved brain.
    try {
        brain_->load(data_dir_ + "/bot_brain.bin");
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        // "cannot open" means first run — expected, stay silent.
        // Anything else (bad magic, topology mismatch, truncation) is worth
        // reporting so the user knows the bot started fresh unexpectedly.
        if (msg.find("cannot open") == std::string::npos) {
            std::cerr << "Warning: bot brain load failed (" << msg
                      << ") — starting fresh\n";
        }
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
        return;  // dead or not yet spawned -- nothing to do
    }

    // Observe.
    BotObservation curr_obs = observe(state, id_);

    // If we have a previous observation, compute the reward and feed it back.
    if (has_prev_) {
        float reward = compute_reward(prev_obs_, curr_obs);
        brain_->on_outcome(prev_obs_, last_action_, curr_obs, reward);
    }

    // Decide.
    std::vector<InputAction> valid = get_valid_actions(state);
    InputAction action = brain_->decide(curr_obs, valid);

    // Inject the action into the game state.
    state.handle_input(id_, action);

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
        // Feed a terminal transition to the brain before signalling game end.
        BotObservation terminal_obs = prev_obs_;
        terminal_obs.alive = false;
        terminal_obs.features[BotObservation::kAlive] = 0.0f;
        float reward = compute_reward(prev_obs_, terminal_obs);
        brain_->on_outcome(prev_obs_, last_action_, terminal_obs, reward);

        brain_->on_game_end();
        has_prev_ = false;

        // Periodic auto-save after every game end
        try {
            save();
        } catch (const std::runtime_error&) {
            // Non-fatal — best effort persistence
        }
    }

    // Detect respawn transition: was dead, now alive.
    if (was_dead_ && !currently_dead) {
        has_prev_ = false;  // fresh episode
    }

    was_dead_ = currently_dead;
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
// save()
// ============================================================================

void BotPlayer::save() const
{
    brain_->save(data_dir_ + "/bot_brain.bin");
}
