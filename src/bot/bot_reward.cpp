#include "bot_reward.hpp"
#include "bot_observation.hpp"

#include <algorithm>

float compute_reward(const BotObservation& prev, const BotObservation& curr) {
    float reward = 0.0f;

    // Kill: +1.0
    if (curr.kills > prev.kills) {
        reward += 1.0f;
    }

    // Death: -1.0
    if (prev.alive && !curr.alive) {
        reward -= 1.0f;
    }

    // Hook landed near own mine: +0.5  (checked before general hook-landed)
    // Hook landed (general): +0.3
    if (curr.hooks_landed > prev.hooks_landed) {
        if (curr.features[BotObservation::kEnemyNearOwnMine] > 0.5f) {
            reward += 0.5f;
        }
        reward += 0.3f;
    }

    // Wasted hook: hook was extending, no longer extending, and no target
    if (prev.hook_extending && !curr.hook_extending && !curr.hook_has_target) {
        reward -= 0.05f;
    }

    // Alive tick: small positive reward for staying alive
    if (curr.alive) {
        reward += 0.001f;
    }

    // In danger zone: any directional enemy-mine signal active
    for (size_t i = BotObservation::kDangerUpStart; i <= BotObservation::kDangerRightEnd; ++i) {
        if (curr.features[i] > 0.5f) {
            reward -= 0.02f;
            break;  // apply penalty once, not per direction
        }
    }

    // Clip to [-1.0, +1.0]
    return std::clamp(reward, -1.0f, 1.0f);
}
