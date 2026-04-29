#include "bot_reward.hpp"
#include "bot_observation.hpp"


float compute_reward(const BotObservation& prev, const BotObservation& curr) {
    float reward = 0.0f;

    // Kill: +2.0  — terminal goal, must dominate
    if (curr.kills > prev.kills) {
        reward += 2.0f;
        // Extra reward for kill with active offensive bonus (MultiHook/SuperHook)
        if (curr.has_active_bonus && curr.features[84] > 0.5f) {
            reward += 0.5f;
        }
    }

    // Death: -1.5  — asymmetric; attacking is worth the risk
    if (prev.alive && !curr.alive) {
        reward -= 1.5f;
    }

    // Bonus pickup: +0.3
    if (curr.bonus_picked_up) {
        reward += 0.3f;
    }

    // Hook landed near own mine: +0.5  (checked before general hook-landed)
    // kEnemyNearOwnMine is now chebyshev distance / 10.0 (0 = no mine, small = close)
    // Trigger when chebyshev <= 3 (i.e., value > 0 and <= 0.3)
    // Hook landed (general): +0.15  — raw hooks aren't the goal
    if (curr.hooks_landed > prev.hooks_landed) {
        float mine_enemy_cheby = curr.features[BotObservation::kEnemyNearOwnMine];
        if (mine_enemy_cheby > 0.0f && mine_enemy_cheby <= 0.3f) {
            reward += 0.5f;
        }
        reward += 0.15f;
    }

    // Wasted hook: hook was extending, no longer extending, and no target
    // -0.15  — hooks are precious with 15-tick cooldown
    if (prev.hook_extending && !curr.hook_extending && !curr.hook_has_target) {
        reward -= 0.15f;
    }

    // Immediate penalty: hook just thrown into empty direction
    // Fires on the tick the hook starts extending. Checks whether the throw
    // direction had a valid target (enemy in ray or mine-pull opportunity)
    // at the time of the decision (prev observation).
    if (!prev.hook_extending && curr.hook_extending) {
        // Skip penalty when MultiHook/SuperHook is active (offensive bonus)
        bool has_offensive_bonus = prev.features[83] > 0.5f && prev.features[84] > 0.5f;
        if (!has_offensive_bonus) {
            // Determine direction from hook head offset in curr
            float hx = curr.features[71];  // hook head rel_x / 10.0
            float hy = curr.features[72];  // hook head rel_y / 10.0
            int dir = -1;
            if (hy < 0) dir = 0;       // Up
            else if (hy > 0) dir = 1;  // Down
            else if (hx < 0) dir = 2;  // Left
            else if (hx > 0) dir = 3;  // Right

            if (dir >= 0) {
                float enemy_in_dir = prev.features[static_cast<size_t>(54 + dir)];
                float mine_pull    = prev.features[static_cast<size_t>(58 + dir)];
                if (enemy_in_dir == 0.0f && mine_pull == 0.0f) {
                    reward -= 0.4f;
                }
            }
        }
    }

    // Mine on hook-line: bot just placed a mine and an enemy is on the same
    // row/col within range 10 with clear line (kHookLine = 62).
    // Detect mine placement: curr mine_count > prev mine_count (kMineCount = 66).
    if (curr.features[BotObservation::kMineCount] > prev.features[BotObservation::kMineCount]
        && curr.features[BotObservation::kHookLine] > 0.5f) {
        reward += 0.08f;
    }

    // Closing distance with preconditions: only when hook_ready AND
    // mine_count >= 1.  Feature 5 = hook_ready, kMineCount = mine_count/3.
    // Feature 16 = manhattan distance to nearest enemy / 58.0.
    if (curr.features[5] > 0.5f && curr.features[BotObservation::kMineCount] >= (1.0f / 3.0f - 0.01f)) {
        float prev_dist = prev.features[16];
        float curr_dist = curr.features[16];
        if (prev_dist > 0.0f && curr_dist < prev_dist) {
            reward += 0.01f;
        }
    }

    return reward;
}
