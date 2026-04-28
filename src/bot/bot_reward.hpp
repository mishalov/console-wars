#pragma once

struct BotObservation;

/// Compute a scalar reward from two consecutive observations.
/// The result is clipped to [-1.0, +1.0].
float compute_reward(const BotObservation& prev, const BotObservation& curr);
