#pragma once

struct BotObservation;

/// Compute a scalar reward from two consecutive observations.
float compute_reward(const BotObservation& prev, const BotObservation& curr);
