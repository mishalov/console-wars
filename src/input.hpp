#pragma once
#include "types.hpp"
#include <string>
#include <vector>

enum class InputAction : uint8_t {
    None,
    MoveUp,
    MoveDown,
    MoveLeft,
    MoveRight,
    HookUp,
    HookDown,
    HookLeft,
    HookRight,
    PlaceMine,
    Quit,
};

// Parses raw input bytes into actions.
// Handles WASD, arrow keys (ESC sequences), IJKL, space, 'q'.
// Strips telnet IAC commands in-band.
// Leaves incomplete ESC sequences in buf for next call.
// Modifies buf in place, erasing consumed bytes.
std::vector<InputAction> parse_input(std::string& buf);
