#include "input.hpp"

static constexpr unsigned char IAC  = 0xFF;
static constexpr unsigned char SB   = 0xFA;
static constexpr unsigned char SE   = 0xF0;
static constexpr unsigned char WILL = 0xFB;
static constexpr unsigned char WONT = 0xFC;
static constexpr unsigned char DO_  = 0xFD;
static constexpr unsigned char DONT = 0xFE;

std::vector<InputAction> parse_input(std::string& buf) {
    std::vector<InputAction> actions;
    size_t pos = 0;
    const size_t len = buf.size();

    while (pos < len) {
        auto ch = static_cast<unsigned char>(buf[pos]);

        // Telnet IAC handling
        if (ch == IAC) {
            if (pos + 1 >= len) {
                // Incomplete IAC — leave in buffer
                break;
            }
            auto next = static_cast<unsigned char>(buf[pos + 1]);
            if (next == WILL || next == WONT || next == DO_ || next == DONT) {
                // 3-byte command: IAC WILL/WONT/DO/DONT <option>
                if (pos + 2 >= len) {
                    break;  // Incomplete
                }
                pos += 3;
                continue;
            } else if (next == SB) {
                // Subnegotiation: scan for IAC SE (0xFF 0xF0)
                size_t scan = pos + 2;
                bool found = false;
                while (scan + 1 < len) {
                    if (static_cast<unsigned char>(buf[scan]) == IAC &&
                        static_cast<unsigned char>(buf[scan + 1]) == SE) {
                        pos = scan + 2;
                        found = true;
                        break;
                    }
                    ++scan;
                }
                if (!found) {
                    break;  // Incomplete subneg — leave in buffer
                }
                continue;
            } else if (next == IAC) {
                // Escaped 0xFF byte — skip both
                pos += 2;
                continue;
            } else {
                // Other 2-byte IAC commands (e.g., IAC NOP)
                pos += 2;
                continue;
            }
        }

        // ESC sequence handling (arrow keys)
        if (ch == 0x1B) {
            if (pos + 1 >= len) {
                // Incomplete: just ESC at end of buffer
                break;
            }
            if (static_cast<unsigned char>(buf[pos + 1]) == '[') {
                if (pos + 2 >= len) {
                    // Incomplete: ESC [ without final byte
                    break;
                }
                auto code = static_cast<unsigned char>(buf[pos + 2]);
                switch (code) {
                    case 'A': actions.push_back(InputAction::MoveUp); break;
                    case 'B': actions.push_back(InputAction::MoveDown); break;
                    case 'C': actions.push_back(InputAction::MoveRight); break;
                    case 'D': actions.push_back(InputAction::MoveLeft); break;
                    default: break;  // Unknown sequence, consume and discard
                }
                pos += 3;
                continue;
            }
            // ESC followed by something other than '[' — discard the ESC
            ++pos;
            continue;
        }

        // Regular character handling
        switch (ch) {
            // WASD (case-insensitive)
            case 'w': case 'W': actions.push_back(InputAction::MoveUp); break;
            case 'a': case 'A': actions.push_back(InputAction::MoveLeft); break;
            case 's': case 'S': actions.push_back(InputAction::MoveDown); break;
            case 'd': case 'D': actions.push_back(InputAction::MoveRight); break;

            // IJKL for hook (case-insensitive)
            case 'i': case 'I': actions.push_back(InputAction::HookUp); break;
            case 'j': case 'J': actions.push_back(InputAction::HookLeft); break;
            case 'k': case 'K': actions.push_back(InputAction::HookDown); break;
            case 'l': case 'L': actions.push_back(InputAction::HookRight); break;

            // Space — place mine
            case ' ': actions.push_back(InputAction::PlaceMine); break;

            // Quit
            case 'q': case 'Q': actions.push_back(InputAction::Quit); break;

            default: break;  // Ignore unknown bytes (CR, LF, etc.)
        }
        ++pos;
    }

    // Erase consumed bytes, leaving any incomplete sequences
    buf.erase(0, pos);
    return actions;
}
