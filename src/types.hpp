#pragma once
#include <cstdint>
#include <vector>

// Convention: x = column (horizontal), y = row (vertical).
// Grid is indexed as grid[y * width + x].

struct Vec2 {
    int x = 0;
    int y = 0;

    bool operator==(const Vec2& o) const { return x == o.x && y == o.y; }
    bool operator!=(const Vec2& o) const { return !(*this == o); }
    Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
    Vec2& operator+=(const Vec2& o) { x += o.x; y += o.y; return *this; }
};

enum class TileType : uint8_t {
    Empty = 0,
    Wall  = 1,
};

enum class Direction : uint8_t {
    Up,
    Down,
    Left,
    Right,
    None,
};

inline Vec2 dir_to_vec(Direction d) {
    switch (d) {
        case Direction::Up:    return { 0, -1};
        case Direction::Down:  return { 0,  1};
        case Direction::Left:  return {-1,  0};
        case Direction::Right: return { 1,  0};
        default:               return { 0,  0};
    }
}

enum class HookState : uint8_t {
    Ready,
    Extending,
    Retracting,
};

using PlayerId = int;
constexpr PlayerId INVALID_PLAYER = -1;
