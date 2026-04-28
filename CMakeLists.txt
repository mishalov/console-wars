cmake_minimum_required(VERSION 3.16)
project(console-wars LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(console-wars
    src/main.cpp
    src/server.cpp
    src/session.cpp
    src/game_state.cpp
    src/pudge.cpp
    src/hook.cpp
    src/renderer.cpp
    src/input.cpp
    src/bot/bot_player.cpp
    src/bot/bot_observation.cpp
    src/bot/bot_reward.cpp
    src/bot/neural_net.cpp
    src/bot/dqn_brain.cpp
)

target_include_directories(console-wars PRIVATE src)

target_compile_options(console-wars PRIVATE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
)
