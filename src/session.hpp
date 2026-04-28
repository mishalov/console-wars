#pragma once
#include "types.hpp"
#include <string>

struct Session {
    int fd = -1;
    PlayerId player_id = INVALID_PLAYER;
    bool connected = false;
    std::string out_buf;
    std::string in_buf;

    void queue_output(const std::string& data);
    void send_telnet_init();
};
