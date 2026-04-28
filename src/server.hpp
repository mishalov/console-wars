#pragma once
#include "game_state.hpp"
#include "renderer.hpp"
#include "session.hpp"
#include <vector>

class Server {
public:
    bool init(int port, const std::string& base_dir);
    void run();

private:
    int listen_fd_ = -1;
    GameState state_;
    Renderer renderer_;
    std::vector<Session> sessions_;

    bool create_listen_socket(int port);
    void accept_new_client();
    void flush_output(Session& s);
    void disconnect(Session& s);
};
