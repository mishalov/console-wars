#pragma once
#include "game_state.hpp"
#include "renderer.hpp"
#include "session.hpp"
#include "bot/bot_player.hpp"
#include <csignal>
#include <vector>
#include <string>

class Server {
public:
    bool init(int port, const std::string& base_dir, int num_bots);
    void run(volatile sig_atomic_t& shutdown_flag);
    void save_bots() const;

private:
    int listen_fd_ = -1;
    GameState state_;
    Renderer renderer_;
    std::vector<Session> sessions_;
    std::vector<BotPlayer> bots_;
    std::string base_dir_;

    bool create_listen_socket(int port);
    void accept_new_client();
    void flush_output(Session& s);
    void disconnect(Session& s);
};
