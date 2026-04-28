#include "server.hpp"
#include "input.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <poll.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>

static bool set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return false;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK) >= 0;
}

bool Server::create_listen_socket(int port) {
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        perror("socket");
        return false;
    }

    int opt = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(port));

    if (bind(listen_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        perror("bind");
        close(listen_fd_);
        return false;
    }

    if (listen(listen_fd_, 16) < 0) {
        perror("listen");
        close(listen_fd_);
        return false;
    }

    set_nonblocking(listen_fd_);
    return true;
}

bool Server::init(int port, const std::string& base_dir) {
    std::string map_path = base_dir + "/maps/default.txt";
    if (!state_.load_map(map_path)) {
        return false;
    }
    if (!create_listen_socket(port)) {
        return false;
    }
    std::cout << "Server listening on port " << port << "\n";
    return true;
}

void Server::accept_new_client() {
    struct sockaddr_in client_addr{};
    socklen_t client_len = sizeof(client_addr);
    int fd = accept(listen_fd_, reinterpret_cast<struct sockaddr*>(&client_addr), &client_len);
    if (fd < 0) {
        return;  // EAGAIN/EWOULDBLOCK — no pending connection
    }

    if (sessions_.size() >= 16) {
        close(fd);
        return;
    }

    set_nonblocking(fd);

    // Assign spawn point and create pudge
    Vec2 spawn = state_.next_spawn_point();
    PlayerId pid = state_.add_pudge(spawn);

    Session s;
    s.fd = fd;
    s.player_id = pid;
    s.connected = true;

    s.send_telnet_init();

    // Send initial clear screen
    s.queue_output("\x1b[2J");

    sessions_.push_back(std::move(s));
    std::cout << "Client connected (fd=" << fd << ", player=" << pid << ")\n";
}

void Server::flush_output(Session& s) {
    if (s.out_buf.empty()) return;

    ssize_t n = write(s.fd, s.out_buf.data(), s.out_buf.size());
    if (n > 0) {
        s.out_buf.erase(0, static_cast<size_t>(n));
    } else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
        disconnect(s);
    }
}

void Server::disconnect(Session& s) {
    if (s.fd >= 0) {
        // Restore cursor before closing
        const char* restore = "\x1b[?25h\x1b[0m";
        write(s.fd, restore, strlen(restore));
        close(s.fd);
        std::cout << "Client disconnected (fd=" << s.fd << ", player=" << s.player_id << ")\n";
    }
    // Remove pudge from game state
    if (s.player_id != INVALID_PLAYER) {
        state_.remove_pudge(s.player_id);
        state_.remove_mines_by_owner(s.player_id);
    }
    s.fd = -1;
    s.connected = false;
}

void Server::run() {
    std::vector<struct pollfd> pollfds;

    static constexpr size_t MAX_OUT_BUF = 256 * 1024;  // 256KB
    constexpr auto TICK_INTERVAL = std::chrono::milliseconds(66);
    auto last_tick = std::chrono::steady_clock::now();

    while (true) {
        pollfds.clear();

        // Slot 0: listening socket
        pollfds.push_back({listen_fd_, POLLIN, 0});

        // Slots 1..N: client sessions
        for (auto& s : sessions_) {
            short events = POLLIN;
            if (!s.out_buf.empty()) events |= POLLOUT;
            pollfds.push_back({s.fd, events, 0});
        }

        int ret = poll(pollfds.data(), static_cast<nfds_t>(pollfds.size()), 66);
        if (ret < 0) {
            if (errno == EINTR) continue;
            perror("poll");
            break;
        }

        // Process client I/O
        size_t num_sessions = sessions_.size();
        for (size_t i = 0; i < num_sessions; ++i) {
            auto& pfd = pollfds[i + 1];
            auto& s = sessions_[i];

            if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
                disconnect(s);
                continue;
            }
            if (pfd.revents & POLLOUT) {
                flush_output(s);
                if (!s.connected) continue;
            }
            if (pfd.revents & POLLIN) {
                char tmp[512];
                ssize_t n = read(s.fd, tmp, sizeof(tmp));
                if (n <= 0) {
                    disconnect(s);
                    continue;
                }
                s.in_buf.append(tmp, static_cast<size_t>(n));
                // Safety: disconnect clients flooding input
                if (s.in_buf.size() > 4096) {
                    disconnect(s);
                }
            }
        }

        // Accept new clients
        if (pollfds[0].revents & POLLIN) {
            accept_new_client();
        }

        // Only tick + render at the fixed rate
        auto now = std::chrono::steady_clock::now();
        if (now - last_tick >= TICK_INTERVAL) {
            last_tick = now;

            // Process input for each connected session
            for (auto& s : sessions_) {
                if (!s.connected) continue;
                auto actions = parse_input(s.in_buf);
                for (auto action : actions) {
                    if (action == InputAction::Quit) {
                        disconnect(s);
                        break;
                    }
                    state_.handle_input(s.player_id, action);
                }
            }

            // Remove disconnected sessions
            sessions_.erase(
                std::remove_if(sessions_.begin(), sessions_.end(),
                    [](const Session& s) { return !s.connected; }),
                sessions_.end()
            );

            // Tick game state
            state_.tick();

            // Render and send frame to each client (per-player view)
            for (auto& s : sessions_) {
                if (s.out_buf.size() > MAX_OUT_BUF) {
                    disconnect(s);
                    continue;
                }
                std::string frame = renderer_.render_full(state_, s.player_id);
                s.queue_output(frame);
            }

            // Remove any sessions disconnected due to buffer overflow
            sessions_.erase(
                std::remove_if(sessions_.begin(), sessions_.end(),
                    [](const Session& s) { return !s.connected; }),
                sessions_.end()
            );
        }
    }
}
