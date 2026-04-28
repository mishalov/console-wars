#include "server.hpp"
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <libgen.h>

static volatile sig_atomic_t g_shutdown = 0;

static void handle_sigint(int) {
    g_shutdown = 1;
}

static std::string exe_dir(const char* argv0) {
    // Resolve the directory containing the executable
    std::string path(argv0);
    // dirname may modify its argument, so work on a copy
    std::vector<char> buf(path.begin(), path.end());
    buf.push_back('\0');
    char* dir = dirname(buf.data());
    return std::string(dir);
}

int main(int argc, char* argv[]) {
    signal(SIGPIPE, SIG_IGN);

    int port = 7777;
    int num_bots = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bot") == 0) {
            num_bots = 1;
        } else if (std::strcmp(argv[i], "--bots") == 0 && i + 1 < argc) {
            num_bots = std::atoi(argv[++i]);
        } else {
            // First positional argument is port
            port = std::atoi(argv[i]);
        }
    }

    // Resolve project root: executable is in build/, maps/ is one level up
    std::string base_dir = exe_dir(argv[0]) + "/..";

    Server server;

    signal(SIGINT, handle_sigint);

    if (!server.init(port, base_dir, num_bots)) {
        std::cerr << "Failed to initialize server\n";
        return 1;
    }

    server.run(g_shutdown);

    // Graceful shutdown: save bot brains
    server.save_bots();
    return 0;
}
