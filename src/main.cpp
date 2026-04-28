#include "server.hpp"
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>
#include <libgen.h>

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
    if (argc > 1) {
        port = std::atoi(argv[1]);
    }

    // Resolve project root: executable is in build/, maps/ is one level up
    std::string base_dir = exe_dir(argv[0]) + "/..";

    Server server;
    if (!server.init(port, base_dir)) {
        std::cerr << "Failed to initialize server\n";
        return 1;
    }

    server.run();
    return 0;
}
