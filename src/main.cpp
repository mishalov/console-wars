#include "server.hpp"
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <libgen.h>

#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __SSE3__
#include <pmmintrin.h>
#endif

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
    // Flush subnormal floats to zero — prevents progressive performance
    // degradation from Adam optimizer's second-moment estimates decaying
    // into subnormal range after hours of training.
#ifdef __SSE__
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
#endif
#ifdef __SSE3__
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
#ifdef __aarch64__
    uint64_t fpcr;
    asm volatile("mrs %0, fpcr" : "=r"(fpcr));
    fpcr |= (1 << 24);  // FZ bit
    asm volatile("msr fpcr, %0" : : "r"(fpcr));
#endif

    signal(SIGPIPE, SIG_IGN);

    int port = 7777;
    int num_bots = 0;
    bool no_train = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bot") == 0) {
            num_bots = 1;
        } else if (std::strcmp(argv[i], "--bots") == 0 && i + 1 < argc) {
            num_bots = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--no-train") == 0) {
            no_train = true;
        } else {
            // First positional argument is port
            port = std::atoi(argv[i]);
        }
    }

    // Resolve project root: executable is in build/, maps/ is one level up
    std::string base_dir = exe_dir(argv[0]) + "/..";

    Server server;

    signal(SIGINT, handle_sigint);

    if (!server.init(port, base_dir, num_bots, no_train)) {
        std::cerr << "Failed to initialize server\n";
        return 1;
    }

    server.run(g_shutdown);

    // Graceful shutdown: save bot brains
    server.save_bots();
    return 0;
}
