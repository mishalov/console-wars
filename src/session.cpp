#include "session.hpp"

void Session::queue_output(const std::string& data) {
    out_buf.append(data);
}

void Session::send_telnet_init() {
    // Put client into character-at-a-time mode:
    // WILL ECHO: server will echo (client should not local-echo)
    // WILL SGA: suppress go-ahead (enables character mode)
    const char init[] = {
        '\xff', '\xfb', '\x01',   // IAC WILL ECHO
        '\xff', '\xfb', '\x03',   // IAC WILL SGA
    };
    out_buf.append(init, sizeof(init));
}
