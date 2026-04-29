#include "session.hpp"

void Session::queue_output(const std::string& data) {
    out_buf.append(data);
}

void Session::send_telnet_init() {
    // IAC WILL ECHO, IAC WILL SGA, IAC DONT LINEMODE
    const char init[] = {
        '\xff', '\xfb', '\x01',   // WILL ECHO
        '\xff', '\xfb', '\x03',   // WILL SGA
        '\xff', '\xfe', '\x22',   // DONT LINEMODE
    };
    out_buf.append(init, sizeof(init));
}
