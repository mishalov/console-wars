# Stage 1: Build
FROM debian:bookworm-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends g++ cmake make && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY CMakeLists.txt .
COPY src/ src/

RUN cmake -DCMAKE_BUILD_TYPE=Release . && make -j$(nproc)

# Stage 2: Runtime
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 tini && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/console-wars /app/console-wars
COPY maps/ maps/
COPY data/bot_brain.bin assets/bot_brain.bin

RUN mkdir -p /app/data

ENV BASE_DIR=/app
ENV DATA_DIR=/app/data
ENV BOT_COUNT=2
ENV NO_TRAIN=0

EXPOSE 7777

STOPSIGNAL SIGTERM
ENTRYPOINT ["tini", "--"]
CMD ["/app/console-wars"]
