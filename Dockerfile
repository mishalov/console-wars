# Stage 1: Build
FROM alpine:3.19 AS builder

RUN apk add --no-cache g++ cmake make

WORKDIR /build
COPY CMakeLists.txt .
COPY src/ src/

RUN cmake -DCMAKE_BUILD_TYPE=Release . && make -j$(nproc)

# Stage 2: Runtime
FROM alpine:3.19

RUN apk add --no-cache libstdc++ libgcc

WORKDIR /app

COPY --from=builder /build/console-wars ./exec
COPY maps/ maps/
COPY data/bot_brain.bin assets/bot_brain.bin

RUN mkdir -p /app/data

ENV BASE_DIR=/app
ENV DATA_DIR=/app/data
ENV BOT_COUNT=2
ENV NO_TRAIN=0

EXPOSE 7777

CMD ["./exec"]
