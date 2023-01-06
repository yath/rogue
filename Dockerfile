FROM rust:1.66 as builder
WORKDIR /rogue
COPY . .
RUN apt-get update && apt-get install -y libasound2-dev pkg-config && rm -rf /var/lib/apt/lists/*
RUN cargo install --path .

FROM debian:buster-slim
RUN apt-get update && apt-get install -y libasound2 && rm -rf /var/lib/apt/lists/* && mkdir -p /rogue
COPY --from=builder /usr/local/cargo/bin/rogue /rogue/rogue
ENTRYPOINT ["/rogue/rogue"]
