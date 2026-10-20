# Builds the server, then copies it into a slim image without the toolchain.
#
#   docker build -t nearby .
#   docker run -p 7700:7700 -v nearby-data:/data -e NEARBY_API_KEY=... nearby

FROM rust:1-slim-bookworm AS build
WORKDIR /src
COPY . .
# rust-toolchain.toml picks the toolchain; --locked keeps Cargo.lock's versions.
RUN cargo build --release --locked -p nearby-server

FROM debian:bookworm-slim
RUN useradd --system --uid 10001 --no-create-home nearby \
    && mkdir -p /data \
    && chown nearby /data
COPY --from=build /src/target/release/nearby-server /usr/local/bin/nearby-server
COPY docker/nearby.toml /etc/nearby/nearby.toml
USER nearby
VOLUME /data
EXPOSE 7700
ENTRYPOINT ["nearby-server", "/etc/nearby/nearby.toml"]
