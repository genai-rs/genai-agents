#!/usr/bin/env bash
# Generate the demo's HTTP server (demo/api) from demo/api/openapi.yaml with openapi-generator-timvw-rust.
#
# The generated crate is checked in: CI builds it without Docker. Run this after changing the spec, and commit what
# it changes. GENERATOR_IMAGE picks another image, such as one built from a generator checkout
# (`docker build -t timvw-rust:dev .`, see the generator's docs/development.md).
set -euo pipefail

TIMVW_RUST_VERSION="${TIMVW_RUST_VERSION:-0.31.0}"
OPENAPI_GENERATOR_VERSION="${OPENAPI_GENERATOR_VERSION:-7.27.0}"
GENERATOR_IMAGE="${GENERATOR_IMAGE:-ghcr.io/timvw/openapi-generator-timvw-rust:${TIMVW_RUST_VERSION}-og${OPENAPI_GENERATOR_VERSION}}"

API_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../api" && pwd)"

# Only what the generator writes; the spec, its config and .openapi-generator-ignore stay.
rm -rf "$API_DIR/src" "$API_DIR/docs" "$API_DIR/.openapi-generator"

docker run --rm \
    -v "$API_DIR:/local" \
    -w /local \
    -u "$(id -u):$(id -g)" \
    "$GENERATOR_IMAGE" generate \
    -i /local/openapi.yaml \
    -g timvw-rust \
    -o /local \
    -c /local/generator-config.json

cargo fmt --manifest-path "$API_DIR/Cargo.toml"
