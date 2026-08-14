#!/usr/bin/env bash
# Dev-only shortcut: build the strider._native accelerator with cargo and drop
# it into the source tree, where strider.thermo.nn_dna / strider.thermo.salt
# pick it up on the next import (abi3: works for any CPython >= 3.8).
# For a packaged build use `pip install .` (setuptools-rust handles the same).
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release --manifest-path native/Cargo.toml
so=$(ls native/target/release/lib_native.so 2>/dev/null || true)
[ -n "$so" ] || { echo "lib_native.so not found (build failed)?" >&2; exit 1; }
cp "$so" strider/_native.abi3.so
echo "OK: strider/_native.abi3.so ($(stat -c%s strider/_native.abi3.so) bytes)"
