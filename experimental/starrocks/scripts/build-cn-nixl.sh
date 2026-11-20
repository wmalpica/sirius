#!/usr/bin/env bash
# Build a private shared SDK for nixl-sys from main's already-pinned NIXL source.
# Main keeps using its existing static SDK and exchange transport unchanged.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
source_dir="$repo/build/release/_deps/nixl-src"
install_dir="$repo/build/starrocks/nixl-install"
build_dir="$repo/build/starrocks/nixl-build"
[[ -f "$source_dir/meson.build" ]] || {
    echo "Build the main Sirius engine before the StarRocks shared NIXL SDK" >&2
    exit 1
}
# CUDA uses different target names on x86-64 and ARM (for example sbsa-linux).
# Reuse CMake's resolved libraries and discover the installed target headers.
cuda_include="$CONDA_PREFIX/include"
for header in "$CONDA_PREFIX"/targets/*/include/cuda_runtime_api.h; do
    if [[ -f "$header" ]]; then
        cuda_include=$(dirname "$header")
        break
    fi
done
cache="$repo/build/release/CMakeCache.txt"
cudart=$(sed -n 's/^CUDA_cudart_LIBRARY:FILEPATH=//p' "$cache")
driver=$(sed -n 's/^CUDA_cuda_driver_LIBRARY:FILEPATH=//p' "$cache")
[[ -f "$cuda_include/cuda_runtime_api.h" && -f "$cudart" && -f "$driver" ]] || {
    echo "The main build has no usable CUDA toolkit configuration" >&2
    exit 1
}
meson setup --reconfigure "$build_dir" "$source_dir" \
    --native-file "$repo/build/release/_deps/nixl-native.ini" \
    --prefix "$install_dir" --libdir lib --wrap-mode nofallback \
    -Dnative_only=true -Ddefault_library=shared -Dstatic_plugins= \
    -Denable_plugins=UCX -Dbuild_tests=false -Dbuild_examples=false \
    -Dbuild_nixl_ep=false -Dwith_trace=false -Dwerror=false \
    -Dcudapath_inc="$cuda_include" -Dcudapath_lib="$(dirname "$cudart")" \
    -Dcudapath_stub="$(dirname "$driver")"
meson compile -C "$build_dir" -j 4
meson install -C "$build_dir" --no-rebuild
