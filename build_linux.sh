#!/usr/bin/env bash
set -e

echo "============================================================"
echo "  DPUF (Deep Packed Unified Format) - Linux Native Build"
echo "============================================================"

# Detect compiler
if command -v gcc >/dev/null 2>&1; then
    CC="gcc"
elif command -v clang >/dev/null 2>&1; then
    CC="clang"
else
    echo "ERROR: Neither gcc nor clang was found. Please install build-essential or clang."
    exit 1
fi

echo "Using compiler: $CC ($($CC --version | head -n 1))"

# Check CPU AVX2 support
if grep -q "avx2" /proc/cpuinfo 2>/dev/null; then
    echo "CPU support: AVX2 detected [OK]"
else
    echo "WARNING: AVX2 flag not detected in /proc/cpuinfo. Build might fail or require compatible CPU."
fi

# Build binary and shared library
make CC="$CC" clean
make CC="$CC" -j"$(nproc 2>/dev/null || echo 4)"

echo "------------------------------------------------------------"
echo "Build succeeded!"
echo "Run interactive CLI:"
echo "  ./dpuf-cli -i"
echo "Run multi-session slots benchmark:"
echo "  ./dpuf-cli --test-slots"
echo "Run local OpenAI HTTP server:"
echo "  ./dpuf-cli --server 8080"
echo "============================================================"
