#!/usr/bin/env bash
# Build the Fortran fftlogtest executable

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use FC environment variable if set, otherwise default to gfortran
FC="${FC:-gfortran}"

# Check if the Fortran compiler is available
if ! command -v "$FC" &> /dev/null; then
    echo "ERROR: Fortran compiler '$FC' not found. Please install it:"
    echo "  - macOS: brew install gcc"
    echo "  - Ubuntu/Debian: sudo apt-get install gfortran"
    echo "  - Fedora/RHEL: sudo dnf install gcc-gfortran"
    echo ""
    echo "Or set FC environment variable to your compiler:"
    echo "  export FC=/path/to/your/fortran/compiler"
    exit 1
fi

echo "Building fftlogtest with $FC..."
"$FC" --std=legacy -O2 -o fftlogtest \
    fftlogtest.f fftlog.f drffti.f drfftf.f drfftb.f cdgamma.f

if [ -f fftlogtest ]; then
    echo "✓ Build successful: fftlogtest"
    chmod +x fftlogtest
else
    echo "✗ Build failed"
    exit 1
fi
