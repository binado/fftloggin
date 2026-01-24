#!/usr/bin/env bash
# Build the Fortran fftlogtest executable
#
# This script downloads the FFTLog Fortran source from the official
# distribution if not already present, applies necessary patches,
# and compiles the executable.

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# URL for the original FFTLog source
FFTLOG_URL="https://jila.colorado.edu/~ajsh/FFTLog/fftlog.tgz"

# Required source files
REQUIRED_FILES="fftlog.f fftlogtest.f cdgamma.f drfftb.f drfftf.f drffti.f"

# Use FC environment variable if set, otherwise default to gfortran
FC="${FC:-gfortran}"

download_source() {
    echo "Downloading FFTLog source from $FFTLOG_URL..."

    # Download and extract
    curl -sL "$FFTLOG_URL" | tar -xzf - $REQUIRED_FILES

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download or extract FFTLog source"
        exit 1
    fi

    echo "✓ Downloaded source files"
}

apply_patches() {
    echo "Applying patches to fftlogtest.f..."

    # Patch 1: Fix dlogr computation (use n-1 instead of n)
    sed -i.bak 's/dlogr=(logrmax-logrmin)\/n/dlogr=(logrmax-logrmin)\/(n-1)/' fftlogtest.f

    # Patch 2: Use wider field width (30) to ensure 'E' is not dropped with 3-digit exponents
    sed -i.bak "s/write (unit,'(3es25)')/write (unit,'(3es30.16e3)')/" fftlogtest.f

    # Clean up backup files
    rm -f fftlogtest.f.bak

    echo "✓ Applied patches"
}

# Check if source files exist, download if not
all_exist=true
for f in $REQUIRED_FILES; do
    if [ ! -f "$f" ]; then
        all_exist=false
        break
    fi
done

if [ "$all_exist" = false ]; then
    download_source
    apply_patches
fi

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
