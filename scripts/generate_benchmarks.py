#!/usr/bin/env python3
"""
Generate benchmark files by running the fftlogtest Fortran executable.

This script creates a comprehensive test suite by running the executable
with all combinations of parameters and saving the results to appropriately
named files for later parsing.

The Fortran source code is downloaded on-demand from the original FFTLog
distribution at https://jila.colorado.edu/~ajsh/FFTLog/
"""

import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import hashlib
from pathlib import Path

# Define paths
ROOT_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = ROOT_DIR / "benchmark"
EXECUTABLE = BENCHMARK_DIR / "fftlogtest"

# URL for the original FFTLog source
FFTLOG_URL = "https://jila.colorado.edu/~ajsh/FFTLog/fftlog.tgz"
FFTLOG_SHA256 = "5548708c1c80c8d00ab87036c8c8fc419a40b981def0699ebb9f26d21ecac2e7"

# Required Fortran source files from the archive
REQUIRED_FILES = [
    "fftlog.f",
    "fftlogtest.f",
    "cdgamma.f",
    "drfftb.f",
    "drfftf.f",
    "drffti.f",
]

# Patches to apply to fftlogtest.f (see benchmark/README.md for details)
FFTLOGTEST_PATCHES = [
    # Fix typo in dlogr computation: use (n-1) instead of n
    ("dlogr=(logrmax-logrmin)/n", "dlogr=(logrmax-logrmin)/(n-1)"),
    # Use wider field width (30) to ensure 'E' is not dropped with 3-digit exponents
    (
        "write (unit,'(3es25)') k,a(i),k**(mu+1.d0)*exp(-k**2/2.d0)",
        "write (unit,'(3es30.16e3)') k,a(i),k**(mu+1.d0)*exp(-k**2/2.d0)",
    ),
]

# Output directory
OUTPUT_DIR = ROOT_DIR / "tests" / "benchmarks"

# Define parameter ranges
LOG10R_PAIRS = [(-1, 1), (-2, 2)]
N_VALUES = [63, 64]
MU_VALUES = [0, 1, 2]
Q_VALUES = [-0.5, 0, 0.5]
KR_VALUES = [0.1, 1, 10]
LOWRING_VALUES = [True, False]


def download_fortran_source():
    """Download FFTLog Fortran source from the official distribution."""
    print(f"Downloading FFTLog source from {FFTLOG_URL}...")

    # Create benchmark directory if it doesn't exist
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            urllib.request.urlretrieve(FFTLOG_URL, tmp_path)

        # Verify SHA256 checksum to ensure reproducible source
        digest = hashlib.sha256(tmp_path.read_bytes()).hexdigest()
        if digest != FFTLOG_SHA256:
            print("ERROR: FFTLog source checksum mismatch")
            print(f"  Expected: {FFTLOG_SHA256}")
            print(f"  Actual:   {digest}")
            return False

        # Extract required files
        with tarfile.open(tmp_path, "r:gz") as tar:
            for member in tar.getmembers():
                member_basename = Path(member.name).name
                if member_basename in REQUIRED_FILES:
                    # Extract to benchmark directory
                    member.name = member_basename  # Remove any path prefix
                    tar.extract(member, BENCHMARK_DIR)
                    print(f"  Extracted: {member.name}")

        # Verify all required files were extracted
        missing = [f for f in REQUIRED_FILES if not (BENCHMARK_DIR / f).exists()]
        if missing:
            print(f"ERROR: Missing files after extraction: {missing}")
            return False

        return True

    except urllib.error.URLError as e:
        print(f"ERROR: Failed to download FFTLog source: {e}")
        return False
    except tarfile.TarError as e:
        print(f"ERROR: Failed to extract archive: {e}")
        return False
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()


def apply_patches():
    """Apply necessary patches to fftlogtest.f."""
    fftlogtest_path = BENCHMARK_DIR / "fftlogtest.f"

    if not fftlogtest_path.exists():
        print("ERROR: fftlogtest.f not found")
        return False

    print("Applying patches to fftlogtest.f...")
    content = fftlogtest_path.read_text()

    for old, new in FFTLOGTEST_PATCHES:
        if old in content:
            content = content.replace(old, new)
            print(f"  Applied: {old[:40]}... -> {new[:40]}...")
        elif new in content:
            print(f"  Already applied: {new[:40]}...")
        else:
            print(f"  WARNING: Pattern not found: {old[:40]}...")

    fftlogtest_path.write_text(content)
    return True


def build_executable():
    """Compile the Fortran source into an executable."""
    # Respect FC environment variable first, then fall back to compiler search
    fc = os.environ.get("FC")
    if fc:
        # Validate that the FC environment variable points to an actual compiler
        fc_path = shutil.which(fc)
        if not fc_path:
            print(
                f"ERROR: FC environment variable is set to '{fc}' but not found in PATH"
            )
            return False
        fc = fc_path
    else:
        # Fall back to searching for common Fortran compilers
        fc = shutil.which("gfortran") or shutil.which("f77") or shutil.which("f95")
        if not fc:
            print("ERROR: No Fortran compiler found. Please install gfortran:")
            print("  - macOS: brew install gcc")
            print("  - Ubuntu/Debian: sudo apt-get install gfortran")
            print("  - Fedora/RHEL: sudo dnf install gcc-gfortran")
            return False

    print(f"Building fftlogtest with {fc}...")

    source_files = [str(BENCHMARK_DIR / f) for f in REQUIRED_FILES]
    cmd = [fc, "--std=legacy", "-O2", "-o", str(EXECUTABLE)] + source_files

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"Build failed:\n{result.stderr}")
            return False

        if EXECUTABLE.exists():
            EXECUTABLE.chmod(0o755)
            print("✓ Build successful: fftlogtest")
            return True
        else:
            print("✗ Build failed: executable not created")
            return False

    except subprocess.TimeoutExpired:
        print("ERROR: Build timed out")
        return False
    except Exception as e:
        print(f"ERROR: Build failed: {e}")
        return False


def ensure_executable():
    """Build the executable if it doesn't exist, downloading source if needed."""
    if EXECUTABLE.exists():
        return True

    print("Fortran executable not found. Setting up from source...")
    print()

    # Check if source files exist, download if not
    source_exists = all((BENCHMARK_DIR / f).exists() for f in REQUIRED_FILES)

    if not source_exists:
        if not download_fortran_source():
            return False

        if not apply_patches():
            return False

    # Build the executable
    return build_executable()


def run_benchmark(log10rmin, log10rmax, n, mu, q, kr, lowring, filename):
    """
    Run the fftlogtest executable with given parameters.

    Returns:
        tuple: (success: bool, error: str)
    """
    # Prepare input - log10rmin and log10rmax on same line
    # Note: Use correct formula with (n-1) instead of n
    lowring_input = "y" if lowring else "n"
    input_str = (
        f"{log10rmin} {log10rmax}\n{n}\n{mu}\n{q}\n{kr}\n{lowring_input}\n{filename}\n"
    )

    try:
        subprocess.run(
            [str(EXECUTABLE)],
            input=input_str,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(OUTPUT_DIR),  # Run in benchmark directory
            check=False,
        )
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "Timeout (30s)"
    except Exception as e:
        return False, str(e)


def normalize_fortran_output(filepath: Path) -> bool:
    """Fix missing exponent markers in Fortran output files."""
    if not filepath.exists():
        return False

    content = filepath.read_text()
    pattern = re.compile(r"(?<![EeDd])([+-]?\d*\.\d+)([+-]\d{2,3})")
    updated = pattern.sub(r"\1e\2", content)
    if updated != content:
        filepath.write_text(updated)
    return True


def main():
    """Generate all benchmark files."""
    if not ensure_executable():
        print("\nCould not build executable. Please install gfortran and try again.")
        print("  - macOS: brew install gcc")
        print("  - Ubuntu/Debian: sudo apt-get install gfortran")
        print("  - Fedora/RHEL: sudo dnf install gcc-gfortran")
        sys.exit(1)

    # Calculate total combinations
    total = (
        len(LOG10R_PAIRS)
        * len(N_VALUES)
        * len(MU_VALUES)
        * len(Q_VALUES)
        * len(KR_VALUES)
        * len(LOWRING_VALUES)
    )

    print(f"Generating {total} benchmark files...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate all combinations
    count = 0
    errors = []

    for log10rmin, log10rmax in LOG10R_PAIRS:
        for n in N_VALUES:
            for mu in MU_VALUES:
                for q in Q_VALUES:
                    for kr in KR_VALUES:
                        for lowring in LOWRING_VALUES:
                            count += 1

                            # Generate filename
                            lowring_str = "y" if lowring else "n"
                            filename = (
                                f"benchmark_log10rmin={log10rmin}_"
                                f"log10rmax={log10rmax}_n={n}_"
                                f"mu={mu}_q={q}_kr={kr}_lowring={lowring_str}.txt"
                            )
                            filepath = OUTPUT_DIR / filename

                            # Run benchmark
                            print(
                                f"[{count:3d}/{total}] Running: {filename}...",
                                end=" ",
                                flush=True,
                            )
                            success, error = run_benchmark(
                                log10rmin, log10rmax, n, mu, q, kr, lowring, filename
                            )

                            if success:
                                # Program creates the file itself
                                if filepath.exists():
                                    normalize_fortran_output(filepath)
                                    print("✓")
                                else:
                                    print("✗ (File not created)")
                                    errors.append((filename, "File not created"))
                            else:
                                print(f"✗ ({error})")
                                errors.append((filename, error))

    # Summary
    print()
    print("=" * 60)
    print(f"Completed {count} benchmark runs")

    if errors:
        print(f"\nEncountered {len(errors)} error(s):")
        for filename, error in errors[:10]:  # Show first 10 errors
            print(f"  - {filename}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    else:
        print("All benchmarks completed successfully!")

    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
