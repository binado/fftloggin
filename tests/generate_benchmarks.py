#!/usr/bin/env python3
"""
Generate benchmark files by running the fftlogtest Fortran executable.

This script creates a comprehensive test suite by running the executable
with all combinations of parameters and saving the results to appropriately
named files for later parsing.
"""

import subprocess
import sys
from pathlib import Path

# Define paths
BENCHMARK_DIR = Path(__file__).parent.parent / "benchmark"
EXECUTABLE = BENCHMARK_DIR / "fftlogtest"
BUILD_SCRIPT = BENCHMARK_DIR / "build.sh"

# Output directory
OUTPUT_DIR = Path(__file__).parent / "benchmarks"

# Define parameter ranges
LOG10R_PAIRS = [(-1, 1), (-2, 2)]
N_VALUES = [63, 64]
MU_VALUES = [0, 1, 2]
Q_VALUES = [-0.5, 0, 0.5]
KR_VALUES = [0.1, 1, 10]
LOWRING_VALUES = [True, False]


def ensure_executable():
    """Build the executable if it doesn't exist."""
    if EXECUTABLE.exists():
        return True

    print("Fortran executable not found. Building from source...")

    if not BUILD_SCRIPT.exists():
        print(f"ERROR: Build script not found at {BUILD_SCRIPT}")
        return False

    try:
        result = subprocess.run(
            ["bash", str(BUILD_SCRIPT)],
            cwd=str(BENCHMARK_DIR),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"Build failed:\n{result.stderr}")
            return False
        print(result.stdout)
        return EXECUTABLE.exists()
    except Exception as e:
        print(f"Build failed: {e}")
        return False


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
