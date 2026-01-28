# benchmark

This directory provides tooling to benchmark our FFTLog implementation against the original Fortran code from [Andrew Hamilton's FFTLog](https://jila.colorado.edu/~ajsh/FFTLog/).

## How it works

The Fortran source code is **not stored in this repository**. Instead, the benchmark generation script automatically downloads the original FFTLog distribution from the official source when needed.

To keep benchmark inputs reproducible, the download is verified against a pinned SHA256 checksum. If the upstream tarball changes, regenerate the checksum and update `FFTLOG_SHA256` in `scripts/generate_benchmarks.py`.

### Patches applied

The following patches are applied to `fftlogtest.f` after download:

1. Fixed a typo in the computation of the log step parameter `dlogr`:

```diff
-dlogr=(logrmax-logrmin)/n
+dlogr=(logrmax-logrmin)/(n-1)
```

2. Changed the output format string in the `write` call to use a wider field width (30 instead of 25) to ensure the `E` is not dropped when exponents have 3 digits:

```diff
-write (unit,'(3es25)') k,a(i),k**(mu+1.d0)*exp(-k**2/2.d0)
+write (unit,'(3es30.16e3)') k,a(i),k**(mu+1.d0)*exp(-k**2/2.d0)
```

## Prerequisites

You need a Fortran compiler installed:

- **macOS**: `brew install gcc`
- **Ubuntu/Debian**: `sudo apt-get install gfortran`
- **Fedora/RHEL**: `sudo dnf install gcc-gfortran`

## Generating benchmark files

Use the Python script to generate benchmark files for the test suite:

```bash
python scripts/generate_benchmarks.py
```

This script:
1. Ensures the Fortran executable exists (building it if necessary)
2. Runs the executable with various parameter combinations
3. Saves the output to `tests/benchmarks/` for use in the test suite

## Files

After running the generation script, this directory will contain:

- `README.md` - This file
- `fftlog.f` - Downloaded FFTLog Fortran source (not tracked in git)
- `fftlogtest.f` - Downloaded test program with patches (not tracked in git)
- `cdgamma.f` - Downloaded gamma function (not tracked in git)
- `drfft*.f` - Downloaded FFT routines (not tracked in git)
- `fftlogtest` - Compiled executable (not tracked in git)

## Reference

Hamilton A. J. S., 2000, MNRAS, 312, 257 ([astro-ph/9905191](https://arxiv.org/abs/astro-ph/9905191))
