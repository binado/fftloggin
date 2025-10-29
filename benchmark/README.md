# benchmark

This directory hosts a copy of the original Fortran code from [https://jila.colorado.edu/~ajsh/FFTLog/](https://jila.colorado.edu/~ajsh/FFTLog/) with the purpose of benchmarking the correctness of our FFTLog algorithm implementation.

Minor changes were made to `fftlogtest.f`:

1 ) Fixed a typo in the computation of the log step parameter `dlogr`:

```diff
-dlogr=(logrmax-logrmin)/n
+dlogr=(logrmax-logrmin)/(n-1)
```

2) Changed the output format string in the `write` call to explicitly write exponents with more than two digits:

```diff
-write (unit,'(3es25)') k,a(i),k**(mu+1.d0)*exp(-k**2/2.d0)
+write (unit,'(3es25.16e3)') k,a(i),k**(mu+1.d0)*exp(-k**2/2.d0)
```

## Compilation

Run

```bash
gfortran --std=legacy -O2 -o fftlogtest fftlogtest.f fftlog.f drffti.f drfftf.f drfftb.f cdgamma.f
```

## Generating benchmark files

We provide a simple python script [`generate_benchmarks.py`](./generate_benchmarks.py) that creates benchmark files for a set of input parameters. These files are used in our test suite.
