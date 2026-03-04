## 0.3.1 (2026-03-04)

### Refactor

- tighten package typing annotations (#46)
- simplify domain checking system (#45)

## 0.3.0 (2026-02-11)

### Feat

- add CombinedKernel class (#29)
- add domain validation for FFTLog bias parameter (#28)

### Fix

- raise ValueError on input size mismatch in forward()/inverse() (#36)
- remove unused pixi-sync-environment pre-commit hook (#24)

### Refactor

- download Fortran benchmark code on-demand instead of storin… (#21)

### Perf

- reduce fftlog allocation churn (#33)

## 0.2.1 (2025-11-02)

### Fix

- calculate Pochhammer symbol in a safer way for kernel derivatives

## 0.2.0 (2025-11-01)

### Feat

- allow batch transforms with array params
- add `_is_in_strip` method to Derivative kernel
- add SphericalBesselJKernel
- add utility functions tests
- add utility functions for array manipulation
- implement updated API with decoupled kernels and support for grids
- add main FFTLog class
- add api for custom kernels, grid utils, and more tests
- add tutorial notebook
- update numexpr backend and move public API to separate file

### Fix

- ensure array_like properties are converted to arrays before calculations
- update bias and dlog type hinting
- change sign of argument shift in spherical bessel kernel
- add r modifier to docstrings that use latex symbols
- use mathematically correct implementation of SphericalBesselJKernel
- remove domain checking in BesselJKernel constructor
- annotate logc as array-like
- read array length from last dimension instead of first
- remove in-place exponential
- remove redundant test
- **tests**: test more kernels in test_grid_identity
- remove bias shifting logic from fftlog module
- check if bias value is in range in kernel constructor
- replace poch with gamma and rgamma to handle complex arrays
- refactor kernel bias and bounds logic
- update test_vectorized_grid expectations
- avoid adding unecessary dimensions in outer_broadcast
- add correct broadcast method in Bessel kernel
- remove broadcast handling in pure math routines
- remove deprecated tests
- prevent deletion of cached properties before being calculated
- move bias parameter to Kernel class
- apply correct broadcasting for Bessel kernel
- add check to force periodicity of kernel coefficients
- apply correct computation of offset in kernel coefficients
- shift k0r0 by correct amount when minimize_ringing is true
- implement correct formula for low-ringing offset
- correctly compute central element index for even arrays
- **tests**: refactor tests to match new API
- update gitignore

### Refactor

- replace outer_broadcast with safe_broadcast
- rename API methods and parameters for better UX (#8)
- make is_in_strip a public method
- decouple bias from kernel classes and add flag for optional bound checking
- add batch support to Grid and helper functions
- use np.angle for phase calculation
- redesign Grid API with clean separation of concerns
- remove intermediary computation in Bessel kernel
- make bessel strip upper bound array-like
- change how derivatives shift the strip of definition
- provide default strip for kernel
- add CLAUDE.md file
- remove unnecessary conditional
- rename fht module to fftlog
