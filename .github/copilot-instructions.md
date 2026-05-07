# Copilot cloud agent instructions for `kokkos/kokkos-kernels`

## Project overview
- KokkosKernels implements local computational kernels for linear algebra (BLAS and LAPACK) and graph operations using the Kokkos shared-memory parallel programming model.
- The project supports CUDA, HIP, SYCL, OpenMP, Threads, and Serial backends (availability depends on how Kokkos is configured).
- For cloud-agent work, use a modern toolchain targeting C++20 and CMake 3.22+.
- Primary development branch is `develop` (not `main`).

## Build instructions
This repository does **not** configure standalone unless Kokkos is already built and discoverable.

1. Clone `kokkos/kokkos` beside this repository (CI commonly uses `5.1.0`).
2. Configure/build/install Kokkos.
3. Configure Kokkos Kernels with `-DKokkos_ROOT=<kokkos-install-prefix>`.

Minimal host flow:
```bash
# from a workspace containing both repos: kokkos/ and kokkos-kernels/
cmake -S kokkos -B kokkos/build \
  -DCMAKE_CXX_STANDARD=20 \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_TESTS=OFF \
  -DKokkos_ENABLE_DEPRECATED_CODE_5=OFF \
  -DCMAKE_INSTALL_PREFIX=$PWD/kokkos/install
cmake --build kokkos/build --target install --parallel $(nproc)

cmake -S kokkos-kernels -B kokkos-kernels/build \
  -DKokkos_ROOT=$PWD/kokkos/install \
  -DKokkosKernels_ENABLE_TESTS=ON \
  -DKokkosKernels_ENABLE_EXAMPLES=ON
cmake --build kokkos-kernels/build --parallel $(nproc)
```

## Naming Style Guidelines for Kokkos Kernels Development
- Public CMake options follow `KokkosKernels_<OPTION>` (camel-case prefix + uppercase option name), for example `KokkosKernels_ENABLE_TESTS`.
- Internal CMake regular variables are typically uppercase `KOKKOSKERNELS_*`.
- Public API headers are generally in `*/src` with `Kokkos*` naming; implementation details are commonly in `*/src/impl` or `*::Impl` and should not be treated as stable public API.
- Preserve existing component naming (`blas`, `lapack`, `graph`, `sparse`, `batched`, `common`, `ode`) in paths and docs updates.

## Repository structure

```text
kokkos-kernels/
├── sparse/        # Sparse kernels (4.7M, ~209 headers, ~15 src) - PRIMARY
│   ├── src/       # Public interfaces + implementations
│   └── unit_test/ # Sparse unit tests
├── batched/       # Batched kernels (3.3M, ~334 headers, ~22 src)
│   ├── dense/     # Dense batched kernels
│   ├── eti/       # Explicit template instantiation sources
│   └── sparse/    # Sparse batched kernels
├── blas/          # BLAS kernels (3.6M, ~229 headers, ~10 src)
│   ├── src/       # Public interfaces + implementations
│   └── unit_test/ # BLAS unit tests
├── common/        # Shared utilities (612K, ~47 headers, ~11 src)
│   ├── src/       # Common utilities and helpers
│   └── unit_test/ # Common unit tests
├── graph/         # Graph kernels (788K, ~37 headers, ~7 src)
│   ├── src/       # Graph interfaces + implementations
│   └── unit_test/ # Graph unit tests
├── lapack/        # LAPACK kernels (652K, ~42 headers, ~10 src)
│   ├── src/       # LAPACK interfaces + implementations
│   └── unit_test/ # LAPACK unit tests
├── ode/           # ODE kernels (252K, ~15 headers, ~7 src)
│   ├── src/       # ODE interfaces + implementations
│   └── unit_test/ # ODE unit tests
├── perf_test/     # Performance tests/drivers (4.8M)
├── benchmarks/    # Benchmark drivers and scripts
├── test_common/   # Shared unit-test infrastructure
├── example/       # Usage examples and integration-style samples
├── docs/source/   # Sphinx docs sources
├── cmake/         # Build system modules and options
├── tpls/          # Third-party content and vendored support code
├── .github/workflows/ # CI: linux.yml, osx.yml, at2.yml, format.yml, docs.yml, codeql.yml
├── CMakeLists.txt # Root configuration (version/options/components)
├── BUILD.md       # Build/setup guide
└── DEVELOPER.md   # Developer conventions and CMake option patterns
```

**Key files:** `CMakeLists.txt`, `cmake/kokkoskernels_tribits.cmake`, `.github/workflows/linux.yml`, `.github/workflows/docs.yml`, `BUILD.md`, `DEVELOPER.md`

## CI/CD workflows
Main workflows under `.github/workflows/`:
- `linux.yml`: Linux PR sanitizer/testing workflow
- `osx.yml`: macOS PR build/test workflow
- `at2.yml`: orchestrates reusable GPU/host workflow set (`h100_lychee.yml`, `v100_kumquat.yml`, `mi210.yml`, `host.yml`, `pv.yml`)
- `format.yml`: clang-format-16 check on changed C/C++ files
- `docs.yml`: API-documentation consistency check + docs build/deploy
- `codeql.yml`: CodeQL analysis workflow
- `dependency-review.yml`, `scorecards.yml`: dependency and security posture checks
- `release.yml`: release artifact packaging and publication on tags

Practical CI behavior to remember:
- Linux/OSX PR workflows ignore docs-only changes (`**/*.md`, `docs/**`, etc.).
- Many build workflows pin Kokkos to `5.1.0` and treat warnings strictly.

## Testing
### Test structure
- Correctness tests are primarily in `*/unit_test`.
- Performance and benchmark-style tests are in `perf_test`.
- Tests are enabled only when configuring with `-DKokkosKernels_ENABLE_TESTS=ON`.

### How to execute tests
From a configured build directory:
```bash
ctest --test-dir kokkos-kernels/build --output-on-failure --timeout 3600
```

Useful focused runs:
```bash
ctest --test-dir kokkos-kernels/build -R <regex> --output-on-failure
```

Docs/API checks (for docs-only or API-surface changes):
```bash
git fetch origin develop:refs/remotes/origin/develop
git diff --name-only origin/develop > modified_files.txt
python3 scripts/check_api_updates.py
python3 -m pip install --require-hashes -r docs/build_requirements.txt
make -C docs html
```

## Critical Rules
### NEVER
- Never assume Kokkos Kernels can build without first installing/configuring Kokkos.
- Never treat `*/src/impl` or `*::Impl` symbols as stable public API contracts.
- Never skip formatting for touched C/C++ source/header files when relevant (`clang-format-16` is the CI formatter).
- Never assume `main` is the primary integration branch; use `develop` for comparisons and most CI-aligned checks.

### ALWAYS
- Always mirror CI workflow options (`linux.yml`, `osx.yml`, reusable workflows) when reproducing issues locally.
- Always pass `-DKokkos_ROOT=<kokkos-install-prefix>` for standalone builds.
- Always run focused `ctest -R ...` first for touched areas, then broader test coverage as needed.
- Always run docs/API consistency checks when public headers or docs are changed.

## Errors encountered during onboarding and workarounds
1. **CMake configure error: missing Kokkos package**
   - Error: `Could not find a package configuration file provided by "Kokkos"`.
   - Workaround: clone/build/install `kokkos/kokkos` first and pass `-DKokkos_ROOT=<install-prefix>` when configuring Kokkos Kernels.
2. **No `CONTRIBUTING.md` found at repository root**
   - Workaround: use `README.md`, `BUILD.md`, `DEVELOPER.md`, and `.github/workflows/*.yml` as authoritative contributor/build/test references.
