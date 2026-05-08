# Copilot cloud agent instructions for `kokkos/kokkos-kernels`

## Project overview
- KokkosKernels implements local computational kernels for linear algebra (BLAS and LAPACK) and graph operations using the Kokkos shared-memory parallel programming model.
- The project supports CUDA, HIP, SYCL, OpenMP, Threads, and Serial backends (availability depends on how Kokkos is configured).
- For cloud-agent work, use a modern toolchain targeting C++20 and CMake 3.22+.
- Primary development branch is `develop` (not `main`).

## Build instructions
This repository does **not** configure standalone unless Kokkos is already built and discoverable.

1. Clone `kokkos/kokkos` beside this repository (CI commonly uses the latest release of Kokkos).
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


### Key CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `KokkosKernels_ENABLE_TESTS` | OFF | Build unit and integration tests |
| `KokkosKernels_ENABLE_EXAMPLES` | OFF | Build example programs |
| `KokkosKernels_ENABLE_PERFTESTS` | OFF | Build `perf_test` targets |
| `KokkosKernels_ENABLE_BENCHMARKS` | OFF | Build benchmark targets |
| `KokkosKernels_RUN_BENCHMARKS` | OFF | Run benchmarks from CTest/benchmark wiring |
| `KokkosKernels_ENABLE_DOCS` | OFF | Enable docs build targets |
| `KokkosKernels_ENABLE_COMPILER_WARNINGS` | OFF | Enable stricter compiler warning flags |
| `KokkosKernels_ENABLE_WARNINGS_AS_ERRORS` | OFF | Treat warnings as errors |
| `KokkosKernels_ENABLE_ALL_COMPONENTS` | ON | Enable all kernel components |
| `KokkosKernels_ENABLE_COMPONENT_BATCHED` | OFF | Enable the batched component |
| `KokkosKernels_ENABLE_COMPONENT_SPARSE` | OFF | Enable the sparse component (pulls dependent components) |
| `KokkosKernels_ENABLE_TPL_BLAS` | OFF | Enable BLAS TPL support |
| `KokkosKernels_ENABLE_TPL_LAPACK` | ON if BLAS is enabled, otherwise OFF | Enable LAPACK TPL support |
| `KokkosKernels_ENABLE_TPL_CUBLAS` | ON if CUDA-enabled Kokkos, otherwise OFF | Enable cuBLAS TPL support |
| `KokkosKernels_ENABLE_TPL_CUSPARSE` | ON if CUDA-enabled Kokkos, otherwise OFF | Enable cuSPARSE TPL support |
| `KokkosKernels_ENABLE_TPL_CUSOLVER` | ON if CUDA-enabled Kokkos, otherwise OFF | Enable cuSOLVER TPL support |
| `KokkosKernels_ENABLE_TPL_ROCBLAS` | OFF | Enable rocBLAS TPL support |
| `KokkosKernels_ENABLE_TPL_ROCSPARSE` | OFF | Enable rocSPARSE TPL support |
| `KokkosKernels_ENABLE_TPL_ROCSOLVER` | OFF | Enable rocSOLVER TPL support |

## Dependencies

| Dependency | Source | Required? |
|-----------|--------|-----------|
| Kokkos 4.7.02+ | External clone/install (CI pins `5.1.0`) | Yes |
| Google Test | Vendored in `tpls/gtest` | For tests only |
| BLAS | System install | Optional — CPU dense linear algebra |
| LAPACK | System install | Optional — auto-enabled when BLAS is enabled |
| MKL | Intel oneAPI | Optional — Intel-optimized CPU math library |
| MAGMA | System install | Optional — GPU-accelerated dense linear algebra |
| cuBLAS | CUDA Toolkit | Optional — auto-enabled with CUDA-enabled Kokkos |
| cuSPARSE | CUDA Toolkit | Optional — auto-enabled with CUDA-enabled Kokkos |
| cuSOLVER | CUDA Toolkit | Optional — auto-enabled with CUDA-enabled Kokkos |
| rocBLAS | ROCm | Optional — AMD GPU (default OFF even with HIP Kokkos) |
| rocSPARSE | ROCm | Optional — AMD GPU (default OFF even with HIP Kokkos) |
| rocSOLVER | ROCm | Optional — AMD GPU (default OFF even with HIP Kokkos) |
| ARMPL | Arm Performance Libraries | Optional — Arm CPU |
| SuperLU | System install | Optional — sparse direct solver |
| CHOLMOD | SuiteSparse | Optional — sparse Cholesky solver |
| METIS | System install | Optional — graph partitioning |
| ACCELERATE | System install | Optional - Apple-optimized math library |

## Common Pitfalls

- **Kokkos must be pre-installed**: KokkosKernels does **not** bundle Kokkos. Build will fail at CMake configure time without a valid `-DKokkos_ROOT=<path>` pointing to an installed Kokkos.
- **Include ordering**: Do NOT sort includes — `SortIncludes: false` is intentional in `.clang-format`. Never run a formatter or editor pass that reorders `#include` directives.
- **Formatter version**: CI enforces `clang-format-16`. The `.clang-format` header comments reference version 8, but always use version 16 when formatting locally to avoid diff noise.
- **Component dependencies**: Enabling `KokkosKernels_ENABLE_COMPONENT_SPARSE` or `GRAPH` automatically forces all other components (BATCHED, BLAS, LAPACK, etc.) on. Set components individually only when you need a minimal build.
- **`develop` is the integration branch**: Never compare against or target `main` for PRs or CI checks; use `develop`.
- **ETI type coverage**: Without explicitly enabling the right `KokkosKernels_INST_*` options, only the default ETI types (double, LayoutLeft, HostSpace, etc.) are pre-instantiated. Missing combos lead to linker errors in downstream code.
- **GPU TPL auto-enable**: cuBLAS/cuSPARSE/cuSOLVER are automatically `ON` when Kokkos is CUDA-enabled. Use `KokkosKernels_NO_DEFAULT_CUDA_TPLS=ON` to suppress this behavior.
- **Docs/API check**: Modifying any public header requires running `scripts/check_api_updates.py` (see the docs workflow). Skipping this will fail the `docs` CI job.

## Security

- **No secrets in code**: Never commit credentials, API keys, tokens, or passwords into source code or configuration files.
- **No sensitive data exposure**: Never share sensitive repository data (code, credentials, internal configurations) with third-party systems.
- **No new vulnerabilities**: Validate that changes do not introduce security vulnerabilities (e.g., buffer overflows, unvalidated inputs, unsafe memory access). The `codeql.yml` and `scorecards.yml` CI workflows enforce automated scanning.
- **Dependency vigilance**: Review new dependencies for known vulnerabilities before adding them. The `dependency-review.yml` workflow blocks PRs that introduce vulnerable dependencies.
- **Respect copyright**: All contributions must comply with the project's Apache-2.0 WITH LLVM-exception license. Do not generate or commit copyrighted content from external sources without explicit permission.

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

## CI/CD Workflows

**Main workflows** (`.github/workflows/`):
1. **linux.yml** - Linux sanitizer CI (`ubuntu-asan-ubsan-ci`) with Kokkos `5.1.0`, ASan/UBSan flags, build, and `ctest`.
2. **osx.yml** - macOS CI matrix (SERIAL/THREADS, Debug/Release/RelWithDebInfo) plus Accelerate-based coverage.
3. **at2.yml** - Orchestrates reusable GPU/host workflows (`h100_lychee.yml`, `v100_kumquat.yml`, `mi210.yml`, `host.yml`, `pv.yml`).
4. **format.yml** - `clang-format-16` check on changed C/C++ files against the PR base branch.
5. **docs.yml** - API-change guard (`scripts/check_api_updates.py`) and Sphinx docs build/deploy.
6. **codeql.yml** - CodeQL static analysis.
7. **dependency-review.yml** and **scorecards.yml** - Dependency and security posture checks.
8. **release.yml** - Release packaging/publication on tag workflows.

**Replicate CI locally:**
```bash
# Build and install Kokkos first
cmake -S kokkos -B kokkos/build \
  -DCMAKE_CXX_STANDARD=20 \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_COMPILER_WARNINGS=ON \
  -DKokkos_ENABLE_TESTS=OFF \
  -DKokkos_ENABLE_DEPRECATED_CODE_5=OFF \
  -DCMAKE_INSTALL_PREFIX=$PWD/kokkos/install
cmake --build kokkos/build --target install --parallel $(nproc)

# Configure/build/test Kokkos Kernels in a CI-like host setup
cmake -S kokkos-kernels -B kokkos-kernels/build \
  -DKokkos_ROOT=$PWD/kokkos/install \
  -DKokkosKernels_ENABLE_TESTS=ON \
  -DKokkosKernels_ENABLE_EXAMPLES=ON \
  -DKokkosKernels_ENABLE_COMPILER_WARNINGS=ON
cmake --build kokkos-kernels/build --parallel $(nproc)
ctest --test-dir kokkos-kernels/build --output-on-failure --timeout 7200

# Match formatting checks used by CI
clang-format-16 -i <changed_file>.{cpp,hpp,h}
```

Practical CI behavior to remember:
- Linux/OSX/AT2 PR workflows ignore docs-only changes (`**/*.md`, `docs/**`, etc.).
- Linux/OSX/AT2 workflows pin Kokkos to `5.1.0`.

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
- Never treat `*/impl` or `*::Impl` symbols as stable public API contracts.
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
