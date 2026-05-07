# Copilot cloud agent instructions for `kokkos/kokkos-kernels`

## What this repository is
- C++ performance-portable kernel library in the Kokkos ecosystem.
- Main public headers are in `*/src`; implementations are commonly in `*/src/impl`.
- Tests are primarily under `*/unit_test` and `perf_test`.
- Documentation sources are under `docs/source`.

## Branches and CI behavior
- Active branches in workflows are `develop` and `master`.
- Linux/OSX PR workflows ignore docs-only changes (`**/*.md`, `docs/**`, etc.), so docs-only PRs generally do not trigger those heavy builds.
- Formatting CI (`.github/workflows/format.yml`) runs `clang-format-16` on changed C/C++ headers/sources.

## First-time setup that works reliably
This repo does **not** build standalone unless Kokkos is installed or available.

Use the same pattern as CI:
1. Check out `kokkos/kokkos` (CI uses `5.1.0`).
2. Configure/build/install Kokkos.
3. Configure Kokkos Kernels with `-DKokkos_ROOT=<kokkos-install-prefix>`.

Minimal host setup example:
```bash
# from a workspace directory containing both repos
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
ctest --test-dir kokkos-kernels/build --output-on-failure --timeout 3600
```

## Validation guidance by change type
- **Docs-only changes**:
  - Run API-doc consistency check used in CI:
    ```bash
    git fetch origin develop:refs/remotes/origin/develop
    git diff --name-only origin/develop > modified_files.txt
    python3 scripts/check_api_updates.py
    ```
  - Build docs:
    ```bash
    python3 -m pip install --require-hashes -r docs/build_requirements.txt
    make -C docs html
    ```
- **C++ code changes**:
  - Build with tests enabled (`-DKokkosKernels_ENABLE_TESTS=ON`).
  - Run focused tests first with `ctest -R <regex> --output-on-failure` before full test runs.
  - Run clang-format-16 on touched `*.cpp`, `*.hpp`, `*.h` files.

## Common pitfalls and efficient work patterns
- Prefer mirroring options from `.github/workflows/linux.yml` and `.github/workflows/osx.yml` to match CI.
- Many workflows use warning/error strictness flags; keep changes warning-clean where possible.
- For markdown/doc-only work, avoid full C++ rebuilds unless requested.

## Errors encountered during onboarding and workarounds
1. **CMake configure error: missing Kokkos package**
   - Error: `Could not find a package configuration file provided by "Kokkos"`.
   - Workaround: clone/build/install `kokkos/kokkos` first and pass `-DKokkos_ROOT=<install-prefix>` when configuring Kokkos Kernels.
2. **No `CONTRIBUTING.md` found at repository root**
   - Workaround: use `README.md`, `BUILD.md`, `DEVELOPER.md`, and `.github/workflows/*.yml` as authoritative contributor/build/test references.
