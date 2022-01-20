# New Developers
## Coding Style
We follow google's c++ coding style. See https://google.github.io/styleguide/cppguide.html and https://github.com/kokkos/kokkos-kernels/blob/master/.clang-format for details. 
### Automate coding style via a pre-commit hook:
```bash
cat kokkos-kernels/.git/hooks/pre-commit
for FILE in $(git diff --cached --name-only | egrep '.*\.cpp$|.*\.hpp$|.*\.h$')
do
        clang-format-8 -i -style=file $FILE
        git add $FILE
done
chmod +x kokkos-kernels/.git/hooks/pre-commit
```
### Conditionally enable or disable formatting
```c++
// clang-format off
cpp code here
// clang-format on
```
##### Ensure that clang-format-8 is installed.
###### Mac
```bash
brew install clang-format-8
```

###### Ubuntu
```bash
apt install clang-format-8
```

## Comment Style
We follow doxygen style comments for both external (API) and internal members. See https://www.doxygen.nl/manual/docblocks.html for details.
Our documentation can be generated using the `-DKokkosKernels_ENABLE_DOCS:BOOL=ON` cmake flag.

In general, we prefer that the prototype has the doxygen style comment rather than the definition. If there is no prototype, then the definition should have the doxygen style comment.
### API Doxygen Style Example
```c++
/// \brief Blocking wrapper for accessing a Kokkos View.
/// \tparam ViewValueType The value type (Scalar or Vector) of each view element
/// \tparam ViewType The view type
/// \param v The view handle
/// \param m The requested row index of v
/// \param n The requested col index of v
/// \return If m and n are within the extents of v, a valid element of v;
///         otherwise, the last element of v.
///
template <class ViewValueType, class ViewType>
KOKKOS_INLINE_FUNCTION ViewValueType
access_view_bounds_check(ViewType v, int m, int n, const BoundsCheck::Yes &);
```

# Library policies

## Upcasting and downcasting
TODO

## Blocking and non-blocking interfaces
All the APIs are non-blocking unless:
1. A TPL is enabled
2. The result vector resides on the host and work is offloaded to a device

When a TPL is enabled, we follow the blocking semantics of the TPL interface.

If no TPLs are enabled, callers can avoid blocking calls by using any overload which accepts a result vector type as a template argument.

# TODOs
- [] Move library policies out of CONTRIBUTING.md?
- [] Code owners of `blas/{sparse,dense}`, `batched{sparse,dense}` review and report policy violations.
  - [] blas/sparse
  - [] blas/dense
  - [] batched/sparse
  - [] batched/dense
- [] Add up/downcosting CI checks?