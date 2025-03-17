from git import Repo

repo = Repo("kokkos-kernels")
files = repo.git.diff('origin/develop', name_only=True).split()
print(files)
