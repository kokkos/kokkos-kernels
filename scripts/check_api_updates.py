src_directories = ["batched/dense/src",
                   "batched/sparse/src",
                   "blas/src",
                   "common/src",
                   "graph/src",
                   "lapack/src",
                   "ode/src",
                   "sparse/src"]

src_doc_mapping = dict([('lapack/src/KokkosLapack_gesv.hpp', ['docs/source/API/lapack/gesv.rst']),
                        ('lapack/src/KokkosLapack_svd.hpp', ['docs/source/API/lapack/gesvd.rst']),
                        ('lapack/src/KokkosLapack_trtri.hpp', ['docs/source/API/lapack/trtri.rst'])])

diffs_list = open("./modified_files.txt", 'r')
modified_files = [line.strip() for line in diffs_list.readlines()]
diffs_list.close()

modified_public_files = []
new_apis_to_document  = []
undocumented_changes  = []

# Loop over changed files
for modified_file in modified_files:
    # Check if file belongs to one of our source directories
    if any(src_dir in modified_file for src_dir in src_directories):
        modified_public_files.append(modified_file)
        # Look for documentation associated with modified file
        doc_files = src_doc_mapping.get(modified_file)
        if doc_files == None:
            new_apis_to_document.append(modified_file)
        else:
            # Construct the intersection of modified files
            # and associated documentation files that should
            # be modified. If the intersection is empty some
            # documentation is likely missing!
            intersection = set.intersection(set(doc_files), set(modified_files))
            if len(intersection) == 0:
                undocumented_changes.append(modified_file)

return_value = 0

print("Modified public files:")
for modified_file in modified_public_files:
    print("   "+str(modified_file))
print("")

if new_apis_to_document:
    print("New undocumented public files:")
    for new_api in new_apis_to_document:
        print("   "+str(new_api))
    print("Note: you will need to update the src_doc_mapping dictionary in check_api_updates.py")
    print("")
    return_value = 1

if undocumented_changes:
    print("Likely undocumented public files:")
    for undoc_change in undocumented_changes:
        print("   "+str(undoc_change))
    return_value = 1

exit(return_value)
