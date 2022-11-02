import os

# check that a file exists and is non-empty
def non_empty_file(path):
    return os.path.exists(path) and os.path.getsize(path) > 0