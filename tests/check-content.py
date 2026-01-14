import os

result = []

for root, dirs, files in os.walk("./runs/runs-rhopa64", topdown=True):
    # skip early if more than one file or any subdirs
    if len(files) == 1 and not dirs:
        result.append(root)

for d in result:
    print(d)
