import os
import sys


def read_file(file_path):
    with open(file_path, "r") as f:
        contents = f.read()
        contents = contents.strip()
        contents = contents.replace("\n", " ")
        return contents


if __name__ == "__main__":
    kernel_source = "kernel.txt"
    kernel_file = open(kernel_source, "w")

    for dirpath, dnames, fnames in os.walk("./linux"):
        for f in fnames:
            print("Reading file: ", f)
            if f.endswith(".c"):
                contents = read_file(os.path.join(dirpath, f))
                kernel_file.write(contents)
            if f.endswith(".h"):
                contents = read_file(os.path.join(dirpath, f))
                kernel_file.write(contents)

    kernel_file.close()
3