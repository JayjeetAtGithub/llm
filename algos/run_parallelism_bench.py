import os
import subprocess


if __name__ == "__main__":
    threads = [1, 2, 4, 8, 16, 32, 40]
    for thread in threads:
        os.environ["OMP_NUM_THREADS"] = str(thread)
        output = subprocess.run(["./bin/profile_hnswlib", "hnsw", "gist", "query", "10", "debug"])
        time = output.stdout.splitlines()[:-1]
        print(time)



