import os
import subprocess


if __name__ == "__main__":
    threads = [40, 32, 16, 8, 4, 2, 1]
    for thread in threads:
        os.environ["OMP_NUM_THREADS"] = str(thread)
        output = subprocess.run(["./bin/profile_hnswlib", "hnsw", "gist", "query", "10", "debug"], capture_output=True)
        time = output.stdout.splitlines()[-1:]
        print(time[0].decode("utf-8").split()[2])



