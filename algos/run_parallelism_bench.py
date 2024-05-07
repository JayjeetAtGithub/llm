import os
import subprocess


if __name__ == "__main__":
    data = {}
    threads = [32, 16]
    for thread in threads:
        for _ in range(3):
            os.environ["OMP_NUM_THREADS"] = str(thread)
            output = subprocess.run(["./bin/profile_hnswlib", "hnsw", "gist", "query", "10", "debug"], capture_output=True)
            time = output.stdout.splitlines()[-1:]
            time_int = int(time[0].decode("utf-8").split()[2]) # ms
            if thread not in data:
                data[thread] = []
            data[thread].append(time_int)
    
    print(data)
