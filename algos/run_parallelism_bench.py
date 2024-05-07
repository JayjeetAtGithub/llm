import os
import sys
import json
import subprocess


if __name__ == "__main__":
    mode = str(sys.argv[1])
    print("using mode: ", mode)
    # search
    data = []
    threads = [32, 16, 8, 4, 2, 1]
    for thread in threads:
        for _ in range(3):
            os.environ["OMP_NUM_THREADS"] = str(thread)
            output = subprocess.run(["./bin/profile_hnswlib", "hnsw", "gist", mode, "10", "debug"], capture_output=True)
            print(output.stdout)
            time = output.stdout.splitlines()[-2]
            print(time)
            time_int = int(time.decode("utf-8").split()[2]) # ms
            print(f"Thread: {thread}, Time: {time_int}")
            data.append({
                "thread": thread,
                "time": time_int
            })

    print(data)
    with open(f"hnswlib_parallelism_{mode}.json", "w") as f:
        json.dump(data, f)
