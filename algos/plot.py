import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    sns.set_style("whitegrid")
    with open("hnsw_parallelism_index.json", "r") as f:
        data = json.loads(f.read())
        print(data)

        df = pd.DataFrame(data)

        sns.barplot(x="thread", y="time", data=df, errorbar="sd", err_kws={'linewidth': 0.6}, capsize=.1, color="orange")
        plt.xlabel("No. of Threads")
        plt.ylabel("Time (ms)")
        plt.title("Indexing Duration vs Parallelism")
        plt.savefig("hnsw_parallelism_index.png")


    data_parallelism_batching = {
        "batch_size": [
            1000,1000,1000,
            500,500,500,
            100,100,100,
            50,50,50,
            10,10,10,
            1,1,1

        ],
        "time": [
            928,928,930,
            935,932,912,
            930,910,916,
            930,930,929,
            918,918,917,
            976,963,979
        ],
        "num_threads": [
            1,1,1,
            1,1,1,
            1,1,1,
            1,1,1,
            1,1,1,
            1,1,1,
        ]
    }