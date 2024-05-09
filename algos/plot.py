import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    sns.set_style("whitegrid")
    # with open("hnsw_parallelism_index.json", "r") as f:
    #     data = json.loads(f.read())
    #     print(data)

    #     df = pd.DataFrame(data)

    #     sns.barplot(x="thread", y="time", data=df, errorbar="sd", err_kws={'linewidth': 0.6}, capsize=.1, color="orange")
    #     plt.xlabel("No. of Threads")
    #     plt.ylabel("Time (ms)")
    #     plt.title("Indexing Duration vs Parallelism")
    #     plt.savefig("hnsw_parallelism_index.png")


    data_parallelism_batching = {
        "batch_size": [
            10000,10000,10000,
            5000,5000,5000,
            1000,1000,1000,
            100,100,100,
            10,10,10,
            1,1,1,

            10000,10000,10000,
            5000,5000,5000,
            1000,1000,1000,
            100,100,100,
            10,10,10,
            1,1,1
        ],
        "time": [
            1614,1541,1553,
            1654,1835,1696,
            1880,1701,1810,
            2192,2203,2128,
            5752,4655,6262,
            28065,28842,28107,

            9175,9188,9179,
            9195,9189,9201,
            9191,9182,9193,
            9183,9200,9201,
            9236,9242,9233,
            9685,9673,9681

        ],
        "mode": [
            "multi-threaded", "multi-threaded", "multi-threaded",
            "multi-threaded", "multi-threaded", "multi-threaded",
            "multi-threaded", "multi-threaded", "multi-threaded",
            "multi-threaded", "multi-threaded", "multi-threaded",
            "multi-threaded", "multi-threaded", "multi-threaded",
            "multi-threaded", "multi-threaded", "multi-threaded",

            "single-threaded", "single-threaded", "single-threaded",
            "single-threaded", "single-threaded", "single-threaded",
            "single-threaded", "single-threaded", "single-threaded",
            "single-threaded", "single-threaded", "single-threaded",
            "single-threaded", "single-threaded", "single-threaded",
            "single-threaded", "single-threaded", "single-threaded",
        ]
    }

    df = pd.DataFrame(data_parallelism_batching)
    print(df)

    sns.barplot(x="batch_size", order=["10000", "5000", "1000", "100", "10", "1"], y="time", data=df, hue="mode", errorbar="sd", err_kws={'linewidth': 0.6}, capsize=.1)
    plt.xlabel("Batch Size")
    plt.ylabel("Time (ms)")
    plt.title("Batching and Parallelism")
    plt.tight_layout()
    plt.savefig("batching.png")
