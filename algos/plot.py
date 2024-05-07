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

        sns.barplot(x="thread", y="time", data=df, errorbar="sd", err_kws={'linewidth': 1.5}, capsize=.1)
        plt.xlabel("No. of Threads")
        plt.ylabel("Time (ms)")
        plt.title("Indexing Duration vs Parallelism")
        plt.savefig("hnsw_parallelism_index.png")
