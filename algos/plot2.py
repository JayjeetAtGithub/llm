import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":

    data_main = []

    sns.set_style("whitegrid")
    with open("hnsw_parallelism_index.json", "r") as f:
        data = json.loads(f.read())
        
        for point in data:
            new_point = {'thread': point['thread'], 'time': point['time'], 'Operation': 'index'}
            data_main.append(new_point)

    with open("hnsw_parallelism_query.json", "r") as f:
        data = json.loads(f.read())
        
        for point in data:
            new_point = {'thread': point['thread'], 'time': point['time'], 'Operation': 'search'}
            data_main.append(new_point)

    df = pd.DataFrame(data_main)

    sns.barplot(x="thread", y="time", hue="Operation", data=df, errorbar="sd", err_kws={'linewidth': 1.5}, capsize=.1)
    plt.xlabel("No. of Threads")
    plt.ylabel("Time (ms)")
    plt.title("Indexing and Search Duration vs Parallelism")
    plt.savefig("hnsw_parallelism.png")
