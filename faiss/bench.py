import time
import faiss
import numpy as np
from datasets import DatasetSIFT1M

def evaluate(index, k = 5):
    # for timing with a single core
    # faiss.omp_set_num_threads(1)

    t0 = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()

    missing_rate = (I == -1).sum() / float(k * nq)
    recall_at_1 = (I == gt[:, :1]).sum() / float(nq)
    print("\t %7.3f ms per query, R@1 %.4f, missing rate %.4f" % (
        (t1 - t0) * 1000.0 / nq, recall_at_1, missing_rate))


if __name__ == "__main__":    

    ds = DatasetSIFT1M()
    xq = ds.get_queries()
    xb = ds.get_database()
    gt = ds.get_groundtruth()
    xt = ds.get_train()

    nq, d = xq.shape

    print("Benchmarking HNSW Flat")
    index = faiss.IndexHNSWFlat(d, 32)

    index.hnsw.efConstruction = 40

    print("add")
    index.add(xb)

    for efSearch in 16, 32:
        for bounded_queue in [True, False]:
            print("efSearch", efSearch, "bounded queue", bounded_queue, end=' ')
            index.hnsw.search_bounded_queue = bounded_queue
            index.hnsw.efSearch = efSearch
            evaluate(index)
