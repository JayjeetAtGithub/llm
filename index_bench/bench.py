import os
import sys
import time
import faiss
import numpy as np


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


if __name__ == "__main__":
    dim = 960
    top_k = 100
    idx = str(sys.argv[1])

    xb = fvecs_read("../algos/gist/gist_base.fvecs")
    print("Shape of xb: ", xb.shape)

    xq = fvecs_read("../algos/gist/gist_query.fvecs")
    xq = xq[0].reshape(1, xq.shape[1])
    print("Shape of xq: ", xq.shape)

    gt = ivecs_read("../algos/gist/gist_groundtruth.ivecs")
    gt = gt[0].reshape(1, gt.shape[1])
    print("Shape of gt: ", gt.shape)

    if not os.path.exists(f"index.{idx}.faiss"):
        if idx == "flat":
            index = faiss.IndexFlatL2(dim)
        elif idx == "ivf":
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, 100)
            index.nprobe = 10
            assert(index.is_trained == False) 
            index.train(xb)
            assert(index.is_trained == True)
        elif idx == "lsh":
            nbits = 16 * dim
            index = faiss.IndexLSH(dim, nbits)
        elif idx == "hnsw":
            M = 64
            ef_search = 32 
            ef_construction = 64
            index = faiss.IndexHNSWFlat(dim, M)
            index.hnsw.efConstruction = ef_construction
            index.hnsw.efSearch = ef_search

        s = time.time()
        index.add(xb)
        print(f"Index Build: {time.time() - s} seconds")
        faiss.write_index(index, f"index.{idx}.faiss")

    index = faiss.read_index(f"index.{idx}.faiss")
    s = time.time()
    D, I = index.search(xq, top_k)
    print(f"Search: {time.time() - s} seconds")

    ks = [1, 10, 100]

    for k in ks:
        recall_at_k = (I[:, :k] == gt[:, :k]).sum() / k
        print("recall@%d: %.3f" % (k, recall_at_k))
