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

    idx = str(sys.argv[1])

    xb = fvecs_read("../algos/gist/gist_learn.fvecs")
    xq = fvecs_read("../algos/gist/gist_query.fvecs")

    if idx == "flat":
        index = faiss.IndexFlatL2(dim)
    elif idx == "ivf":
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, 100)
        index.nprobe = 8
        assert(index.is_trained == False) 
        index.train(xb)
        assert(index.is_trained == True)
    elif idx == "lsh":
        nbits = dim * 4
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

    k = 10

    s = time.time()
    D, I = index.search(xq, k)
    print(f"Search: {time.time() - s} seconds")
