import faiss


if __name__ == "__main__":    

    print("Testing HNSW Flat")

    index = faiss.IndexHNSWFlat(d, 32)

    # training is not needed

    # this is the default, higher is more accurate and slower to
    # construct
    index.hnsw.efConstruction = 40

    print("add")
    # to see progress
    index.verbose = True
    index.add(xb)

    print("search")
    for efSearch in 16, 32, 64, 128, 256:
        for bounded_queue in [True, False]:
            print("efSearch", efSearch, "bounded queue", bounded_queue, end=' ')
            index.hnsw.search_bounded_queue = bounded_queue
            index.hnsw.efSearch = efSearch
            evaluate(index)
