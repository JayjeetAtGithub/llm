#include <cassert>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <vector>
#include <thread>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

#include "utils.h"


std::shared_ptr<faiss::Index> create_index(std::string index, size_t dim) {
    int M = 2<<4;
    if (index == "flat") {
        return std::make_shared<faiss::IndexFlatL2>(dim);
    } else if (index == "hnsw") {
        return std::make_shared<faiss::IndexHNSWFlat>(dim, M);
    }
    return nullptr;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cout << "usage: " << argv[0] << " [index (hnsw/flat)] [dataset (siftsmall/sift/gist/bigann)] [operation (index/query)] [top_k] [mode(debug/profile)]" << std::endl;
    }

    std::string index = argv[1];
    std::string dataset = argv[2];
    std::string operation = argv[3];
    int top_k = std::stoi(argv[4]);
    std::string mode = argv[5];
    print_pid();

    std::cout << "[ARG] index: " << index << std::endl;
    std::cout << "[ARG] dataset: " << dataset << std::endl;
    std::cout << "[ARG] operation: " << operation << std::endl;
    std::cout << "[ARG] top_k: " << top_k << std::endl;
    std::cout << "[ARG] mode: " << mode << std::endl;

    if (operation == "index") {
        size_t dim_learn, n_learn;
        float* data_learn;
        std::string dataset_path_learn = dataset + "/" + dataset + "_base.fvecs";
        read_dataset(dataset_path_learn.c_str(), data_learn, &dim_learn, &n_learn);
        std::cout << "[INFO] learn dataset shape: " << dim_learn << " x " << n_learn << std::endl;
        preview_dataset(data_learn);

        std::cout << "[INFO] performing " << index << " indexing" << std::endl;
        std::shared_ptr<faiss::Index> idx = create_index(index, dim_learn);
        
        auto s = std::chrono::high_resolution_clock::now();
        idx->add(n_learn, data_learn);
        auto e = std::chrono::high_resolution_clock::now();
        std::cout << "[TIME] " << index << "_index: " << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() << " ms" << std::endl;

        std::string index_path = get_index_file_name(index, dataset);
        write_index(idx.get(), index_path.c_str());
        std::cout << "[FILESIZE] " << index << "_index_size: " << filesize(index_path.c_str()) << " bytes" << std::endl;

        delete[] data_learn;
    }

    if (operation == "query") {
        size_t dim_query, n_query;
        float* data_query;
        std::string dataset_path_query = dataset + "/" + dataset + "_learn.fvecs";
        read_dataset(dataset_path_query.c_str(), data_query, &dim_query, &n_query);
        std::cout << "[INFO] query dataset shape: " << dim_query << " x " << n_query << std::endl;
        preview_dataset(data_query);

        if (index == "hnsw") {
            n_query = 100000;
        } else {
            n_query = 100;
        }

        std::vector<faiss::idx_t> nns(top_k * n_query);
        std::vector<float> dis(top_k * n_query);

        std::string index_path = get_index_file_name(index, dataset);
        faiss::Index* idx = faiss::read_index(index_path.c_str());
        std::cout << "[INFO] " << index << " index loaded" << std::endl;

        if (mode == "profile") {
            std::cout << "[INFO] start profiler....waiting for 20 seconds" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(20));
        }

        std::cout << "[INFO] starting query " << index << " for " << n_query << " queries" << std::endl;
        auto s = std::chrono::high_resolution_clock::now();
        idx->search(n_query, data_query, top_k, dis.data(), nns.data());
        auto e = std::chrono::high_resolution_clock::now();
        std::cout << "[TIME] " << index  << "_query: " << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() << " ms" << std::endl;

        delete[] data_query;
    }

    return 0;
}
