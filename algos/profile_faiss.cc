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

#define TOP_K 10


std::shared_ptr<faiss::Index> create_index(int index_id, size_t dim) {
    if (index_id == 0) {
        return std::make_shared<faiss::IndexFlatL2>(dim);
    } else if (index_id == 1) {
        std::shared_ptr<faiss::IndexFlatL2> quantizer = std::make_shared<faiss::IndexFlatL2>(dim);
        return std::make_shared<faiss::IndexIVFFlat>(quantizer.get(), dim, 100);
    } else if (index_id == 2) {
        return std::make_shared<faiss::IndexHNSWFlat>(dim, 32);
    }
    return nullptr;
}


int main(int argc, char** argv) {
    if (argc < 5) {
        std::cout << "usage: " << argv[0] << " [index (hnsw/flat)] " << "[dataset (siftsmall/sift/gist/bigann)] " << "[operation (index/query)]" << " [top_k]" << std::endl;
    }

    std::string index = argv[1];
    std::string dataset = argv[2];
    std::string operation = argv[3];
    int top_k = std::stoi(argv[4]);
    print_pid();

    std::cout << "[ARG] index: " << index << std::endl;
    std::cout << "[ARG] dataset: " << dataset << std::endl;
    std::cout << "[ARG] operation: " << operation << std::endl;
    std::cout << "[ARG] top_k: " << top_k << std::endl;

    if (operation == "index") {
        size_t dim_learn, n_learn;
        float* data_learn;
        std::string dataset_path_learn = dataset + "/" + dataset + "_base.fvecs";
        read_dataset(dataset_path_learn.c_str(), data_learn, &dim_learn, &n_learn);
        std::cout << "Learn dataset shape: " << dim_learn << " x " << n_learn << std::endl;
        preview_dataset(data_learn);

        std::shared_ptr<faiss::Index> index = create_index(index_id, dim_learn);
        if (index_id == 1) {
            index->train(n_learn, data_learn);
        }
        index->add(n_learn, data_learn);
        std::string index_path = get_index_file_name(index_id, dataset);
        write_index(index.get(), index_path.c_str());
        delete data_learn;
    }

    if (operation == "query") {
        size_t dim_query, n_query;
        float* data_query;
        std::string dataset_path_query = dataset + "/" + dataset + "_learn.fvecs";
        read_dataset(dataset_path_query.c_str(), data_query, &dim_query, &n_query);
        std::cout << "Query dataset shape: " << dim_query << " x " << n_query << std::endl;
        preview_dataset(data_query);

        std::vector<faiss::idx_t> nns(TOP_K * n_query);
        std::vector<float> dis(TOP_K * n_query);

        std::string index_path = get_index_file_name(index_id, dataset);
        faiss::Index* index = faiss::read_index(index_path.c_str());

        std::cout << "Sleeping for 20 seconds..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(20));

        std::cout << "Starting search..." << std::endl;

        auto s = std::chrono::high_resolution_clock::now();
        index->search(n_query, data_query, TOP_K, dis.data(), nns.data());
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        std::cout << "Search took: " << diff.count() << " s" << std::endl;
        delete data_query;
    }

    return 0;
}
