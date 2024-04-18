/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <chrono>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

#include "utils.h"

#define TOP_K 5



std::shared_ptr<faiss::Index> create_index(int index_id, size_t dim) {
    if (index_id == 0) {
        return std::make_shared<faiss::IndexFlatL2>(dim);
    } else if (index_id == 1) {
        faiss::IndexFlatL2 quantizer(dim);
        return std::make_shared<faiss::IndexIVFFlat>(&quantizer, dim, 100);
    } else if (index_id == 2) {
        return std::make_shared<faiss::IndexHNSWFlat>(dim, 32);
    }
    return nullptr;
}


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "usage: " << argv[0] << " [index_id] " << "[dataset]" << std::endl;
        exit(1);
    }

    int index_id = std::stoi(argv[1]);
    std::string dataset = argv[2];
    std::cout << "Using index: " << index_id_to_name(index_id) << "\n";
    std::cout << "Using dataset: " << dataset << std::endl;
    
    size_t dim_learn, n_learn;
    float* data_learn;
    std::string dataset_path_learn = dataset + "/" + dataset + "_base.fvecs";
    read_dataset(dataset_path_learn.c_str(), data_learn, &dim_learn, &n_learn);
    std::cout << "Read in learn dataset " << dim_learn << " x " << n_learn << std::endl;
    preview_dataset(data_learn);

    size_t dim_query, n_query;
    float* data_query;
    std::string dataset_path_query = dataset + "/" + dataset + "_query.fvecs";
    read_dataset(dataset_path_query.c_str(), data_query, &dim_query, &n_query);
    std::cout << "Read in query dataset " << dim_query << " x " << n_query << std::endl;
    preview_dataset(data_query);


    std::shared_ptr<faiss::Index> index = create_index(index_id, dim_learn);
    if (index_id == 1) {
        index->train(n_learn, data_learn);
    }
    index->add(n_learn, data_learn);
    std::string index_path =  "index." + index_id_to_name(index_id) + ".faiss";
    write_index(index.get(), index_path.c_str());
    delete data_learn;




    // if (index_id == 0) {
    //     std::cout << "Using IndexFlatL2\n";
    //     faiss::IndexFlatL2 index(dim);
    //     index.add(nb, xb);
    //     // std::cout << "Starting search....Waiting 10s\n";
    //     // std::this_thread::sleep_for(std::chrono::seconds(10));
    //     // auto s  = std::chrono::high_resolution_clock::now();
    //     // index.search(nq, xq, top_k, D, I);
    //     // auto e  = std::chrono::high_resolution_clock::now();
    //     // std::chrono::duration<double> diff = e-s;
    //     // std::cout << "Time taken for search: " << diff.count() << " s\n";

    // } else if (index_id == 1) {
    //     std::cout << "Using IndexIVFFlat\n";
    //     faiss::IndexFlatL2 quantizer(dim);
    //     faiss::IndexIVFFlat index(&quantizer, dim, 100);
    //     assert(!index.is_trained);
    //     index.train(nb, xb);
    //     assert(index.is_trained);
    //     index.add(nb, xb);
    //     // std::cout << "Starting search....Waiting 10s\n";
    //     // std::this_thread::sleep_for(std::chrono::seconds(10));
    //     // auto s = std::chrono::high_resolution_clock::now();
    //     // index.search(nq, xq, top_k, D, I);
    //     // auto e = std::chrono::high_resolution_clock::now();
    //     // std::chrono::duration<double> diff = e-s;
    //     // std::cout << "Time taken for search: " << diff.count() << " s\n";

    // } else if (index_id == 2) {
    //     std::cout << "Using IndexHNSWFlat\n";
    //     faiss::IndexHNSWFlat index(dim, 32);
    //     index.add(nb, xb);
    //     // std::cout << "Starting search....Waiting 10s\n";
    //     // std::this_thread::sleep_for(std::chrono::seconds(10));
    //     // auto s = std::chrono::high_resolution_clock::now();
    //     // index.search(nq, xq, top_k, D, I);
    //     // auto e = std::chrono::high_resolution_clock::now();
    //     // std::chrono::duration<double> diff = e-s;
    //     // std::cout << "Time taken for search: " << diff.count() << " s\n";

    // }


    // // printf("I=\n");
    // // for (int i = 0; i < nq; i++) {
    // //     for (int j = 0; j < top_k; j++)
    // //         printf("%5zd ", I[i * top_k + j]);
    // //     printf("\n");
    // // }

    // delete[] I;
    // delete[] D;
    // delete[] xb;
    // delete[] xq;

    return 0;
}
