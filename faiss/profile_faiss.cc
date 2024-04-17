/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <cstdlib>
#include <random>
#include <iostream>
#include <chrono>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexHNSW.h>

using idx_t = faiss::idx_t;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "usage: " << argv[0] << " [index_id]" << std::endl;
        exit(1);
    }

    int index_id = std::stoi(argv[1]);

    // Declare parameters
    int dim = 512;
    int nb = 100000;
    int nq = 10000;
    int top_k = 5;

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float* xb = new float[dim * nb];
    float* xq = new float[dim * nq];
    idx_t *I = new idx_t[top_k * nq];
    float *D = new float[top_k * nq];
    
    auto s = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < dim; j++)
            xb[dim * i + j] = distrib(rng);
        xb[dim * i] += i / 1000.;
    }
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < dim; j++)
            xq[dim * i + j] = distrib(rng);
        xq[dim * i] += i / 1000.;
    }
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e-s;
    std::cout << "Time taken for data generation: " << diff

    if (index_id == 0) {
        std::cout << "Using IndexFlatL2\n";
        faiss::IndexFlatL2 index(dim);
        index.add(nb, xb);
        auto s  = std::chrono::high_resolution_clock::now();
        index.search(nq, xq, top_k, D, I);
        auto e  = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e-s;
        std::cout << "Time taken for search: " << diff.count() << " s\n";

    } else if (index_id == 1) {
        std::cout << "Using IndexIVFFlat\n";
        faiss::IndexFlatL2 quantizer(dim);
        faiss::IndexIVFFlat index(&quantizer, dim, 100);
        assert(!index.is_trained);
        index.train(nb, xb);
        assert(index.is_trained);
        index.add(nb, xb);
        auto s = std::chrono::high_resolution_clock::now();
        index.search(nq, xq, top_k, D, I);
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e-s;
        std::cout << "Time taken for search: " << diff.count() << " s\n";

    } else if (index_id == 2) {
        std::cout << "Using IndexHNSWFlat\n";
        faiss::IndexHNSWFlat index(dim, 32);
        index.add(nb, xb);
        auto s = std::chrono::high_resolution_clock::now();
        index.search(nq, xq, top_k, D, I);
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e-s;
        std::cout << "Time taken for search: " << diff.count() << " s\n";

    }


    // printf("I=\n");
    // for (int i = 0; i < nq; i++) {
    //     for (int j = 0; j < top_k; j++)
    //         printf("%5zd ", I[i * top_k + j]);
    //     printf("\n");
    // }

    delete[] I;
    delete[] D;
    delete[] xb;
    delete[] xq;

    return 0;
}
