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

    if (index_id == 0) {
        std::cout << "IndexFlat" << std::endl;
    } else if (index_id == 1) {
        std::cout << "IndexIVFFlat" << std::endl;
    } else if (index_id == 2){
        std::cout << "IndexHNSWFlat" << std::endl;
    } else {
        std::cout << "Invalid index id" << std::endl;
        exit(1);
    }

    // Declare parameters
    int dim = 1536;
    int nb = 1000000;
    int nq = 10;
    int top_k = 5;

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    // Randomly generate the training and test vector datasets
    float* xb = new float[dim * nb];
    float* xq = new float[dim * nq];

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

    faiss::IndexFlat index;
    
    // Create the index
    if (index_id == 0) {
        index = faiss::IndexFlatL2(dim);
    } else if (index_id == 1) {
        faiss::IndexFlatL2 quantizer(dim);
        index = faiss::IndexIVFFlat(&quantizer, dim, 100);
        assert(!index.is_trained);
        index.train(nb, xb);
        assert(index.is_trained);
    } else if (index_id == 2) {
        index = faiss::IndexHNSWFlat(dim, 32);
        index.train(nb, xb);
    }

    // Add the vectors to the index
    index.add(nb, xb);

    // Search the index
    idx_t *I = new idx_t[top_k * nq];
    float *D = new float[top_k * nq];

    index.search(nq, xq, top_k, D, I);

    printf("I=\n");
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < top_k; j++)
            printf("%5zd ", I[i * top_k + j]);
        printf("\n");
    }

    delete[] I;
    delete[] D;
    delete[] xb;
    delete[] xq;

    return 0;
}
