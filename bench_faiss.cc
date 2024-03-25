/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include <sys/time.h>

#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    double t0 = elapsed();
    int dim = 128;
    size_t num_vectors = 200 * 1000;
    size_t num_train_vectors = 100 * 1000;

    faiss::IndexHNSW index(dim);

    std::mt19937 rng;

    { // training
        printf("[%.3f s] Generating %ld vectors in %dD for training\n",
               elapsed() - t0,
               num_train_vectors,
               dim);

        std::vector<float> trainvecs(num_train_vectors * dim);
        std::uniform_real_distribution<> distrib;
        for (size_t i = 0; i < num_train_vectors * dim; i++) {
            trainvecs[i] = distrib(rng);
        }

        printf("[%.3f s] Training the index\n", elapsed() - t0);
        index.verbose = true;

        index.train(nt, trainvecs.data());
    }

    { // I/O demo
        const char* outfilename = "/tmp/index_trained.faissindex";
        printf("[%.3f s] storing the pre-trained index to %s\n",
               elapsed() - t0,
               outfilename);

        write_index(&index, outfilename);
    }

    size_t nq;
    std::vector<float> queries;

    { // populating the database
        printf("[%.3f s] Building a dataset of %ld vectors to index\n",
               elapsed() - t0,
               nb);

        std::vector<float> database(nb * dim);
        std::uniform_real_distribution<> distrib;
        for (size_t i = 0; i < nb * dim; i++) {
            database[i] = distrib(rng);
        }

        printf("[%.3f s] Adding the vectors to the index\n", elapsed() - t0);

        index.add(nb, database.data());

        printf("[%.3f s] imbalance factor: %g\n",
               elapsed() - t0,
               index.invlists->imbalance_factor());

        // remember a few elements from the database as queries
        int i0 = 1234;
        int i1 = 1243;

        nq = i1 - i0;
        queries.resize(nq * dim);
        for (int i = i0; i < i1; i++) {
            for (int j = 0; j < dim; j++) {
                queries[(i - i0) * dim + j] = database[i * dim + j];
            }
        }
    }

    { // searching the database
        int k = 5;
        printf("[%.3f s] Searching the %d nearest neighbors "
               "of %ld vectors in the index\n",
               elapsed() - t0,
               k,
               nq);

        std::vector<faiss::idx_t> nns(k * nq);
        std::vector<float> dis(k * nq);

        index.search(nq, queries.data(), k, dis.data(), nns.data());

        printf("[%.3f s] Query results (vector ids, then distances):\n",
               elapsed() - t0);

        for (int i = 0; i < nq; i++) {
            printf("query %2d: ", i);
            for (int j = 0; j < k; j++) {
                printf("%7ld ", nns[j + i * k]);
            }
            printf("\n     dis: ");
            for (int j = 0; j < k; j++) {
                printf("%7g ", dis[j + i * k]);
            }
            printf("\n");
        }

        printf("note that the nearest neighbor is not at "
               "distance 0 due to quantization errors\n");
    }

    return 0;
}