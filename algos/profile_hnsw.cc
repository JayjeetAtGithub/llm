#include "lib/hnswlib.h"
#include "utils.h"

#define TOP_K 10

int main(int argc, char **argv) {
    std::string dataset = argv[1];
    std::string operation = argv[2];
    print_pid();

    std::cout << "Using dataset: " << dataset << std::endl;
    std::cout << "Performing operation: " << operation << std::endl;

    int M = 32;
    int ef_construction = 200;

    size_t dim, n;
    float* data;
    std::string dataset_path_learn = dataset + "/" + dataset + "_base.fvecs";
    read_dataset(dataset_path_learn.c_str(), data, &dim, &n);

    std::cout << "Dimension: " << dim << std::endl;
    std::cout << "Num Vectors: " << n << std::endl;

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, n, M, ef_construction);

    for (int i = 0; i < n; i++) {
        alg_hnsw->addPoint(data + i * dim, i);
    }

    float correct = 0;
    for (int i = 0; i < n; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, TOP_K);
        std::cout << result.size() << std::endl;
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    float recall = correct / n;
    std::cout << "Recall@" << TOP_K << recall << "\n";

    delete[] data;
    delete alg_hnsw;
    return 0;
}
