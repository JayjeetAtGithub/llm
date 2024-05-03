#include "lib/hnswlib.h"
#include "utils.h"

int main(int argc, char **argv) {
    std::string dataset = argv[1];
    std::string operation = argv[2];
    int top_k = std::stoi(argv[3]);
    print_pid();

    std::cout << "Using dataset: " << dataset << std::endl;
    std::cout << "Performing operation: " << operation << std::endl;

    int M = 32;
    int ef_construction = 200;

    if (operation == "index") {
        size_t dim_learn, n_learn;
        float* data_learn;
        std::string dataset_path_learn = dataset + "/" + dataset + "_base.fvecs";
        read_dataset(dataset_path_learn.c_str(), data_learn, &dim_learn, &n_learn);
        std::cout << "Learn dataset shape: " << dim_learn << " x " << n_learn << std::endl;

        hnswlib::L2Space space(dim_learn);
        hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, n_learn, M, ef_construction);
        
        std::cout << "Adding vectors to index" << std::endl;
        #pragma omp parallel for
        for (int i = 0; i < n_learn; i++) {
            alg_hnsw->addPoint(data_learn + i * dim_learn, i);
        }

        std::string hnsw_path = "index." + dataset + ".hnswlib";
        alg_hnsw->saveIndex(hnsw_path);
        
        delete alg_hnsw;
        delete[] data_learn;
    }

    if (operation == "query") {
        size_t dim_query, n_query;
        float* data_query;
        std::string dataset_path_query = dataset + "/" + dataset + "_learn.fvecs";
        read_dataset(dataset_path_query.c_str(), data_query, &dim_query, &n_query);
        std::cout << "Query dataset shape: " << dim_query << " x " << n_query << std::endl;
        
        hnswlib::L2Space space(dim_query);
        std::string hnsw_path = "index." + dataset + ".hnswlib";
        hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
        
        #pragma omp parallel for
        for (int i = 0; i < n_query; i++) {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data_query + i * dim_query, top_k);
        }

        delete alg_hnsw;
        delete[] data_query;
    }
    
    return 0;
}
