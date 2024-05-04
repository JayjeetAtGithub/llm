#include "lib/hnswlib.h"
#include "utils.h"

#define MAX_ELEMENTS 10e6

int main(int argc, char **argv) {
    if (argc < 5) {
        std::cout << "usage: " << argv[0] << " [index (hnsw/flat/hnsw_recall)] " << "[dataset (siftsmall/sift/gist/bigann)] " << "[operation (index/query)]" << " [top_k]" << std::endl;
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

    int M = 2<<4;
    int ef_construction = 200;

    if (operation == "index") {
        size_t dim_learn, n_learn;
        float* data_learn;
        std::string dataset_path_learn = dataset + "/" + dataset + "_base.fvecs";
        read_dataset(dataset_path_learn.c_str(), data_learn, &dim_learn, &n_learn);
        std::cout << "[INFO] learn dataset shape: " << dim_learn << " x " << n_learn << std::endl;
        preview_dataset(data_learn);
        
        hnswlib::L2Space space(dim_learn);

        if (index == "hnsw" || index == "hnsw_recall") {
            std::cout << "[INFO] performing hnsw indexing" << std::endl;
            hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, n_learn, M, ef_construction);

            auto s = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for
            for (int i = 0; i < n_learn; i++) {
                alg_hnsw->addPoint(data_learn + i * dim_learn, i);
            }
            auto e = std::chrono::high_resolution_clock::now();
            std::cout << "[TIME] hnsw_index: " << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() << " ms" << std::endl;

            std::string hnsw_path = "index." + dataset + ".hnswlib";
            alg_hnsw->saveIndex(hnsw_path);
            std::cout << "[FILESIZE] hnsw_index_size: " << alg_hnsw->indexFileSize() << " bytes" << std::endl;
            
            delete alg_hnsw;
        }

        if (index == "flat" || index == "hnsw_recall") {
            std::cout << "[INFO] performing flat indexing" << std::endl;
            hnswlib::BruteforceSearch<float>* alg_flat = new hnswlib::BruteforceSearch<float>(&space, MAX_ELEMENTS);

            auto s = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for
            for (int i = 0; i < n_learn; i++) {
                alg_flat->addPoint(data_learn + i * dim_learn, i);
            }
            auto e = std::chrono::high_resolution_clock::now();
            std::cout << "[TIME] flat_index: " << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() << " ms" << std::endl;

            std::string flat_path = "index." + dataset + ".flat";
            alg_flat->saveIndex(flat_path);
            std::cout << "[FILESIZE] flat_index_size: " << filesize(flat_path.c_str()) << " bytes" << std::endl;

            delete alg_flat;
        }
        
        delete[] data_learn;
    }

    if (operation == "query") {
        size_t dim_query, n_query;
        float* data_query;
        std::string dataset_path_query = dataset + "/" + dataset + "_learn.fvecs";
        read_dataset(dataset_path_query.c_str(), data_query, &dim_query, &n_query);
        std::cout << "[INFO] query dataset shape: " << dim_query << " x " << n_query << std::endl;
        preview_dataset(data_query);

        std::unordered_map<int, std::vector<int>> results_hnsw_map;
        std::unordered_map<int, std::vector<int>> results_flat_map;

        if (index == "hnsw_recall") {
            results_hnsw_map.reserve(n_query);
            results_flat_map.reserve(n_query);
            for (int i = 0; i < n_query; i++) {
                results_hnsw_map[i] = std::vector<int>(top_k, 0);
                results_flat_map[i] = std::vector<int>(top_k, 0);
            }
        }

        hnswlib::L2Space space(dim_query);

        if (index == "hnsw" || index == "hnsw_recall") {
            n_query = 100000;
            std::string hnsw_path = "index." + dataset + ".hnswlib";
            hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
            std::cout << "[INFO] hnsw index loaded" << std::endl;
            std::cout << "[INFO] start profiler....waiting for 20 seconds" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(20));

            std::cout << "[INFO] starting query hnsw for " << n_query << " queries" << std::endl;
            auto s = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < n_query; i++) {
                std::priority_queue<std::pair<float, hnswlib::labeltype>> result_hnsw = alg_hnsw->searchKnn(data_query + i * dim_query, top_k);
                if (index == "hnsw_recall") {
                    std::cout << "[INFO] saving kNN result for recall calculation" << std::endl;
                    for (int j = 0; j < top_k; j++) {
                        results_hnsw_map[i][j] = result_hnsw.top().second;
                        result_hnsw.pop();
                    }
                    assert(results_hnsw_map[i].size() == top_k);
                }
            }
            auto e = std::chrono::high_resolution_clock::now();
            std::cout << "[TIME] hnsw_query: " << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() << " ms" << std::endl;

            delete alg_hnsw;
        }

        if (index == "flat" || index == "hnsw_recall") {
            n_query = 100;
            std::string flat_path = "index." + dataset + ".flat";
            hnswlib::BruteforceSearch<float>* alg_flat = new hnswlib::BruteforceSearch<float>(&space, flat_path);
            std::cout << "[INFO] flat index loaded" << std::endl;
            std::cout << "[INFO] start profiler....waiting for 20 seconds" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(20));

            std::cout << "[INFO] starting query flat for " << n_query << " queries" << std::endl;
            auto s = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < n_query; i++) {
                std::priority_queue<std::pair<float, hnswlib::labeltype>> result_flat = alg_flat->searchKnn(data_query + i * dim_query, top_k);
                if (index == "hnsw_recall") {
                    std::cout << "[INFO] saving kNN result for recall calculation" << std::endl;
                    for (int j = 0; j < top_k; j++) {
                        results_flat_map[i][j] = result_flat.top().second;
                        result_flat.pop();
                    }
                    assert(results_flat_map[i].size() == top_k);
                }
            }
            auto e = std::chrono::high_resolution_clock::now();
            std::cout << "[TIME] flat_query: " << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() << " ms" << std::endl;

            delete alg_flat;
        }

        delete[] data_query;

        if (index == "hnsw_recall") {
            std::cout << "[INFO] calculating recall@" << top_k << std::endl;
            std::vector<double> recalls(n_query);
            for (int i = 0; i < n_query; i++) {
                auto v1 = results_flat_map[i];
                auto v2 = results_hnsw_map[i];
                int correct = 0;            
                for (int j = 0; j < v2.size(); j++) {
                    if (std::find(v1.begin(), v1.end(), v2[j]) != v1.end()) {
                        correct++;
                    }
                }
                recalls[i] = (float)correct / top_k;         
            }
            assert(recalls.size() == n_query);
            std::cout << "[RECALL] mean recall@" << top_k << ": " << std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size() << std::endl;
        }
    }
    
    return 0;
}
