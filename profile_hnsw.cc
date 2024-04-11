using namespace unum::usearch;

metric_punned_t metric(1536, metric_kind_t::l2sq_k, scalar_kind_t::f32_k);
int32_t num_vectors = 1000000;

// initialize the index
index_dense_t index = index_dense_t::make(metric);
index.reserve(num_vectors);

// reserve space for vectors
std::vector<std::vector<float32_t> vecs;
vecs.reserve(num_vectors);

for (auto &vec : vecs) {
    index.add(42, &vec[0]);
}
// auto results = index.search(&vec[0], 5); // Pass a query and limit number of results

// for (std::size_t i = 0; i != results.size(); ++i)
//     results[i].element.key, results[i].element.vector, results[i].distance;