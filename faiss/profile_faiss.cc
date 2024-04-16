#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>


int main() {
    faiss::IndexFlatL2 index(d);
    printf("is_trained = %s\n", index.is_trained ? "true" : "false");
    printf("ntotal = %ld\n", index.ntotal);
    return 0;
}
