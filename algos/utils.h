#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <thread>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <sys/time.h>

std::ifstream::pos_type filesize(const char* filename)
{
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    return in.tellg(); 
}

void print_pid() {
    std::cout << "PID: " << getpid() << std::endl;
}

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

void preview_dataset(float* xb) {
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout << xb[i * 10 + j] << " ";
        }
        std::cout << std::endl;
    }
}

void read_dataset(const char* filename, float* &xb, size_t *d, size_t *n) {
    xb = fvecs_read(filename, d, n);
}

std::string get_index_file_name(std::string index, std::string dataset, std::string lib) {
    if (index == "hnsw_recall") {
        index = "hnsw";
    }
    return "index." + index + "." + dataset + "." + lib;
}
