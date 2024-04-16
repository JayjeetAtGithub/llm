# Vector Database Benchmarks

In this work, we are profiling vector search operations on different indexes in different vector databases and indexing libraries. So, far we have 
profiled Qdrant, PGVector, and DiskANN. 

1. Clone the repository
```bash
git clone https://github.com/JayjeetAtGithub/vectordb-benchmarks
```

2. Build [Qdrant](https://qdrant.tech) in Release mode
```bash
./scripts/install_qdrant.sh
```

3. Start a Qdrant server
```bash
./target/release/qdrant
```

3. Download the dataset
```bash
sudo apt-get -y install git-lfs
git lfs install
git clone [dataset-uri]
```

4. Ingest data into the vector database instance
```bash
python3 bench.py --ingest --bench [benchmark-name]
```
