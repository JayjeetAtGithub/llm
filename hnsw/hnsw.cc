#include <iostream>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>


arrow::Result<std::shared_ptr<arrow::Table>> read_parquet_file(const std::string& filename) {
  std::cout << "Reading Parquet file: " << filename << std::endl;
  arrow::MemoryPool* pool = arrow::default_memory_pool();
  std::shared_ptr<arrow::io::RandomAccessFile> input;
  ARROW_ASSIGN_OR_RAISE(input, arrow::io::ReadableFile::Open(filename));

  std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
  ARROW_RETURN_NOT_OK(parquet::arrow::OpenFile(input, pool, &arrow_reader));

  std::shared_ptr<arrow::Table> table;
  ARROW_RETURN_NOT_OK(arrow_reader->ReadTable(&table));
  std::cout << table->ToString() << std::endl;
  return table;
}

int main() {
  std::cout << "Hello, World!" << std::endl;
  read_parquet_file("../dbpedia-entities-openai-1M/data/train-00000-of-00026-3c7b99d1c7eda36e.parquet");
  return 0;
}
