#!/bin/bash
set -ex

DISKANN_HOME=/mnt/workspace/DiskANN
DISKANN_DATA=${DISKANN_HOME}/build/data

# gist
DISKANN_DATA_GIST_LEARN=${DISKANN_DATA}/gist/gist_learn.fvecs
DISKANN_DATA_GIST_QUERY=${DISKANN_DATA}/gist/gist_query.fvecs
DISKANN_DATA_GIST_LEARN_FBIN=${DISKANN_DATA}/gist/gist_learn.fbin
DISKANN_DATA_GIST_QUERY_FBIN=${DISKANN_DATA}/gist/gist_query.fbin
DISKANN_DATA_GIST_GROUNDTRUTH=${DISKANN_DATA}/gist/gist_groundtruth.ivecs
DISKANN_DATA_GIST_GROUNDTRUTH_BIN=${DISKANN_DATA}/gist/gist_groundtruth.fbin
DISKANN_DATA_GIST_INDEX=${DISKANN_DATA}/gist/gist_index
DISKANN_DATA_GIST_RES=${DISKANN_DATA}/gist/res

# sift
DISKANN_DATA_SIFT_LEARN=${DISKANN_DATA}/sift/sift_learn.fvecs
DISKANN_DATA_SIFT_QUERY=${DISKANN_DATA}/sift/sift_query.fvecs
DISKANN_DATA_SIFT_LEARN_FBIN=${DISKANN_DATA}/sift/sift_learn.fbin
DISKANN_DATA_SIFT_QUERY_FBIN=${DISKANN_DATA}/sift/sift_query.fbin
DISKANN_DATA_SIFT_GROUNDTRUTH=${DISKANN_DATA}/sift/sift_groundtruth.ivecs
DISKANN_DATA_SIFT_GROUNDTRUTH_BIN=${DISKANN_DATA}/sift/sift_groundtruth.fbin
DISKANN_DATA_SIFT_INDEX=${DISKANN_DATA}/sift/sift_index
DISKANN_DATA_SIFT_RES=${DISKANN_DATA}/sift/res


if [ ! -d "${DISKANN_DATA_SIFT_LEARN_FBIN}" ]; then
    ${DISKANN_HOME}/build/apps/utils/fvecs_to_bin float ${DISKANN_DATA_SIFT_LEARN} ${DISKANN_DATA_SIFT_LEARN_FBIN}
fi

if [ ! -d "${DISKANN_DATA_SIFT_QUERY_FBIN}" ]; then
    ${DISKANN_HOME}/build/apps/utils/fvecs_to_bin float ${DISKANN_DATA_SIFT_QUERY} ${DISKANN_DATA_SIFT_QUERY_FBIN}
fi

if [ ! -d "${DISKANN_DATA_SIFT_GROUNDTRUTH_BIN}" ]; then
    ${DISKANN_HOME}/build/apps/utils/ivecs_to_bin ${DISKANN_DATA_SIFT_GROUNDTRUTH} ${DISKANN_DATA_SIFT_GROUNDTRUTH_BIN}
fi

${DISKANN_HOME}/build/apps/build_memory_index  --data_type float --dist_fn l2 --data_path ${DISKANN_DATA_SIFT_LEARN_FBIN} --index_path_prefix ${DISKANN_DATA_SIFT_INDEX} -R 32 -L 50 --alpha 1.2
${DISKANN_HOME}/build/apps/search_memory_index  --data_type float --dist_fn l2 --index_path_prefix ${DISKANN_DATA_SIFT_INDEX} --query_file ${DISKANN_DATA_SIFT_QUERY_FBIN}  --gt_file ${DISKANN_DATA_SIFT_GROUNDTRUTH_BIN} -K 10 -L 10 20 30 40 50 100 --result_path ${DISKANN_DATA_SIFT_RES}
