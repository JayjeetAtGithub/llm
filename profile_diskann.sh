#!/bin/bash
set -ex

DISKANN_HOME=/mnt/workspace/DiskANN
DISKANN_DATA=${DISKANN_HOME}/build/data

# gist
DISKANN_DATA_GIST_BASE=${DISKANN_DATA}/gist/gist_base.fvecs
DISKANN_DATA_GIST_LEARN=${DISKANN_DATA}/gist/gist_learn.fvecs
DISKANN_DATA_GIST_QUERY=${DISKANN_DATA}/gist/gist_query.fvecs
DISKANN_DATA_GIST_BASE_FBIN=${DISKANN_DATA}/gist/gist_base.fbin
DISKANN_DATA_GIST_LEARN_FBIN=${DISKANN_DATA}/gist/gist_learn.fbin
DISKANN_DATA_GIST_QUERY_FBIN=${DISKANN_DATA}/gist/gist_query.fbin
DISKANN_DATA_GIST_GROUNDTRUTH_1=${DISKANN_DATA}/gist/gist_groundtruth_1
DISKANN_DATA_GIST_GROUNDTRUTH_2=${DISKANN_DATA}/gist/gist_groundtruth_2
DISKANN_DATA_GIST_INDEX=${DISKANN_DATA}/gist/gist_index
DISKANN_DATA_GIST_RES=${DISKANN_DATA}/gist/gist_res

# sift
DISKANN_DATA_SIFT_BASE=${DISKANN_DATA}/sift/sift_base.fvecs
DISKANN_DATA_SIFT_LEARN=${DISKANN_DATA}/sift/sift_learn.fvecs
DISKANN_DATA_SIFT_QUERY=${DISKANN_DATA}/sift/sift_query.fvecs
DISKANN_DATA_SIFT_BASE_FBIN=${DISKANN_DATA}/sift/sift_base.fbin
DISKANN_DATA_SIFT_LEARN_FBIN=${DISKANN_DATA}/sift/sift_learn.fbin
DISKANN_DATA_SIFT_QUERY_FBIN=${DISKANN_DATA}/sift/sift_query.fbin
DISKANN_DATA_SIFT_GROUNDTRUTH_1=${DISKANN_DATA}/sift/sift_groundtruth_1
DISKANN_DATA_SIFT_GROUNDTRUTH_2=${DISKANN_DATA}/sift/sift_groundtruth_2
DISKANN_DATA_SIFT_INDEX=${DISKANN_DATA}/sift/sift_index
DISKANN_DATA_SIFT_RES=${DISKANN_DATA}/sift/sift_res

# convert the datasets into bin format
if [ ! -d "${DISKANN_DATA_SIFT_BASE_FBIN}" ]; then
    ${DISKANN_HOME}/build/apps/utils/fvecs_to_bin float ${DISKANN_DATA_SIFT_BASE} ${DISKANN_DATA_SIFT_BASE_FBIN}
fi

if [ ! -d "${DISKANN_DATA_SIFT_LEARN_FBIN}" ]; then
    ${DISKANN_HOME}/build/apps/utils/fvecs_to_bin float ${DISKANN_DATA_SIFT_LEARN} ${DISKANN_DATA_SIFT_LEARN_FBIN}
fi

if [ ! -d "${DISKANN_DATA_SIFT_QUERY_FBIN}" ]; then
    ${DISKANN_HOME}/build/apps/utils/fvecs_to_bin float ${DISKANN_DATA_SIFT_QUERY} ${DISKANN_DATA_SIFT_QUERY_FBIN}
fi

# compute the groundtruth for query and base datasets
if [ ! -d "${DISKANN_DATA_SIFT_GROUNDTRUTH_1}" ]; then
   ${DISKANN_HOME}/build/apps/utils/compute_groundtruth  --data_type float --dist_fn l2 --base_file ${DISKANN_DATA_SIFT_LEARN_FBIN}  --query_file  ${DISKANN_DATA_SIFT_QUERY_FBIN} --gt_file ${DISKANN_DATA_SIFT_GROUNDTRUTH_1} --K 100
fi

if [ ! -d "${DISKANN_DATA_SIFT_GROUNDTRUTH_2}" ]; then
   ${DISKANN_HOME}/build/apps/utils/compute_groundtruth  --data_type float --dist_fn l2 --base_file ${DISKANN_DATA_SIFT_LEARN_FBIN}  --query_file  ${DISKANN_DATA_SIFT_BASE_FBIN} --gt_file ${DISKANN_DATA_SIFT_GROUNDTRUTH_2} --K 100
fi

${DISKANN_HOME}/build/apps/build_memory_index  --data_type float --dist_fn l2 --data_path ${DISKANN_DATA_SIFT_LEARN_FBIN} --index_path_prefix ${DISKANN_DATA_SIFT_INDEX} -R 32 -L 50 --alpha 1.2
${DISKANN_HOME}/build/apps/search_memory_index  --data_type float --dist_fn l2 --index_path_prefix ${DISKANN_DATA_SIFT_INDEX} --query_file ${DISKANN_DATA_SIFT_BASE_FBIN}  --gt_file ${DISKANN_DATA_SIFT_GROUNDTRUTH_2} -K 10 -L 10 20 30 40 50 100 -T 1 --result_path ${DISKANN_DATA_SIFT_RES}
