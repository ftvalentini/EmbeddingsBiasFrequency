#!/bin/bash -xe

OUT_DIR=data/working

VOCAB_MINCOUNT=100 # words with lower frequency are removed before windows
WINDOW_SIZE=10
DISTANCE_WEIGHTING=1 # normalized co-occurrence counts (vanilla GloVe)
VECTOR_SIZE=300
ETA=0.05 # learning rate
MAX_ITER=100
MODEL=2 # 1:W, 2:W+C
SEED=1

IDS=(wiki2021 wiki2021s1 wiki2021s2 wiki2021s3 wiki2021s4 wiki2021s5)

for id in ${IDS[@]}; do

    corpus=corpora/$id.txt
    echo "Training vectors for $corpus..."
    src/corpus2glove.sh $corpus $OUT_DIR $VOCAB_MINCOUNT \
    $WINDOW_SIZE $DISTANCE_WEIGHTING $VECTOR_SIZE $ETA $MAX_ITER $MODEL $SEED

done
