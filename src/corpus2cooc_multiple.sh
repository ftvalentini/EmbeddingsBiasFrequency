#!/bin/bash -xe

OUT_DIR=data/working

# COOC params
VOCAB_MINCOUNT=100 # words with lower frequency are removed before windows
WINDOW_SIZE=10
DISTANCE_WEIGHTING=0 # normalized co-occurrence counts (vanilla GloVe)

IDS=(wiki2021 wiki2021s1 wiki2021s2 wiki2021s3 wiki2021s4 wiki2021s5)


for id in ${IDS[@]}; do

    corpus=corpora/$id.txt

    coocfile=$OUT_DIR/cooc-$id-V$VOCAB_MINCOUNT-W$WINDOW_SIZE-D$DISTANCE_WEIGHTING.npz
    if [[ ! -f $coocfile ]]; then
        echo "$corpus: building sparse co-occ. matrix ..."
        src/corpus2cooc.sh $corpus $OUT_DIR $VOCAB_MINCOUNT $WINDOW_SIZE $DISTANCE_WEIGHTING
        echo
    else
        echo "$coocfile exists. Skipping."
    fi

done

