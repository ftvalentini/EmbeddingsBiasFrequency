#!/bin/bash -xe

CORPUS_IN=$1

SEEDS=(1 2 3 4 5)

for seed in ${SEEDS[@]}; do

    corpus_out=corpora/wiki2021s$seed.txt
    echo "$corpus_out -- START"
    echo "Shuffling..."
    python3 -u src/data/shuffle_corpus.py $CORPUS_IN $corpus_out $seed
    echo "Removing empty lines / useless whitespaces..."
    sed -i 's/^ *//; s/ *$//; /^$/d' $corpus_out
    echo "$corpus_out -- DONE"

done
