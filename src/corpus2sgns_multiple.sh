#!/bin/bash -xe

VOCAB_MINCOUNT=100

SG=1 # 0:cbow, 1:sgns
SIZE=300 # vector size
WINDOW=10 # window size
MINCOUNT=100
SEED=1

IDS=(wiki2021 wiki2021s1 wiki2021s2 wiki2021s3 wiki2021s4 wiki2021s5)

for id in ${IDS[@]}; do

    corpus=corpora/$id.txt
    vocabfile="data/working/vocab-$id-V$VOCAB_MINCOUNT.txt"
    echo "Training vectors for $corpus..."
    python3 -u src/corpus2sgns.py --corpus $corpus --vocab $vocabfile \
    --size $SIZE --window $WINDOW --count $MINCOUNT --sg $SG --seed $SEED
    echo

done

