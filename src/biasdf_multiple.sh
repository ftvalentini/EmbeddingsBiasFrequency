#!/bin/bash -xe

# GloVe + word2vec + PMI params
WINDOW=10 # window size
MINCOUNT=100 # vocab min count
# GloVe + word2vec params
SIZE=300 # vector dimension
SEED=1 # random seed
# word2vec params
SG=1 # skipgram
# GloVe params
ETA=0.05 # initial learning rate
ITER=100 # max iterations
DIST_GLOVE=1 # distance weighting
# PMI params
DIST_COOC=0 # no distance weighting
SMOOTHING=0.01 # epsilon smoothing

# context words
A=${1:-"FEMALE"}
B=${2:-"MALE"}

# corpora
IDS=(wiki2021 wiki2021s1 wiki2021s2 wiki2021s3 wiki2021s4 wiki2021s5)

for id in ${IDS[@]}; do

    vocabfile="data/working/vocab-$id-V$MINCOUNT.txt"

    # SGNS:
    embedfile=data/working/w2v-$id-V$MINCOUNT-W$WINDOW-D$SIZE-SG$SG-S$SEED.npy
    outfile=results/bias_sgns-$id-$A-$B.csv
    python3 -u src/we2biasdf.py \
        --vocab $vocabfile --matrix $embedfile --a $A --b $B --out $outfile

    # GloVe
    embedfile=data/working/glove-$id-V$MINCOUNT-W$WINDOW-D$DIST_GLOVE-D$SIZE-R$ETA-E$ITER-M2-S$SEED.npy
    outfile=results/bias_glove-$id-$A-$B.csv
    python3 -u src/we2biasdf.py \
        --vocab $vocabfile --matrix $embedfile --a $A --b $B --out $outfile
    
    # PMI
    coocfile=data/working/cooc-${id}-V${MINCOUNT}-W${WINDOW}-D${DIST_COOC}.npz
    outfile=results/bias_pmi-$id-$A-$B-s${SMOOTHING}.csv
    python3 -u src/cooc2biasdf.py \
        --vocab $vocabfile --cooc $coocfile --a $A --b $B --out $outfile --smoothing $SMOOTHING

done
