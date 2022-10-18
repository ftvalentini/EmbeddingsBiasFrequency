
Code to replicate _The Undesirable Dependence on Frequency of Gender Bias Metrics Based on Word Embeddings_ (2022).

The following guide was run in Ubuntu 18.04.4 LTS with python=3.9.12 and R=4.2.0. You can set up a [conda environment](#conda-environment) but it is not compulsory. 

## Requirements

Install **Python requirements**:

```
python -m pip install -r requirements.txt
```

Install **R requirements**:

```
Rscript install_packages.R
```

Clone [Stanford](https://nlp.stanford.edu/)'s GloVe repo into the repo:

```
git clone https://github.com/stanfordnlp/GloVe.git
```

or alternatively add it as submodule:

```
git submodule add https://github.com/stanfordnlp/GloVe
```

To build GloVe:

* In Linux: `cd GloVe && make`

* In Windows: `make -C "GloVe"`


## Guide

### Wikipedia corpus

1. Download 2021 English Wikipedia dump into `corpora` dir:

```
mkdir -p corpora
WIKI_URL=https://archive.org/download/enwiki-20210401
WIKI_FILE=enwiki-20210401-pages-articles.xml.bz2
wget -c -b -P corpora/ $WIKI_URL/$WIKI_FILE
# flag "-c": continue getting a partially-downloaded file
# flag "-b": go to background after startup. Output is redirected to wget-log.
```

2. Extract dump into a raw .txt file:

```
src/data/extract_wiki_dump.sh corpora/enwiki-20210401-pages-articles.xml.bz2
```

3. Create text file with one line per sentence and removing articles of less than 50 words:

```
python3 -u src/data/tokenize_and_clean_corpus.py corpora/enwiki-20210401-pages-articles.txt
```

4. Remove non alpha-numeric symbols from sentences, clean whitespaces and convert caps to lower:

```
CORPUS_IN=corpora/enwiki-20210401-pages-articles_sentences.txt
CORPUS_OUT=corpora/wiki2021.txt
src/data/clean_corpus.sh $CORPUS_IN > $CORPUS_OUT
```

See number of lines, tokens, characters in the preprocessed corpus:

```
wc corpora/wiki2021.txt
# 78051838  1748884626 10453280228 corpora/wiki2021.txt
```

### Shuffle corpus


Shuffle the corpus multiple times. Set seeds in `src/data/shuffle_corpus_multiple.sh`. Each new corpus is named as `corpora/wiki2021s<seed>.txt`.

```
CORPUS_IN=corpora/wiki2021.txt
bash src/data/shuffle_corpus_multiple.sh $CORPUS_IN
```

### Co-occurrence counts

1. Create vocabulary of original and shuffled corpora using GloVe module:

```
mkdir -p data/working &&
OUT_DIR=data/working &&
VOCAB_MINCOUNT=100 &&
IDS=(wiki2021 wiki2021s1 wiki2021s2 wiki2021s3 wiki2021s4 wiki2021s5) && 
for id in ${IDS[@]}; do
    corpus=corpora/$id.txt
    src/corpus2vocab.sh $corpus $OUT_DIR $VOCAB_MINCOUNT
done
```

2. Create co-occurrence matrices with `scipy.sparse` format (`.npz` file) using GloVe module:

```
src/corpus2cooc_multiple.sh
```

### Word embeddings


1. Download GloVe and word2vec pretrained embeddings:

```
python3 -u src/download_pretrained_we.py
```

2. Train SGNS on the corpora with `gensim` library. For each corpus, this saves a `.model` with trained model and `.npy` with the embeddings in array format. If the model is large, files with extension `.trainables.syn1neg.npy` and `.wv.vectors.npy` might be saved alongside `.model`.

```
bash src/corpus2sgns_multiple.sh
```

3. Train GloVe on the corpora and save one `.npy` file for each corpus with the vectors in array format.

```
bash src/corpus2glove_multiple.sh
```


### Bias quantification


1. Compute female vs male $Bias_{WE}$ with pre-trained word embeddings. The lists of contexts words are specified in `words_lists/` as text files. Results are saved into `results/bias_{modelname}_{A}_{B}.csv` with one row per word in the vocabulary. 

```
mkdir -p results &&
A="FEMALE" &&
B="MALE" &&
python3 -u src/pretrained2biasdf.py $A $B "glove-wiki-gigaword-300" &&
python3 -u src/pretrained2biasdf.py $A $B "word2vec-google-news-300"
```

2. Compute female vs male $Bias_{WE}$ and $Bias_{PMI}$ of the original and shuffled corpora. The lists of contexts words are specified in `words_lists/` as text files. Results are saved into `results/bias_{modelname}_{A}_{B}.csv` with one row per word in the vocabulary. 

```
A="FEMALE" &&
B="MALE" &&
nohup src/biasdf_multiple.sh $A $B &
```

### Figures

To replicate figures for $Bias_{WE}$ with pretrained word embeddings:

```
mkdir -p results/plots
R -e 'rmarkdown::render("plots_pretrained.Rmd", "html_document")'
```

Replicate tables and figures for $Bias_{WE}$ and $Bias_{PMI}$ with the original and shuffled 2021 Wikipedia with:

```
R -e 'rmarkdown::render("plots_trained.Rmd", "html_document")'
```

Results are saved as html documents.

## conda environment

You can create a `bias-frequency` conda environment to install requirements and dependencies. This is not compulsory. 

To install miniconda if needed, run:

```
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh
sha256sum Miniconda3-py37_4.10.3-Linux-x86_64.sh
bash Miniconda3-py37_4.10.3-Linux-x86_64.sh
# and follow stdout instructions to run commands with `conda`
```

To create a conda env with Python and R:

```
conda config --add channels conda-forge
conda create -n "bias-pmi" --channel=defaults python=3.9.12
conda install --channel=conda-forge r-base=4.2.0
```

Activate the environment with `conda activate bias-frequency` and install pip with `conda install pip`.
