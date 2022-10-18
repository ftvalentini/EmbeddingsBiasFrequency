"""Download pretrained word embeddings with gensim into $HOME/gensim-data
"""

import logging

import gensim.downloader


logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)


MODELS = [
    "glove-wiki-gigaword-100",
    "glove-wiki-gigaword-300",
    "word2vec-google-news-300",
    "fasttext-wiki-news-subwords-300",
]


def main():

    logging.info("Avaialable pretrained word embeddings in gensim:")
    print(*list(gensim.downloader.info()['models'].keys()), sep='\n')

    for model in MODELS:
        logging.info(f"Downloading {model}")
        print(gensim.downloader.load(model, return_path=True))

    logging.info("DONE")


if __name__ == '__main__':
    main()
