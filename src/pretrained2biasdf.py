
import argparse
import logging

import pandas as pd
import numpy as np
import gensim.downloader

from utils.figures import gensim2biasdf


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('a', type=str, default="FEMALE")
    parser.add_argument('b', type=str, default="MALE")
    parser.add_argument('modelname', type=str, default="word2vec-google-news-300")
    args = parser.parse_args()

    logging.info("Getting attribute words...")
    words_a = [
        line.strip().lower() for line in open(f'words_lists/{args.a}.txt','r')]
    words_b = [
        line.strip().lower() for line in open(f'words_lists/{args.b}.txt','r')]

    logging.info(f"Loading {args.modelname} model...")
    # loads previously downloaded model in $HOME/gensim-data
    model = gensim.downloader.load(args.modelname)

    logging.info("Creating tidy bias DataFrame...")
    bias_df = gensim2biasdf(model, words_a, words_b)

    assert sanity_check(model, bias_df, words_a, words_b), \
        "Our cosine is different from Gensim's :(((("

    logging.info("Saving DataFrame...")
    bias_df["bins"] = bias_df["bins"].astype(str)
    bias_df.to_csv(
        f"results/bias_{args.modelname}_{args.a}_{args.b}.csv", 
        index=False)


def sanity_check(model, df, words_a, words_b):
    """
    """
    target_word = "basketball"
    vec_a = [model[w] for w in words_a]
    vec_b = [model[w] for w in words_b]
    sims_a = model.cosine_similarities(model[target_word], vec_a)
    sims_b = model.cosine_similarities(model[target_word], vec_b)
    bias_gensim = np.mean(sims_a,) - np.mean(sims_b,)
    bias_ours = df.query("word == @target_word")["bias"].values[0]
    print(f"'{target_word}' bias (Gensim) = ", bias_gensim)
    print(f"'{target_word}' bias (Ours) = ", bias_ours)
    return np.isclose(bias_gensim, bias_ours, atol=1e-4)


if __name__ == "__main__":
    main()
