
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
from gensim.models.keyedvectors import KeyedVectors
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests


def similarities(
    M: np.ndarray, idx_target: list, idx_attr: list, use_norm=True
) -> np.ndarray:
    """use_norm: cosine distance
    """
    M_t = M[idx_target, :] # matriz de target words
    M_a = M[idx_attr, :] # matriz de attr words
    res = M_t @ M_a.T # rows: target words // cols: dot with each attr
    if use_norm:
        normas_t = np.linalg.norm(M_t, axis=1)
        normas_a = np.linalg.norm(M_a, axis=1)
        denominadores = np.outer(normas_t, normas_a)
        res = res / denominadores
    return res


def gensim2biasdf(model: KeyedVectors, words_a: list, words_b: list) -> pd.DataFrame:
    """
    """
    # DF with vocab and freq rank
    df = pd.DataFrame(model.key_to_index.items(), columns=['word', 'idx'])
    df["freq_rank"] = df["word"].apply(lambda x: model.get_vecattr(x, "count"))
    df.sort_values(by="freq_rank", ascending=False, inplace=True)
    df["reverse_freq_rank"] = df["freq_rank"].iloc[::-1].values
    max_value = np.log10(df["freq_rank"].max())
    cuts = [0] + list(np.arange(1.5, max_value + .5, .5))
    # cuts = [0, 1.5, 2, 3, 4, 5, 6]
    df["bins"] = pd.cut(np.log10(df["reverse_freq_rank"]), bins=cuts, right=False)
    assert df["bins"].isnull().sum() == 0
    # exclude attribute words
    idx_a = df.query("word in @words_a")["idx"].values
    idx_b = df.query("word in @words_b")["idx"].values
    idx_target = np.setdiff1d(df["idx"].values, np.union1d(idx_a, idx_b))
    df = df.query("idx in @idx_target").copy()
    # compute similarities with attributes
    similarities_a = similarities(model.vectors, idx_target, idx_a)
    similarities_b = similarities(model.vectors, idx_target, idx_b)
    # compute bias
    biases = np.mean(similarities_a, axis=1) - np.mean(similarities_b, axis=1)
    df["bias"] = biases    
    return df


def boxplots_plt(data, x_var, y_var, xlabel=None, ylabel=None, title=None):
    """
    """
    fig, ax = plt.subplots(figsize=(5,3))
    ax = sns.boxplot(
        x=data[x_var], y=data[y_var], showfliers=False, color="lightblue",
        showmeans=True, meanprops={"marker":"s", "markerfacecolor":"white", "markeredgecolor":"blue"})
    ax.axhline(0, ls='--', color='black', linewidth=1)
    ax.invert_xaxis()
    ax.tick_params(labelsize=9)
    plt.xticks(rotation=15)
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.title(title, fontsize=12, multialignment="left")
    return fig, ax


def scatter_plt(
    data, x_var, y_var, flag_color=None, colors=['#1f77b4','#ff7f0e'], 
    x_log=True, n_sample=None, smooth=False, frac=0.1, seed=123):
    """
    Scatter plot (x with log10)
    Param:
        - flag_color: name of binary variable to color points with
        - n_sample: size of sample or None
        - lowess: plot smooth curve or not
        - frac: lowess frac
    """
    if n_sample:
        data = data.sample(n_sample, random_state=seed)
    data_resto = data
    if flag_color:
        data_flag = data[data[flag_color] == 1]
        data_resto = data[data[flag_color] == 0]
    fig, ax = plt.subplots()
    if x_log:
        ax.set_xscale('log')
        #ax.set_xlim(left=10, right=10**8)
    plt.scatter(
        x_var, y_var, linewidth=0, c=colors[0], s=8, data=data_resto, label=0)
    if flag_color:
        plt.scatter(
            x_var, y_var, linewidth=0, c=colors[1], s=8, data=data_flag, label=1)
    if smooth:
        x_data = data[x_var]
        if x_log:
            x_data = np.log10(data[x_var])
        smooth_data = lowess(data[y_var], x_data, frac=frac)
        x_smooth = smooth_data[:,0]
        if x_log:
            x_smooth = 10**smooth_data[:,0]
        line = ax.plot(
            x_smooth, smooth_data[:,1], color='black', lw=1.0, ls='--')
    ax.axhline(0, ls='--', color='gray', linewidth=0.5)
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.legend()
    return fig, ax


def str2floats(s):
    """Convert strings separated by '|' to list
    """
    res = [float(i) for i in s.split("|")]
    return res
