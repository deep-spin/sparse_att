import glob
import os
import pickle
import pandas as pd

# plot heatmap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib

import torch
from analyze import get_dfa_probs as calculate_dfa_probs
from ngram import (
    predict_with_n_gram_back_off,
    prob_distance,
    prob_distance_dfa,
    prob_distance_dfa_ngram,
)

from batched_baum_welch import predict_with_baumwelch


class Vocab:
    def __init__(self, vocab: list):
        self.vocab = vocab
        # inverse vocab
        self.inv_vocab = {v: k for k, v in enumerate(vocab)}

    def get_vocab(self, id):
        return self.vocab[id]

    def get_id(self, char):
        return self.inv_vocab[char]

    def __len__(self):
        return len(self.vocab)


def get_ngram_probs(results, ngram=3, uniform=False, backoff=False, addone=False):
    vocab = Vocab(results[0]["vocab"])
    n_gram_probs = []
    # make multiprocessing


    for b in range(len(results)):
        input = results[b]["input"]
        target = [vocab.get_id(t) for t in results[b]["target"]]
        probs = predict_with_n_gram_back_off(
            input,
            N=ngram,
            global_vocab=vocab,
            uniform=uniform,
            backoff=backoff,
            addone=addone,
        )
        n_gram_probs.append(probs)
    return n_gram_probs


def get_baumwelch_probs(results):
    vocab = Vocab(results[0]["vocab"])
    baumwelch_probs = []
    for b in range(len(results)):
        input = results[b]["input"]
        probs = predict_with_baumwelch(input, vocab, max_states=12)
        baumwelch_probs.append(probs)
    return baumwelch_probs

# from concurrent.futures import ThreadPoolExecutor

# def get_baumwelch_probs(results):
#     vocab = Vocab(results[0]["vocab"])

#     def process_result(result):
#         input = result["input"]
#         return predict_with_baumwelch(input, vocab, max_states=12)

#     with ThreadPoolExecutor() as executor:
#         baumwelch_probs = list(executor.map(process_result, results))

#     return baumwelch_probs


def get_dfa_probs(results):
    vocab = Vocab(results[0]["vocab"])
    dfa_probs = []
    for b in range(len(results)):
        input = results[b]["input"]
        target = [vocab.get_id(t) for t in results[b]["target"]]
        probs = calculate_dfa_probs(input, results[b]["dfa"], vocab=vocab)
        dfa_probs.append(probs)
    return dfa_probs


def get_model_probs(results, softmax=True):
    model_probs = []
    for b in range(len(results)):
        if softmax:
            probs = (
                torch.softmax(torch.tensor(results[b]["probs"]), dim=-1)
                .detach()
                .cpu()
                .numpy()
            )
        else:
            probs = results[b]["probs"]
        model_probs.append(probs)
    return model_probs


import numpy as np


def get_greedy_dfa_accuracy(probs, dfa_probs, offset=0, max_len=None):
    total = 0.0
    correct = 0.0
    for p1, pdfa in zip(probs, dfa_probs):
        if max_len is not None:
            pdfa = pdfa[offset:max_len]
        indices = p1.argmax(axis=-1)[: len(pdfa)]
        correct += (pdfa[np.arange(len(pdfa)), indices] > 0).sum()
        total += len(pdfa)
    return correct / total


EPS = 1e-7


def get_kl(probs, dfa_probs, offset=0, max_len=None):
    total = 0.0
    cross_entropy = 0.0
    for p1, pdfa in zip(probs, dfa_probs):
        # calculate the soft cross-entropy between p1 and pdfa
        if max_len is not None:
            pdfa = pdfa[offset:max_len]
        log_p1 = np.log(p1[: len(pdfa)] + EPS)
        log_pdfa = np.log(pdfa + EPS)
        cross_entropy += -((log_p1 - log_pdfa) * pdfa).sum()
        total += len(pdfa)
    return cross_entropy / total


def get_l1_loss(probs1, probs2, probsdfa, offset=0, max_len=None):
    total = 0.0
    correct = 0.0
    for p1, p2, pdfa in zip(probs1, probs2, probsdfa):
        if max_len is not None:
            pdfa = pdfa[offset:max_len]
        total += len(pdfa)
        correct += np.abs(
            p1[offset : offset + len(pdfa)] - p2[offset : offset + len(pdfa)]
        ).sum()
    return correct / total

# glob all checkpoints
run_folders = glob.glob(
    "experiments/hiddens_*/**/**/generations/*test_batch/", recursive=True
)
# create a map
name_to_folder = {}
for folder in run_folders:
    folder = folder.replace("//", "/").strip("/")
    subpaths = folder.split("/")
    name = subpaths[2]
    num_examples = subpaths[1].split("_")[1]
    nlayer = "" if subpaths[3] == "generations" else subpaths[3]
    name = f"{num_examples}/{name}/{nlayer}"
    name_to_folder[name] = folder
from probe import get_results
results = {}
for num_examples in (1000, 2500, 5000, 10000, 20000, 40000):
    results[num_examples] = {}
    for model in (
        "mm_relumax/128_8_2_0.01",
    ):
        name = f"{num_examples}/{model}"
        if model in results[num_examples]:
            continue
        results[num_examples][model] = get_results(
        name_to_folder[name], probs_only=True)

for num_examples in (1000, 2500, 5000, 10000,20000, 40000):
    results[num_examples] = {}
    for model in (
        "mm_relumax/128_8_2_0.01",

    ):
        name = f"{num_examples}/{model}"
        if model in results[num_examples]:
            continue
        results[num_examples][model] = get_results(
            name_to_folder[name], probs_only=True
        )

# compute DFA probs for each model individually (safe copy of keys)
for num_examples in (1000, 2500, 5000, 10000, 20000, 40000):
    model_keys = list(results[num_examples].keys())  # snapshot
    for model in model_keys:
        results[num_examples][f"{model}_dfa"] = get_dfa_probs(results[num_examples][model])

dfa_metrics = []
for metric in ("l1", "kl", "acc"):
    for num_examples in (1000, 2500, 5000, 10000, 20000, 40000):
        for model in results[num_examples].keys():
            if model.endswith("_dfa"):
                continue

            if model not in ("3gram", "2gram"):
                model_probs = get_model_probs(results[num_examples][model])
            else:
                model_probs = get_model_probs(results[num_examples][model], softmax=False)

            dfa_model_key = f"{model}_dfa"
            if dfa_model_key not in results[num_examples]:
                continue
            dfa_probs = results[num_examples][dfa_model_key]

            try:
                if metric == "l1":
                    value = get_l1_loss(model_probs, dfa_probs, dfa_probs) / 2
                elif metric == "kl":
                    value = get_kl(model_probs, dfa_probs)
                elif metric == "acc":
                    value = get_greedy_dfa_accuracy(model_probs, dfa_probs)

                dfa_metrics.append(
                    {
                        "model": model,
                        "metric": metric,
                        "value": value,
                        "num_examples": num_examples,
                    }
                )
            except Exception as e:
                print(e)
                print(model, num_examples)


# deep json to dataframe
df = pd.DataFrame(dfa_metrics)
df.to_csv("figures/dfa_metrics.csv")

df = pd.read_csv("figures/dfa_metrics.csv", header=None,
                 names=["idx", "model", "metric", "value", "num_examples"])


df["num_examples"] = pd.to_numeric(df["num_examples"].astype(str).str.strip(), errors="coerce")
df["value"] = pd.to_numeric(df["value"].astype(str).str.strip(), errors="coerce")
df = df.dropna()
pivot = df.pivot_table(
    index="num_examples",
    columns=["metric", "model"],
    values="value"
)

pivot.to_csv("figures/dfa_metrics.csv")
# make neurips conference quality plots
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import scienceplots



plt.rcParams['xtick.top'] = False
plt.rcParams['ytick.right'] = False
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['xtick.minor.visible'] = False

color_palette = """1AA89E
302DBE
F17011
D3226F
6A6AF8
1461F0
5E00C9
E2BC09
BD4805
107F4B
B0E926""".split(
    "\n"
)
# convert to rgb
color_palette = [matplotlib.colors.to_rgb("#"+c) for c in color_palette]

metric_names = {
    "kl": "KL",
    "l1": "TVD",
    "acc": "Accuracy",
}

model_names = {
    "transformer": "Transformer",
    "transformer_2": "Transformer (2 layers)",
    "transformer_1": "Transformer (1 layers)",
    "lstm": "LSTM",
    "hyena": "Hyena",
    "h3": "H3",
    "s4d": "S4D",
    "linear_transformer": "Linear Transformer",
    "rwkv": "RWKV",
    "retention": "RetNet",
    "mamba": "Mamba",
    "mm": "Memory Mosaic",
}

for metric in ("l1", "acc"):
    data = df[
        (df.metric == metric)
        & ~(
            df.model.isin(
                [
                    "transformer_8",
                    "transformer_4",
                    "transformer_2",
                    "transformer_1",
                    "3gram",
                    "2gram",
                ]
            )
        )
    ]
    data = data.replace({"model": model_names})
    # new figure
    fig, ax = plt.subplots()
    ax = sns.lineplot(
        data=data,
        x="num_examples",
        y="value",
        hue="model",
        style="model",
        markers=True,
        dashes=False,
        palette=color_palette,
        legend="full",
    )
    ax.set_xlabel("\# Training Examples")
    ax.set_ylabel(metric_names[metric])
    ax.set(xscale="log")
    ax.set_xticks([1000, 1000, 5000, 10000, 20000, 40000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # remove minor
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())

    # show 2gram and 3gram as horizontal line with texts on them
    ax.axhline(
        y=df[(df.metric == metric) & (df.model == "3gram")].value.max(),
        color="black",
        linestyle="--",
        linewidth=0.5,
    )
    ax.text(
        1000,
        df[(df.metric == metric) & (df.model == "3gram")].value.max() + 0.01,
        "3-gram",
        color="black",
#         fontsize=12,
    )
    ax.axhline(
        y=df[(df.metric == metric) & (df.model == "2gram")].value.max(),
        color="gray",
        linestyle="--",
        linewidth=0.5,
    )
    ax.text(
        1000,
        df[(df.metric == metric) & (df.model == "2gram")].value.max() + 0.01,
        "2-gram",
        color="gray",
#        fontsize=12,
    )
    plt.rcParams["text.usetex"] = False
    # save to pdf
    fig.savefig(f"figures/dfa_{metric}.pdf", bbox_inches="tight")