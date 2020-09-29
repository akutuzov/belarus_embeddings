#! /bin/env python3
# coding: utf-8

import gensim
import logging
import sys
from os import path
from helpers import load_model

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

# Loading model, word similarity dataset, analogy dataset
modelfile, simfile, analogfile = sys.argv[1:]

print("Model\tSimLex965\tAnalogy\tAverage")

name = modelfile.split(".")[0]

fasttext = False
if path.basename(modelfile).startswith("cc.be") and path.basename(modelfile).endswith(".bin"):
    fasttext = True
model = load_model(modelfile, fasttext=fasttext)
print(f"Model vocabulary: {len(model.wv.vocab)}")

similarities = model.wv.evaluate_word_pairs(
    simfile, dummy4unknown=True, restrict_vocab=300000
)
analogies = model.wv.evaluate_word_analogies(
    analogfile, restrict_vocab=30000, dummy4unknown=True
)

spearman = similarities[1][0]
analogies_accuracy = analogies[0]

print(
    f"{name}\t{spearman}\t{analogies_accuracy}\t{(spearman + analogies_accuracy) / 2}"
)
