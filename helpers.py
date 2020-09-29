#! python3
# coding: utf-8

import numpy as np
import gensim
from collections import OrderedDict
from sklearn.decomposition import PCA


def load_model(embeddings_file, fasttext=False):
    if fasttext:
        emb_model = gensim.models.fasttext.load_facebook_vectors(embeddings_file)
        return emb_model
    # Определяем формат модели по её расширению:
    if embeddings_file.endswith(".bin.gz") or embeddings_file.endswith(
        ".bin"
    ):  # Бинарный формат word2vec
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=True, unicode_errors="replace"
        )
    elif (
        embeddings_file.endswith(".txt.gz")
        or embeddings_file.endswith(".txt")
        or embeddings_file.endswith(".vec.gz")
        or embeddings_file.endswith(".vec")
    ):  # Текстовый формат word2vec
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=False, unicode_errors="replace"
        )
    else:  # Нативный формат Gensim?
        emb_model = gensim.models.Word2Vec.load(embeddings_file)
    emb_model.init_sims(
        replace=True
    )  # На всякий случай приводим вектора к единичной длине (нормируем)
    return emb_model


def jaccard(list0, list1):
    # Близость двух массивов по коэффициенту Жаккара
    set_0 = set(list0)
    set_1 = set(list1)
    n = len(set_0.intersection(set_1))
    return n / (len(set_0) + len(set_1) - n)


def jaccard_f(word, models, row=10):
    # Сравнение слова в нескольких моделях через сравнение его ближайших ассоциатов
    associations = OrderedDict()
    similarities = {word: OrderedDict()}
    previous_state = None
    for m in models:
        model = models[m]
        word_neighbors = [i[0] for i in model.most_similar(positive=[word], topn=row)]
        associations[m] = word_neighbors
        if previous_state:
            similarity = jaccard(previous_state[1], word_neighbors)
            similarities[word][m] = similarity
        previous_state = (m, word_neighbors)
    return similarities, associations


def wordvectors(words, emb_model):
    # Функция получения векторов слов из модели
    matrix = np.zeros((len(words), emb_model.vector_size))
    for i in range(len(words)):
        matrix[i, :] = emb_model[words[i]]
    return matrix


def get_number(word, vocab=None):
    # Функция получения номера слова в словаре модели
    if word in vocab:
        return vocab[word].index
    else:
        return 0
