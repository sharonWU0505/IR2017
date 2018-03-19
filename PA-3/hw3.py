# !/usr/local/bin/python3
# -*- coding: utf-8 -*-
import math
import numpy as np
from utils import (get_file_indices, process_content,
                   get_contents, get_stopword_list)


DOCUMENT_DIR = "documents/"
DATA_DIR = "data/"
CLASSES = 13


def get_vocab_list(vocab_file):
    vocab_list = []
    with open(vocab_file, 'r') as f:
        for line in f:
            vocab_list.append(line.replace('\n', ''))

    return vocab_list


def get_vocab_count(vocab_list, doc_list):
    """by class"""
    vocab_count_list = []
    for vocab in vocab_list:
        vocab_count = 1
        for doc in doc_list:
            vocab_count += doc.count(vocab)
        vocab_count_list.append(vocab_count)

    vocab_count_list = np.array(vocab_count_list)
    vocab_sum = vocab_count_list.sum() + 500
    vocab_count_list = vocab_count_list / vocab_sum
    return vocab_count_list


def train_multinomial_model(vocab_list, contents_list):
    vocab_prob_matrix = []
    for contents in contents_list:
        vocab_prob_matrix.append(get_vocab_count(vocab_list, contents))

    return vocab_prob_matrix


def test_multinomial_model(vocab_count_matrix, vocab_list, doc_list):
    print("=============== Start Testing ===============")
    class_result = []
    for doc in doc_list:
        vocab_count_list = []
        for vocab in vocab_list:
            vocab_count_list.append(doc.count(vocab))

        score_list = []
        for c_index in range(CLASSES):
            score = 0
            for v_index, vocab_count in enumerate(vocab_count_list):
                score += vocab_count * math.log(vocab_count_matrix[c_index][v_index])
            score_list.append(score)

        class_result.append(score_list.index(max(score_list)) + 1)

    return class_result


file_indices = get_file_indices('training.txt')
stopword_list = get_stopword_list('stopwords.txt')

contents_list = []
for i in range(CLASSES):
    content_by_class = get_contents(file_indices[i])
    contents_list.append(content_by_class)  # length is 13

# further processing for contents
updated_contents_list = []
for contents in contents_list:
    updated_contents = process_content(stopword_list, contents)
    updated_contents_list.append(updated_contents)

vocab_list = get_vocab_list('vocabs.txt')
vocab_prob_matrix = train_multinomial_model(
    vocab_list, updated_contents_list)

doc_list = get_contents(list(range(1, 1096)))
updated_doc_list = process_content(stopword_list, doc_list)
class_result = test_multinomial_model(vocab_prob_matrix, vocab_list, updated_doc_list)

indices = []
for index in file_indices:
    indices.extend(index)

with open('output.txt', 'w') as f:
    for index, c in enumerate(class_result, start=1):
        if str(index) not in indices:
            f.write(str(index) + ' ' + str(c) + '\n')
