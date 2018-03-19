# !/usr/local/bin/python3
# -*- coding: utf-8 -*-
from nltk import PorterStemmer
import numpy as np
import operator
import math
from utils import (get_file_indices, process_content,
                   get_contents, get_stopword_list, remove_stopwords)


CLASSES = 13


def generate_terms(content_list, stopword_list):
    cleared_content = ' '.join(content_list)

    tokens = cleared_content.split(' ')
    tokens = list(set(tokens))  # remove duplicates

    stopword_list.append('')
    stopword_list.extend(list(map(chr, range(97, 123))))  # alphebat letters
    updated_tokens = remove_stopwords(stopword_list, tokens)

    stemmed_tokens = [PorterStemmer().stem(token) for token in updated_tokens]
    stemmed_tokens = list(set(stemmed_tokens))  # remove duplicates
    terms = remove_stopwords(stopword_list, stemmed_tokens)

    return terms


def get_llr_value(pro_list, pt, p1, p2):
    if pt == 0.0:
        h1 = math.log(1 - pt) * (pro_list[1] + pro_list[3])
    elif pt == 1.0:
        h1 = math.log(pt) * (pro_list[0] + pro_list[2])
    else:
        h1 = math.log(pt) * (pro_list[0] + pro_list[2]) + math.log(1 - pt) * (pro_list[1] + pro_list[3])  # noqa

    if p1 == 0.0:
        h2_1 = math.log(1 - p1) * pro_list[1]
    elif p1 == 1.0:
        h2_1 = math.log(p1) * pro_list[0]
    else:
        h2_1 = math.log(p1) * pro_list[0] + math.log(1 - p1) * pro_list[1]

    if p2 == 0.0:
        h2_2 = math.log(1 - p2) * pro_list[3]
    elif p2 == 1.0:
        h2_2 = math.log(p2) * pro_list[2]
    else:
        h2_2 = math.log(p2) * pro_list[2] + math.log(1 - p2) * pro_list[3]
    h2 = h2_1 + h2_2

    llr = -2 * (h1 - h2)

    return llr


def feature_selection(contents_list, terms_list):
    total_LLR_list = []
    for t_index, terms in enumerate(terms_list):
        LLR_list = []
        for term in terms:  # for each term candidate of class i
            pro_list = np.zeros(4)
            for c_index, contents in enumerate(contents_list):
                if t_index == c_index:  # on-topic
                    for content in contents:
                        if term in content:
                            pro_list[0] += 1
                else:  # off-topic
                    for content in contents:
                        if term in content:
                            pro_list[2] += 1

            pro_list[1] = 15 - pro_list[0]
            pro_list[3] = 180 - pro_list[2]
            pro_list = np.float32(pro_list)

            # calculate LLR
            pt = (pro_list[0] + pro_list[2]) / pro_list.sum()
            p1 = pro_list[0] / (pro_list[0] + pro_list[1])
            p2 = pro_list[2] / (pro_list[2] + pro_list[3])

            LLR_value = get_llr_value(pro_list, pt, p1, p2)
            LLR_list.append([term, LLR_value])

        LLR_list.sort(key=operator.itemgetter(1), reverse=True)
        total_LLR_list.append(LLR_list[:40])

    return total_LLR_list


def generate_vocabs(vocab_list, vocabs_file):
    final_vocabs = []
    for vocabs in vocab_list:
        for vocab in vocabs:
            final_vocabs.append(vocab[0])

    final_vocabs = list(set(final_vocabs))[:500]
    with open(vocabs_file, 'w') as f:
        for vocab in final_vocabs:
            f.write(vocab + '\n')


file_indices = get_file_indices('training.txt')
stopword_list = get_stopword_list('stopwords.txt')

contents_list = []
terms_list = []
for i in range(CLASSES):
    content_by_class = get_contents(file_indices[i])
    contents_list.append(content_by_class)  # length is 13
    terms_by_class = generate_terms(content_by_class, stopword_list)
    terms_list.append(terms_by_class)  # length is 13

# further processing for contents
updated_contents_list = []
for contents in contents_list:
    updated_contents = process_content(stopword_list, contents)
    updated_contents_list.append(updated_contents)

total_LLR_list = feature_selection(updated_contents_list, terms_list)
generate_vocabs(total_LLR_list, 'vocabs.txt')
