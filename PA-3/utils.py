# !/usr/local/bin/python3
# -*- coding: utf-8 -*-
from nltk import PorterStemmer
import re
import string


DOCUMENT_DIR = "documents/"
DATA_DIR = "data/"


def get_file_indices(training_file):
    file_indices = []
    with open(DATA_DIR + training_file, 'r') as f:
        for line in f:
            line = line.replace(' \n', '').split(' ')
            file_indices.append(line[1:16])

    return file_indices


def _clear_content(content):
    content = re.sub(r'http\S+', ' ', content)
    content = re.sub(r'www\S+', ' ', content)
    content = re.sub("[" + string.punctuation + "]", ' ', content)
    content = re.sub("[^a-zA-Z]+", ' ', content)

    return content.lower()


def process_content(stopword_list, contents):
    updated_contents = []
    for content in contents:
        tokens = content.split(' ')
        updated_tokens = remove_stopwords(stopword_list, tokens)
        stemmed_tokens = [PorterStemmer().stem(token) for token in updated_tokens]
        updated_content = remove_stopwords(stopword_list, stemmed_tokens)
        updated_contents.append((' ').join(updated_content))

    return updated_contents


def get_contents(file_indices):
    contents = []
    for file_index in file_indices:
        file = open(DOCUMENT_DIR + str(file_index) + '.txt', 'r', encoding='utf-8')
        content = file.read()
        file.close()
        contents.append(_clear_content(content))

    return contents


def get_stopword_list(stopwords_file):
    stopword_list = []
    with open(DATA_DIR + stopwords_file, 'r') as f:
        stopword_list = [line.rstrip() for line in f]

    return stopword_list


def remove_stopwords(stopword_list, words):
    updated_words = []
    for word in words:
        if word not in stopword_list:
            updated_words.append(word)

    return updated_words
