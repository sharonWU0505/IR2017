"""Functions for processing content (documents)."""

# !/usr/local/bin/python3
# -*- coding: utf-8 -*-
from nltk.corpus import stopwords
from nltk import PorterStemmer
import string
import re


def clear_content(content_list):
    """
    Clear content.
    1. remove strings begin with "http" or "www" (not sure)
    2. remove punctuation (for cases with hyphen)
    3. remove non-alphabet
    """
    cleared_content_list = []
    for content in content_list:
        content = re.sub(r'http\S+', " ", content)
        content = re.sub(r'www\S+', " ", content)
        content = re.sub("[" + string.punctuation + "]", " ", content)
        content = re.sub("[^a-zA-Z]+", " ", content)
        cleared_content_list.append(content.lower())

    return cleared_content_list


def stem(tokens):
    """Do stemming by Porter's Algorithm."""
    stemmed_tokens = [PorterStemmer().stem(token) for token in tokens]

    return stemmed_tokens


def _remove_stopwords(terms):
    stopword_list = stopwords.words("english")
    stopword_list.append('')
    stopword_list.extend(list(map(chr, range(97, 123))))  # alphebat letters

    updated_terms = []
    for term in terms:
        if term not in stopword_list:
            updated_terms.append(term)

    return updated_terms


def generate_term_list(cleared_content):
    """Generate term list."""
    tokens = cleared_content.split(" ")
    tokens = list(set(tokens))  # remove duplicates
    stemmed_tokens = stem(tokens)
    stemmed_tokens = list(set(stemmed_tokens))  # remove duplicates
    term_list = _remove_stopwords(stemmed_tokens)

    return sorted(term_list)
