# !/usr/local/bin/python3
# -*- coding: utf-8 -*-
from nltk.corpus import stopwords
from nltk import PorterStemmer
import re
import string


def _tokenize(content):
    # remove tabs and newlines
    content = re.sub("[\t\n\r\f\v]", "", content)
    # remove 's
    content = re.sub("'s", "", content)
    # remove '\ufeff'
    content = re.sub("\ufeff", "", content)
    # remove punctuations
    content = re.sub("[" + string.punctuation + "]", "", content)
    # split by space
    tokens = content.split(" ")
    return tokens


def _normalize(tokens):
    # lowercase
    tokens = [token.lower() for token in tokens]
    # remove duplicates
    tokens = list(set(tokens))
    # remove stop words
    terms = []
    for token in tokens:
        if token not in stopwords.words("english"):
            terms.append(token)
    return terms


def _stem(terms):
    # by Porter's Algorithm
    stemmed_terms = [PorterStemmer().stem(term) for term in terms]
    return stemmed_terms


def process_document(content):
    tokens = _tokenize(content)
    terms = _normalize(tokens)
    stemmed_terms = _stem(terms)
    return stemmed_terms


# read file
file = open("news.txt", "r", encoding="utf-8")
content = file.read()
file.close()

# process document
stemmed_terms = sorted(process_document(content))

# write file
output_file = open("result.txt", "w", encoding="utf-8")
for stemmed_term in stemmed_terms:
    output_file.write(stemmed_term + "\n")
output_file.close()
