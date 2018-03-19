from nltk import PorterStemmer
import string
import re, os, math
import numpy as np


DOC_NUM = 1095


def generate_dictionary(term_list, doc_list):
    """
    Generate df and tf value for each term and each doc.
    dictionary = [
        {
            df: df-value,
            tf: {
                doc_id: tf-value, ...
            }
        }, ...
    ]
    """

    # initialize
    dictionary = [{'df': 0, 'tf': {}} for term in term_list]
    doc_by_terms = [[] for doc in doc_list]

    # generate df and tf value
    for term_idx, term in enumerate(term_list):
        for doc_idx, doc in enumerate(doc_list):
            if term in doc:
                dictionary[term_idx]['df'] += 1
                doc_by_terms[doc_idx].append(term_idx)
                tf_value = doc.count(term)
                dictionary[term_idx]['tf'][str(doc_idx)] = tf_value

    # write dictionary.txt
    dict_file = open('data/dictionary.txt', 'w', encoding='utf-8')
    for idx, dict_term in enumerate(dictionary):
        dict_file.write('{id:<10}{term:<30}{df:<10}\n'.format(
                id=str(idx + 1), term=term_list[idx], df=str(dict_term['df'])))
    dict_file.close()

    return dictionary, doc_by_terms


def get_doc_as_vector(dict, docs, idf_list):
    """ Turn documents into vectors """
    doc_vectors = []
    dict_length = len(dict)

    for idx, doc in enumerate(docs):
        doc_length = len(doc)

        # get tf values
        tf_list = [dict[term_idx]['tf'][str(idx)] for term_idx in doc]

        # calculate tf-idf values
        tf_idf_list = []
        term_idx = 0
        for i in range(dict_length):
            if i == doc[term_idx]:
                tf_idf = float(tf_list[term_idx] * idf_list[doc[term_idx]])
                tf_idf_list.append(tf_idf)
                if term_idx <= doc_length - 2:
                    term_idx += 1
            else:
                tf_idf_list.append(float(0))

        # normalize tf-idf value
        doc_length = math.sqrt(sum([value**2 for value in tf_idf_list]))
        ntf_idf = np.array(np.array(tf_idf_list) / doc_length)
        doc_vectors.append(ntf_idf)

    return doc_vectors


def process_docs(content_list, stop_words):
    total_terms = []
    doc_terms_list = []  # docs are turned into terms

    for content in content_list:
        content = re.sub(r'http\S+', " ", content)
        content = re.sub(r'www\S+', " ", content)
        content = re.sub("[" + string.punctuation + "]", " ", content)
        content = re.sub("[^a-zA-Z]+", " ", content)
        content = content.lower()
        tokens = content.split(' ')

        cleaned_tokens = []
        for token in tokens:
            if token not in stop_words:
                cleaned_tokens.append(token)

        stemmed_tokens = [PorterStemmer().stem(token) for token in cleaned_tokens]

        terms = []
        for token in stemmed_tokens:
            if token not in stop_words:
                terms.append(token)

        doc_terms_list.append(terms)
        total_terms.extend(list(set(terms)))

    # generate dictionary and docs (by terms)
    final_terms = sorted(list(set(total_terms)))

    dictionary, doc_by_terms = generate_dictionary(final_terms, doc_terms_list)

    # get idf list
    idf_list = [math.log10(DOC_NUM / term['df']) for term in dictionary]

    # get normalized tf_idf_value
    final_doc_list = get_doc_as_vector(dictionary, doc_by_terms, idf_list)

    return final_doc_list


def get_stop_words():
    with open('data/stopwords.txt', 'r') as f:
        stop_words = [line.rstrip() for line in f]
        stop_words.append('')
        stop_words.extend(list(map(chr, range(97, 123))))  # alphabet letters

        return stop_words


def get_doc_vectors(directory):
    doc_list = []
    for filename in sorted(
            os.listdir(directory),
            key=lambda filename: int(filename.replace('.txt', ''))):
        file = open(directory + filename, 'r', encoding='utf-8')
        doc_list.append(file.read())
        file.close()

    stop_word_list = get_stop_words()

    print(">>>>> get doc vectors ...")
    doc_vector_list = process_docs(doc_list, stop_word_list)

    return doc_vector_list
