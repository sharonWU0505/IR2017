# !/usr/local/bin/python3
# -*- coding: utf-8 -*-
import math
import os
from process_content import (
    clear_content, stem, generate_term_list)


DOCUMENT_DIRECTORY = "documents/"
DICTIONARY_FILEPATH = "dictionary.txt"
RESULT_DIRECTORY = "result/"


def prepare_content(directory):
    """Read all documents and do pre processing."""
    content = []
    for filename in sorted(
            os.listdir(directory),
            key=lambda filename: int(filename.replace(".txt", ""))):
        file = open(directory + filename, "r", encoding="utf-8")
        content.append(file.read())
        file.close()

    cleared_content = clear_content(content)
    return cleared_content


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
    dictionary = [
        {
            "df": 0,
            "tf": {}
        }
        for term in term_list
    ]
    terms_for_doc = [[] for doc in doc_list]

    # generate df and tf value
    for term_index, term in enumerate(term_list):
        for doc_index, doc in enumerate(doc_list):
            if term in doc:
                dictionary[term_index]["df"] += 1
                terms_for_doc[doc_index].append(term_index)
                tf_value = doc.count(term)
                dictionary[term_index]["tf"][str(doc_index)] = tf_value

    return dictionary, terms_for_doc


def get_tf_related_values_and_write_file(dictionary,
                                         terms_for_doc,
                                         idf_list,
                                         doc_index_list):
    for doc_index in doc_index_list:
        # get tf values
        term_index_list = terms_for_doc[doc_index]
        tf_list = [
            dictionary[term_index]["tf"][str(doc_index)]
            for term_index in term_index_list
        ]

        # process tf-idf values
        tf_idf_list = []
        for term_index, tf in enumerate(tf_list):
            tf_idf = float(tf * idf_list[term_index_list[term_index]])
            tf_idf_list.append(tf_idf)

        # process normalized tf-idf value and write file
        output_filepath = "Doc{id}.txt".format(id=(doc_index + 1))
        doc_file = open(RESULT_DIRECTORY + output_filepath, "w", encoding="utf-8")
        doc_file.write("{term_count}\n".format(
            term_count=str(len(term_index_list))))

        vector_length = math.sqrt(sum([value**2 for value in tf_idf_list]))
        for index, tf_idf in enumerate(tf_idf_list):
            doc_file.write(
                '{id:<10}{ntf_idf:<30}\n'.format(
                    id=str(term_index_list[index]),
                    ntf_idf=str(tf_idf / vector_length))
            )
        doc_file.close()


# Generate term list
cleared_content = prepare_content(DOCUMENT_DIRECTORY)
term_list = generate_term_list(" ".join(cleared_content))

# Process documents and generate dictionary
processed_content = []
for content in cleared_content:
    content = content.split(" ")
    processed_content.append(stem(content))
doc_count = len(processed_content)
dictionary, terms_for_doc = generate_dictionary(
    term_list, processed_content)

# Write dictionary.txt
dict_file = open(DICTIONARY_FILEPATH, "w", encoding="utf-8")
for term_index, term in enumerate(dictionary):
    dict_file.write(
        '{id:<10}{term:<30}{df:<10}\n'.format(
            id=str(term_index + 1),
            term=term_list[term_index],
            df=str(term["df"]))
    )
dict_file.close()

# Get idf list
idf_list = [math.log10(doc_count / term["df"]) for term in dictionary]

# Check if the folder for storing results exists
if not os.path.exists('result'):
    os.makedirs('result')

# Get n_tf_idf_value and write file
get_tf_related_values_and_write_file(
    dictionary, terms_for_doc, idf_list, doc_index_list=[0])
