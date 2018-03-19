"""Calculate cosine value of two docs."""

# !/usr/local/bin/python3
# -*- coding: utf-8 -*-

DOC_DIRECTORY = "result/"


def _read_doc(filename):
    data_list = []
    with open(DOC_DIRECTORY + filename) as file:
        next(file)  # skip the first line
        for line in file:
            data = line.strip().split()
            data_list.append(
                [int(data[0]), float(data[1])]
            )

        return data_list


def cosine(doc_index_x, doc_index_y):
    """Calculate cosine value of two docs."""
    doc_x_filename = "Doc{}.txt".format(str(doc_index_x + 1))
    doc_y_filename = "Doc{}.txt".format(str(doc_index_y + 1))

    doc_x = _read_doc(doc_x_filename)
    doc_y = _read_doc(doc_y_filename)

    index_x, index_y, cosine_value = 0, 0, 0
    while index_x < len(doc_x) and index_y < len(doc_y):
        if doc_x[index_x][0] == doc_y[index_y][0]:
            cosine_value += doc_x[index_x][1] * doc_y[index_y][1]
            index_x += 1
            index_y += 1
        elif doc_x[index_x][0] > doc_y[index_y][0]:
            index_y += 1
        else:
            index_x += 1

    return cosine_value
