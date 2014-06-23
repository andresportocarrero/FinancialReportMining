"""
util module

General utility functions are defined here.
"""
import csv

__author__ = 'kensk8er'


def enpickle(data, file):
    import cPickle

    fo = open(file, 'w')
    cPickle.dump(data, fo, protocol=2)
    fo.close()


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


def write_similarity_csv(pickle_indices, pickle_data, csv_name):
    indices = unpickle(pickle_indices)
    data = unpickle(pickle_data)
    writer = csv.writer(file(csv_name, 'w'))

    # 1st row
    row = ['']
    for index in indices:
        row.append(index)
    writer.writerow(row)

    # 2nd row and onwards
    index_num = len(indices)
    for row_num in range(index_num):
        row = [indices[row_num]]

        for col_num in range(index_num):
            row.append(data[row_num, col_num])

        writer.writerow(row)
