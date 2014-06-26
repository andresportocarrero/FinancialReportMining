"""
util module

General utility functions are defined here.
"""
import csv
import sys

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
    print 'loading pickled data...'
    indices = unpickle(pickle_indices)
    data = unpickle(pickle_data)
    writer = csv.writer(file(csv_name, 'w'))
    count = 1
    index_num = len(indices)

    print 'writing csv file...'
    print '\r', count, '/', index_num,
    # 1st row
    row = ['']
    for index in indices:
        row.append(index)
    writer.writerow(row)
    count += 1

    # 2nd row and onwards
    for row_num in range(index_num):
        print '\r', count, '/', index_num,
        row = [indices[row_num]]

        for col_num in range(index_num):
            row.append(data[row_num, col_num])

        writer.writerow(row)
        count += 1
    print '\r', count, '/', index_num,


if __name__ == '__main__':
    args = sys.argv
    assert len(args) > 1, 'at least one argument needed!'

    if args[1] == 'write_similarity_csv':
        assert len(args) == 5, '4 arguments needed for function write_similarity_csv!'
        write_similarity_csv(pickle_indices=args[2], pickle_data=args[3], csv_name=args[4])

    else:
        print 'no function executed'
