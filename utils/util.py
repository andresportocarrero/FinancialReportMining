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


def write_topic_csv(plsa, csv_name, k):
    plsa = unpickle(plsa)
    k = int(k)
    p_w_z = plsa['p_w_z']
    word_indices = plsa['word_indices']
    Z = p_w_z.shape[1]
    writer = csv.writer(file(csv_name, 'w'))

    output = [[0 for i in range(Z)] for j in range(k)]
    for z in range(Z):
        rank = 0
        for index in p_w_z[:,z].argsort()[::-1]:
            output[rank][z] = {}
            output[rank][z]['word'] = word_indices[index]
            output[rank][z]['p'] = p_w_z[index,z]
            rank += 1
            if rank >= k:
                break

    # 1st row
    row = ['']
    for z in range(Z):
        row.append('z = ' + str(z))
    writer.writerow(row)

    # 2nd row and onwards
    for rank in range(k):
        row = [str(rank+1)]

        for z in range(Z):
            row.append(output[rank][z]['word'] + ':' + str(output[rank][z]['p']))

        writer.writerow(row)


def write_from_similarity(from_similarity, csv_name):
    from_similarity = unpickle(from_similarity)
    similarity = from_similarity['similarity']
    id2from = from_similarity['id2from']
    from_num = len(id2from)
    writer = csv.writer(file(csv_name, 'w'))

    # 1st row
    row = ['']
    for from_name in id2from:
        row.append(from_name)
    writer.writerow(row)

    # 2nd row and onwards
    for row_num in range(from_num):
        row = [id2from[row_num]]

        for col_num in range(from_num):
            row.append(similarity[row_num, col_num])

        writer.writerow(row)


def write_time_topics(time2topics, csv_name):
    time2topics = unpickle(time2topics)
    Z = len(time2topics[0])
    writer = csv.writer(file(csv_name, 'w'))

    # 1st row
    row = ['']
    for time in range(len(time2topics)):
        row.append('time = ' + str(time))
    writer.writerow(row)

    # 2nd row and onwards
    for topic in range(Z):
        row = ['z = ' + str(topic)]

        for time in range(len(time2topics)):
            row.append(time2topics[time][topic])

        writer.writerow(row)

if __name__ == '__main__':
    args = sys.argv
    assert len(args) > 1, 'at least one argument needed!'

    if args[1] == 'write_similarity_csv':
        assert len(args) == 5, '4 arguments needed for function write_similarity_csv!'
        write_similarity_csv(pickle_indices=args[2], pickle_data=args[3], csv_name=args[4])
    elif args[1] == 'write_topic_csv':
        assert len(args) == 5, '4 arguments needed for function write_similarity_csv!'
        write_topic_csv(plsa=args[2], csv_name=args[3], k=args[4])
    elif args[1] == 'write_from_similarity':
        assert len(args) == 4, '3 arguments needed for function write_similarity_csv!'
        write_from_similarity(from_similarity=args[2], csv_name=args[3])
    elif args[1] == 'write_time_topics':
        assert len(args) == 4, '3 arguments needed for function write_similarity_csv!'
        write_time_topics(time2topics=args[2], csv_name=args[3])
    else:
        print 'no function executed'
