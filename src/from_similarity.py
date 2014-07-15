"""
Compute the similarities between each 'from' of a report over time.

Using P(d|z) as document features.
"""
from collections import defaultdict
from utils.util import unpickle, enpickle
import numpy as np

__author__ = 'kensk8er'


# interval setting
WEEK = 0
MONTH = 1


def classify_documents_by_from(documents):
    relation = defaultdict(list)

    for index, document in documents.items():
        relation[document['from']].append(index)

    return relation


def convert_from2titles_from2doc_ids(from2titles, id2title):
    """

    :param from2titles:
    :param id2title:
    :return:
    """
    from2doc_ids = {}
    for from_name, titles in from2titles.items():
        from2doc_ids[from_name] = []
        for title in titles:
            from2doc_ids[from_name].append(id2title.index(title))
    return from2doc_ids


def create_from_vectors(doc_vectors, from2doc_ids, time2doc_ids, time):
    """
    Generate from vectors out of document vectors (P(z|d), by combining all the document vectors which belong to the
    same from.

    :param doc_vectors: P(z|d)
    :param from2doc_ids: relationship between froms and document ids
    :param time2doc_ids: mapping from time to doc_ids
    :param time: specific time in which to create from_vectors
    :return: from_vectors, from_frequencies
    """
    Z = doc_vectors.shape[0]
    from_vectors = {}
    from_frequencies = {}
    for from_name, doc_ids in from2doc_ids.items():
        from_vector = np.array([0. for i in range(Z)])

        # iterate only over the doc_ids which are in the certain time
        doc_ids = set(doc_ids).intersection(time2doc_ids[time])
        for doc_id in doc_ids:
            if sum(doc_vectors[:, doc_id]) > 0:
                from_vector += doc_vectors[:, doc_id]
        from_vectors[from_name] = from_vector
        from_frequencies[from_name] = len(doc_ids)

    return from_vectors, from_frequencies


def compute_similarity(document_matrix):
    """
    Calculate the similarities between every document vector which is contained in the document matrix given as an
    argument.

    :rtype : matrix[float]
    :param document_matrix: Document Matrix whose each row contains a document vector.
    """
    # calculate inner products
    print 'calculate inner products...'
    inner_product_matrix = np.dot(document_matrix, document_matrix.T)

    # calculate norms
    print 'calculate norms...'
    norms = np.sqrt(np.multiply(document_matrix, document_matrix).sum(1))
    norm_matrix = np.dot(norms, norms.T)

    # calculate similarities
    print 'calculate similarities...'
    similarity_matrix = inner_product_matrix / norm_matrix

    return np.nan_to_num(similarity_matrix).tolist()  # convert to list such that it can be dumpt into json


def convert_from_vectors(from_vectors):
    from_indices = []
    N = len(from_vectors)
    Z = len(from_vectors.items()[0][1])
    from_matrix = np.zeros([N, Z])
    for from_name, vector in from_vectors.items():
        from_matrix[len(from_indices), :] = vector
        from_indices.append(from_name)
    return np.matrix(from_matrix), from_indices


def convert_from_id(from_frequencies, from_indices):
    N = len(from_frequencies)
    id_frequencies = [0 for i in range(N)]
    for from_name, frequency in from_frequencies.items():
        id_frequencies[from_indices.index(from_name)] = frequency
    return id_frequencies


def sort_by_time(documents, doc_id2title, interval):
    """
    Sort documents by certain time-interval, and return doc_id:time-interval pairs.

    :param documents: documents (having time data)
    :param doc_id2title: document index
    :param time_interval: time-interval by which doc_id is sorted
    :return: doc_id:time-interval pairs
    """
    def convert_time(documents, doc_indices):
        doc_id2raw_time = {}
        for title, document in documents.items():
            doc_id = doc_indices.index(title)
            time = document['date']
            doc_id2raw_time[doc_id] = time
        return doc_id2raw_time

    doc_id2raw_time = convert_time(documents, doc_id2title)
    start_date = sorted(doc_id2raw_time.items(), key=lambda x:x[1])[0][1]

    time2doc_ids = defaultdict(list)
    for doc_id, raw_time in doc_id2raw_time.items():
        # TODO: implement MONTH
        if interval is WEEK:
            diff = (raw_time - start_date).days
            time = diff // 7
            time2doc_ids[time].append(doc_id)

    return time2doc_ids


if __name__ == '__main__':
    interval = WEEK  # only WEEK is implemented for now
    plsa = unpickle('result/plsa_100.pkl')
    documents = unpickle('data/txt/lemmatized_noun_documents.pkl')

    print('classify documents by from...')
    from2titles = classify_documents_by_from(documents)

    print('convert from2titles into from2doc_ids...')
    from2doc_ids = convert_from2titles_from2doc_ids(from2titles, plsa['doc_indices'])

    print('sort by time...')
    time2doc_ids = sort_by_time(documents, plsa['doc_indices'], interval)

    # iterate over every interval
    from_similarity = {}
    for time in range(max(time2doc_ids.keys())):
        print('\ncompute similarity for time = ' + str(time) + '...')
        from_vectors, from_frequencies = create_from_vectors(plsa['p_z_d'], from2doc_ids, time2doc_ids, time)
        from_matrix, from_indices = convert_from_vectors(from_vectors)
        similarities = compute_similarity(from_matrix)
        id_frequencies = convert_from_id(from_frequencies, from_indices)
        from_similarity[time] = {'similarity': similarities, 'id2from': from_indices, 'frequency': id_frequencies}

    enpickle(from_similarity, 'result/from_similarity.pkl')
