"""
Compute the similarities between each 'from' of a report over time.

Using P(z|d) as document features.
"""
from collections import defaultdict
import logging
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from utils.util import unpickle, enpickle
import numpy as np

__author__ = 'kensk8er'


# interval setting
WEEK = 0
MONTH = 1


def create_from_vectors(p_z_d, from2doc_ids, time2doc_ids, time):
    """
    Generate from vectors out of document vectors (P(z|d), by combining all the document vectors which belong to the
    same from.

    :param p_z_d: P(z|d)
    :param from2doc_ids: relationship between froms and document ids
    :param time2doc_ids: mapping from time to doc_ids
    :param time: specific time in which to create from_vectors
    :return: from_vectors, from_frequencies
    """
    Z = p_z_d.shape[0]
    from_vectors = {}
    from_frequencies = {}
    for from_name, doc_ids in from2doc_ids.items():
        from_vector = np.array([0. for i in range(Z)])

        # iterate only over the doc_ids which are in the certain time
        doc_ids = set(doc_ids).intersection(time2doc_ids[time])
        for doc_id in doc_ids:
            if sum(p_z_d[:, doc_id]) > 0:
                from_vector += p_z_d[:, doc_id]
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


def sort_by_time(docid2date, interval):
    """
    Sort documents by certain time-interval, and return doc_id:time-interval pairs.

    :param docid2date: document_id - time mapping
    :param interval: time-interval by which doc_id is sorted
    :return: doc_id:time-interval pairs
    """
    start_date = sorted(docid2date)[0]

    time2docids = defaultdict(list)
    for doc_id, date in enumerate(docid2date):
        # TODO: implement MONTH
        if interval is WEEK:
            diff = (date - start_date).days
            time = diff // 7
            time2docids[time].append(doc_id)

    return time2docids


def convert_docid2from_from2docids(docid2from):
    from2docids = defaultdict(list)
    for docid, from_name in enumerate(docid2from):
        from2docids[from_name].append(docid)
    return from2docids


if __name__ == '__main__':
    # logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    interval = WEEK  # only WEEK is implemented for now

    model = LdaModel.load('result/model_wiki.lda')
    dictionary = Dictionary.load('data/dictionary/report_(NN).dict')

    from2docids = convert_docid2from_from2docids(dictionary.docid2from)
    time2docids = sort_by_time(dictionary.docid2date, interval)

    p_z_d = model.inference(dictionary.corpus)[0].T
    p_z_d = p_z_d / p_z_d.sum(axis=0).reshape(1, p_z_d.shape[1])  # normalize to make it probability

    # iterate over every interval
    from_similarity = {}
    for time in range(max(time2docids.keys())):
        print('\ncompute similarity for time = ' + str(time) + '...')
        from_vectors, from_frequencies = create_from_vectors(p_z_d, from2docids, time2docids, time)
        from_matrix, from_indices = convert_from_vectors(from_vectors)
        similarities = compute_similarity(from_matrix)
        id_frequencies = convert_from_id(from_frequencies, from_indices)
        from_similarity[time] = {'similarity': similarities, 'id2from': from_indices, 'frequency': id_frequencies,
                                 'topic': from_matrix}

    enpickle(from_similarity, 'result/from_similarity_wiki.pkl')
