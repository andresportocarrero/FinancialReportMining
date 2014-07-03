"""
Compute the similarities between each 'from' of a report over time.

Using P(d|z) as document features.
"""
from collections import defaultdict
from utils.util import unpickle, enpickle
import numpy as np

__author__ = 'kensk8er'


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


def create_from_vectors(doc_vectors, from2doc_ids):
    """
    Generate from vectors out of document vectors (P(z|d), by combining all the document vectors which belong to the
    same from.

    :param doc_vectors: P(z|d)
    :param from2doc_ids: relationship between froms and document ids
    :return: from_vectors
    """
    # from_vectors = {'p_z_d': [], 'id2from': {}}
    Z = doc_vectors.shape[0]
    from_vectors = {}
    for from_name, doc_ids in from2doc_ids.items():
        from_vector = np.array([0. for i in range(Z)])
        for doc_id in doc_ids:
            if sum(doc_vectors[:, doc_id]) > 0:
                from_vector += doc_vectors[:, doc_id]
        from_vectors[from_name] = from_vector
    return from_vectors


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

    return similarity_matrix


def convert_from_vectors(from_vectors):
    from_indices = []
    N = len(from_vectors)
    Z = len(from_vectors.items()[0][1])
    from_matrix = np.zeros([N, Z])
    for from_name, vector in from_vectors.items():
        from_matrix[len(from_indices), :] = vector
        from_indices.append(from_name)
    return np.matrix(from_matrix), from_indices


if __name__ == '__main__':
    plsa = unpickle('result/plsa.pkl')
    documents = unpickle('data/txt/lemmatized_noun_documents.pkl')

    from2titles = classify_documents_by_from(documents)
    from2doc_ids = convert_from2titles_from2doc_ids(from2titles, plsa['doc_indices'])
    from_vectors = create_from_vectors(plsa['p_z_d'], from2doc_ids)
    from_matrix, from_indices = convert_from_vectors(from_vectors)
    similarities = compute_similarity(from_matrix)
    from_similarity = {'similarity': similarities, 'id2from': from_indices}

    enpickle(from_similarity, 'result/from_similarity.pkl')
