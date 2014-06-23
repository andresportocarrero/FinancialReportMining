"""
Executable file for LSA (Latent Semantic Analysis).

When running this file, 'working directory' need to be specified as Project Root (FinancialReportMining).
"""
import datetime
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from utils.util import unpickle, enpickle
import numpy as np

__author__ = 'kensk8er'


def convert_dict_list(dict_data):
    """
    Convert dictionary format data into list format data. Return both list format data and indices that show which list
    element corresponds to which dictionary element.

    :param dict_data: dict[dict]
    :return: list[str] list_data, list[str] indices
    """
    list_data = []
    indices = []

    for index, dict_datum in dict_data.items():
        assert dict_datum.has_key('text'), 'dictionary data need to have the field \'text\'!'
        assert isinstance(dict_datum['text'], str), 'str required!'
        list_data.append(unicode(dict_datum['text'], 'utf-8'))  # convert str into unicode
        indices.append(index)

    return list_data, indices

def calculate_similarities(document_matrix):
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


if __name__ == '__main__':
    print 'read documents...'
    documents = unpickle('data/txt/documents.pkl')
    doc_num = len(documents)

    # convert dictionary format into list format
    print 'convert dictionary into list format...'
    doc_lists, doc_indices = convert_dict_list(documents)

    # Perform an IDF normalization on the output of HashingVectorizer
    hasher = HashingVectorizer(stop_words='english', non_negative=True,
                               norm=None, binary=False)
    vectorizer = Pipeline((
        ('hasher', hasher),
        ('tf_idf', TfidfTransformer())  # TODO: you should try many different parameters here
    ))

    # reduce the number of documents for now
    doc_lists = doc_lists[:400]
    doc_indices = doc_indices[:400]

    # calculate TF-IDF
    print 'calculate TF-IDF...'
    X = vectorizer.fit_transform(doc_lists)

    # perform LSA
    print 'perform LSA...'
    lsa = TruncatedSVD(n_components=300, algorithm='arpack')
    X = np.matrix(lsa.fit_transform(X))

    # calculate cosine similarities between each text
    print 'calculate cosine similarities...'
    similarities = calculate_similarities(X)

    print 'save similarities and indices...'
    #date_time = datetime.datetime.today().strftime("%m%d%H%M%S")
    enpickle(similarities, 'result/similarities.pkl')
    enpickle(doc_indices, 'result/indices.pkl')

