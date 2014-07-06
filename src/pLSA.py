"""
Executable file for PLSA (Probabilistic Latent Semantic Analysis).

When running this file, 'working directory' need to be specified as Project Root (FinancialReportMining).
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from module.plsa.gen_prob import gen_p_w, gen_p_wz, gen_p_z_w, gen_p_d, gen_p_dz, gen_p_z_d
from module.plsa.plsa import pLSA, normalize
from module.text.stopword import extended_stopwords
import utils.taskmanager.taskmanager as tm
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


@tm.task(str)
def feat(folder):
    pass


@tm.task(feat, int, int)
def train(data, Z, maxiter=500, debug=True):
    plsa = pLSA()
    plsa.debug = debug
    return plsa.train(td=data, Z=Z, maxiter=maxiter)


if __name__ == '__main__':
    # expand stopwords list
    stop_words = extended_stopwords

    # parameters
    test_num = 9000
    Z = 100
    maxiter = 3000

    tm.TaskManager.OUTPUT_FOLDER = "./tmp"
    documents = unpickle('data/txt/lemmatized_noun_documents.pkl')
    doc_num = len(documents)

    # convert dictionary format into list format
    print 'convert dictionary into list format...'
    doc_lists, doc_indices = convert_dict_list(documents)

    # decrease the number of documents
    if doc_num > test_num:
        doc_lists = doc_lists[:test_num]
        doc_indices = doc_indices[:test_num]

    token_pattern = u'(?u)[\\s\\t\\n!\\?\\^\\(-]([A-Za-z_]{2,15})[\\s\\t\\n!\\?\\$\\.\\)-]'
    vectorizer = CountVectorizer(stop_words=stop_words, dtype='float64', token_pattern=token_pattern,
                                 strip_accents='unicode', min_df=5, max_df=0.5)

    print 'fitting vectorizer...'
    X = vectorizer.fit_transform(doc_lists)
    word_indices = vectorizer.get_feature_names()

    print 'compute idf...'
    transformer = TfidfTransformer(norm=None)
    transformer.fit(X)
    idf = transformer.idf_

    print 'perform PLSA...'
    model = train(data=X.transpose(), Z=Z, maxiter=maxiter)
    p_z, p_w_z, p_d_z = model

    print 'compute P(w|z), P(w|z), P(w|z), and P(w|z), etc...'
    p_w = gen_p_w(p_w_z, p_z)
    p_wz = gen_p_wz(p_w_z, p_z)
    p_z_w = gen_p_z_w(p_wz, p_w)
    p_d = gen_p_d(p_d_z, p_z)
    p_dz = gen_p_dz(p_d_z, p_z)
    p_z_d = gen_p_z_d(p_dz, p_d)

    print 'computing idf-weighted P(w_z)...'
    p_w_z_idf = np.multiply(p_w_z, idf.reshape((-1, 1)))

    print 'save results into .pkl file...'
    plsa = {'p_z': p_z, 'p_w_z': p_w_z, 'p_d_z': p_d_z, 'p_w': p_w, 'p_wz': p_wz, 'p_z_w': p_z_w, 'idf': idf,
            'p_w_z_idf': p_w_z_idf, 'p_d': p_d, 'p_dz': p_dz, 'p_z_d': p_z_d, 'doc_indices': doc_indices,
            'word_indices': word_indices}
    enpickle(plsa, 'result/plsa.pkl')
