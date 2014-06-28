from sklearn.feature_extraction.text import CountVectorizer
from module.plsa.plsa import pLSA
import utils.taskmanager.taskmanager as tm
from utils.util import unpickle, enpickle

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
    test_num = 1000

    tm.TaskManager.OUTPUT_FOLDER = "./tmp"
    documents = unpickle('data/txt/documents.pkl')
    doc_num = len(documents)

    # convert dictionary format into list format
    print 'convert dictionary into list format...'
    doc_lists, doc_indices = convert_dict_list(documents)

    # decrease the number of documents
    doc_lists = doc_lists[:test_num]
    doc_indices = doc_indices[:test_num]

    vectorizer = CountVectorizer(stop_words='english', dtype='float64')
    #analyze = vectorizer.build_analyzer()
    X = vectorizer.fit_transform(doc_lists)
    feature_names = vectorizer.get_feature_names()
    print 'perform PLSA...'
    model = train(data=X.transpose(), Z=30, maxiter=3000)
    p_z, p_w_z, p_d_z = model

    print 'save results into .pkl file...'
    enpickle(p_z, 'result/plsa/p_z.pkl')
    enpickle(p_w_z, 'result/plsa/p_w_z.pkl')
    enpickle(p_d_z, 'result/plsa/p_d_z.pkl')
    enpickle(doc_lists, 'result/plsa/doc_indices.pkl')
    enpickle(doc_lists, 'result/plsa/doc_indices.pkl')
    enpickle(feature_names, 'result/plsa/feature_names.pkl')
