"""
Generate the distribution of topics P(z) on each time (weekly, monthly, etc.)
"""
from collections import defaultdict
import logging
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from utils.util import unpickle, enpickle

__author__ = 'kensk8er'


# interval setting
WEEK = 0
MONTH = 1


def sort_by_time(dictionary, time_interval):
    """
    Sort documents by certain time-interval, and return time-interval:doc_ids pairs.

    :param dictionary: dictionary (having time data)
    :param time_interval: time-interval by which doc_ids are sort
    :return: time-sorted document indices
    """
    start_date = sorted(dictionary.docid2date)[0]

    time2docids = defaultdict(list)
    for doc_id, raw_time in enumerate(dictionary.docid2date):
        # TODO: implement MONTH
        if time_interval is WEEK:
            diff = (raw_time - start_date).days
            time = diff // 7
            time2docids[time].append(doc_id)
    return time2docids


def compute_topics_by_time(time2doc_ids, model, dictionary):
    N = len(time2doc_ids)
    logging.info('Performing inference on corpora...')
    p_z_d = model.inference(dictionary.corpus)[0].T
    p_z_d = p_z_d / p_z_d.sum(axis=0).reshape(1, p_z_d.shape[1])  # normalize to make it probability
    Z = p_z_d.shape[0]
    time2topics = [[0 for i in range(Z)] for j in range(N)]

    for time, doc_ids in time2doc_ids.items():  # FIXME: improve this for loop (not element-wise)
        for z in range(Z):
            for doc_id in doc_ids:
                if p_z_d[z, doc_id] > 0:
                    time2topics[time][z] += p_z_d[z, doc_id]
    return time2topics


if __name__ == '__main__':
    # logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    interval = WEEK
    logging.info('Loading model...')
    model = LdaModel.load(fname='result/model.lda')
    logging.info('Loading dictinary...')
    dictionary = Dictionary.load('data/dictionary/report_(NN).dict')
    logging.info('Sort documents by time...')
    time2docids = sort_by_time(dictionary, interval)
    logging.info('Compute topic distribution for each time...')
    time2topics = compute_topics_by_time(time2docids, model, dictionary)
    enpickle(time2topics, 'result/week2topics.pkl')
