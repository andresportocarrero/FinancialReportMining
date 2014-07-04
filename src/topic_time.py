"""
Generate the distribution of topics P(z) on each time (weekly, monthly, etc.)
"""
from collections import defaultdict
from utils.util import unpickle, enpickle

__author__ = 'kensk8er'


# interval setting
WEEK = 0
MONTH = 1


def sort_by_time(documents, doc_indices, time_interval):
    """
    Sort documents by certain time-interval, and return time-interval:doc_ids pairs.

    :param documents: documents (having time data)
    :param doc_indices: document indices
    :param time_interval: time-interval by which doc_ids are sort
    :return: time-sorted document indices
    """
    def convert_time(documents, doc_indices):
        doc_id2raw_time = {}
        for title, document in documents.items():
            doc_id = doc_indices.index(title)
            time = document['date']
            doc_id2raw_time[doc_id] = time
        return doc_id2raw_time

    doc_id2raw_time = convert_time(documents, doc_indices)
    start_date = sorted(doc_id2raw_time.items(), key=lambda x:x[1])[0][1]

    time2doc_ids = defaultdict(list)
    for doc_id, raw_time in doc_id2raw_time.items():
        if time_interval is WEEK:
            diff = (raw_time - start_date).days
            time = diff // 7
            time2doc_ids[time].append(doc_id)
    return time2doc_ids


def compute_topics_by_time(time2doc_ids, p_z_d):
    N = len(time2doc_ids)
    Z = p_z_d.shape[0]
    time2topics = [[0 for i in range(Z)] for j in range(N)]

    for time, doc_ids in time2doc_ids.items():  # TODO: improve this for loop (not element-wise)
        for z in range(Z):
            for doc_id in doc_ids:
                if p_z_d[z, doc_id] > 0:
                    time2topics[time][z] += p_z_d[z, doc_id]
    return time2topics


if __name__ == '__main__':
    interval = WEEK
    plsa = unpickle('result/plsa.pkl')
    documents = unpickle('data/txt/lemmatized_noun_documents.pkl')
    time2doc_ids = sort_by_time(documents, plsa['doc_indices'], interval)
    time2topics = compute_topics_by_time(time2doc_ids, plsa['p_z_d'])
    enpickle(time2topics, 'result/week2topics.pkl')
