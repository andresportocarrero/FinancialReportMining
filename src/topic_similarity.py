"""
Computing similarity between each category.
"""
import logging
from gensim.models import LdaModel
from sklearn.metrics.pairwise import cosine_similarity
from utils.util import enpickle

__author__ = 'kensk8er'


if __name__ == '__main__':
    # logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    logging.info('Loading the model...')
    model = LdaModel.load('result/model_wiki.lda')
    topics = []
    for topic_id in range(model.num_topics):
        topics.append(model.return_topic(topicid=topic_id))

    similarity = cosine_similarity(topics)
    enpickle(similarity, 'result/topic_similarity/lda_wiki.pkl')
