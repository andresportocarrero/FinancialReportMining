"""
Executable file for LDA (Latent Dirichlet Allocation).

When running this file, 'working directory' need to be specified as Project Root (FinancialReportMining).
"""
import logging
from gensim import corpora
from gensim.models import LdaModel, TfidfModel
from module.text.stopword import extended_stopwords
import numpy as np

__author__ = 'kensk8er'


if __name__ == '__main__':
    # expand stopwords list
    stop_words = extended_stopwords

    # parameters
    max_doc = 9000
    num_topics = 100

    # logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    # load data
    dictionary = corpora.Dictionary.load('data/dictionary/noun_dictionary.dict')
    corpus = dictionary.corpus

    # compute TFIDF
    logging.info('compute TFIDF...')
    tfidf = TfidfModel(dictionary=dictionary, id2word=dictionary)
    idfs = [0 for i in xrange(len(tfidf.idfs))]
    for index, idf in tfidf.idfs.items():
        idfs[index] = idf
    idfs = np.array(idfs)

    # perform LDA
    logging.info('perform LDA...')
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10, iterations=50, alpha='auto')
    lda.save('result/model.lda')

    # weight LDA model with TF-IDF
    logging.info('weight LDA with TFIDF...')
    lda.state.sstats = np.multiply(lda.state.sstats, idfs.T)

    # print LDA model
    lda.print_topics(topics=num_topics, topn=10)
    lda.idf = idfs
    lda.save('result/model_tfidf.lda')