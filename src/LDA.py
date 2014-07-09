"""
Executable file for LDA (Latent Dirichlet Allocation).

When running this file, 'working directory' need to be specified as Project Root (FinancialReportMining).
"""
import logging
from gensim import corpora
from gensim.models import LdaModel, TfidfModel
from nltk import RegexpTokenizer
from module.text.stopword import extended_stopwords
from utils.util import unpickle
import numpy as np

__author__ = 'kensk8er'


if __name__ == '__main__':
    # expand stopwords list
    stop_words = extended_stopwords

    # parameters
    max_doc = 9000
    num_topics = 30

    # logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    # load documents
    logging.info('load documents...')
    documents = unpickle('data/txt/lemmatized_noun_documents.pkl')
    max_doc = min(max_doc, len(documents))

    # tokenizing
    tokenizer = RegexpTokenizer('([A-Za-z_]{2,15})')
    texts = []
    doc_id2title = []
    logging.info('tokenizing texts...')
    count = 1
    for index, document in documents.items():
        print '\r', count, '/', max_doc,
        texts.append(tokenizer.tokenize(document['text']))
        doc_id2title.append(index)
        count += 1
        if len(doc_id2title) >= max_doc:
            break
    print ''

    # filter stop words
    logging.info('filter stop words...')
    texts = [[w for w in text if not w in stop_words] for text in texts]

    # create corpus
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # compute TFIDF
    logging.info('compute TFIDF...')
    tfidf = TfidfModel(dictionary=dictionary, id2word=dictionary)
    idfs = [0 for i in xrange(len(tfidf.idfs))]
    for index, idf in tfidf.idfs.items():
        idfs[index] = idf
    idfs = np.array(idfs)

    # perform LDA
    logging.info('perform LDA...')
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10)
    lda.save('result/model.lda')

    # weight LDA model with TF-IDF
    logging.info('weight LDA with TFIDF...')
    lda.expElogbeta = np.multiply(lda.expElogbeta, idfs.T)

    # print LDA model
    lda.print_topics(topics=num_topics, topn=10)
    lda.save('result/model_tfidf.lda')
