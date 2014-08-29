"""
Executable file for PLSA (Probabilistic Latent Semantic Analysis).

When running this file, 'working directory' need to be specified as Project Root (FinancialReportMining).
"""
import logging
from gensim import corpora
from gensim.models import LdaModel
from utils.util import unpickle

__author__ = 'kensk8er'

if __name__ == '__main__':
    # parameters
    num_topics = 100
    use_wiki = False
    passes = 1
    iterations = 50
    chunksize = 200

    # logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    # load data
    report_dict = corpora.Dictionary.load('data/dictionary/report_(NN).dict')
    report_corpus = report_dict.corpus

    if use_wiki is True:
        # wiki_dict = corpora.Dictionary.load('data/dictionary/wiki_(NN).dict')
        # wiki_corpus = wiki_dict.corpus
        #
        # logging.info('combine report and wiki dictionary...')
        # wiki_to_report = report_dict.merge_with(wiki_dict)
        # merged_dict = report_dict
        #
        # logging.info('combine report and wiki corpus...')
        # merged_corpus = wiki_to_report[wiki_corpus].corpus + report_corpus
        logging.info('generate wiki corpus...')
        wiki_txt = unpickle('data/txt/processed_wiki.pkl')
        wiki_corpus = [report_dict.doc2bow(wiki) for wiki in wiki_txt]

        logging.info('combine report and wiki corpus...')
        merged_corpus = wiki_corpus + report_corpus

    # compute TFIDF
    # logging.info('compute TFIDF...')
    # tfidf = TfidfModel(dictionary=report_dict, id2word=report_dict)

    # perform PLSA
    logging.info('perform PLSA...')
    if use_wiki is True:
        lda = LdaModel(corpus=merged_corpus, id2word=report_dict, num_topics=num_topics, passes=passes,
                       iterations=iterations, chunksize=chunksize, alpha=1., eta=1.)
        lda.save('result/model_wiki.plsa')
        lda.print_topics(topics=num_topics, topn=10)
    else:
        lda = LdaModel(corpus=report_corpus, id2word=report_dict, num_topics=num_topics, passes=passes,
                       iterations=iterations, chunksize=chunksize, alpha=1., eta=1.)
        lda.save('result/model.plsa')
        lda.print_topics(topics=num_topics, topn=10)
