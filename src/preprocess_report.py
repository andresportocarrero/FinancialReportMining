"""
Preprocessing text (takes approximately 30 minutes for 9,000 financial reports)
"""
import logging
from gensim import corpora
from gensim.utils import lemmatize, revdict
import re
from module.text.preprocessing import convert_compound, clean_text
from module.text.stopword import extended_stopwords
from utils.util import unpickle

__author__ = 'kensk8er'


if __name__ == '__main__':
    # hyper-parameters
    allowed_pos = re.compile('(NN)')
    max_doc = float('inf')
    title_weight = 3

    # logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    # expand stopwords list
    stop_words = extended_stopwords

    logging.info('load documents...')
    documents = unpickle('data/txt/documents.pkl')

    logging.info('lemmatize...')
    count = 0
    doc_num = len(documents)
    new_documents = []
    titles = []
    froms = []
    dates = []
    for index, document in documents.items():
        count += 1
        if count > max_doc:
            break

        print '\r', count, '/', doc_num,
        text = document['text'] + (' ' + index) * title_weight  # incorporate title information
        from_name = document['from']
        date = document['date']

        cleaned = clean_text(text)  # delete irrelevant characters

        document = []
        tokens = lemmatize(content=cleaned, allowed_tags=allowed_pos)  # lemmatize
        for token in tokens:
            word, pos = token.split('/')
            document.append(word)

        # convert compound word into one token
        document = convert_compound(document)

        # filter stop words, long words, and non-english words
        document = [w for w in document if not w in stop_words and 2 <= len(w) <= 15 and w.islower()]

        new_documents.append(document)
        titles.append(index)
        froms.append(from_name)
        dates.append(date)

    print '\n'
    logging.info('create dictionary and corpus...')
    dictionary = corpora.Dictionary(new_documents)
    dictionary.docid2title = titles
    dictionary.docid2from = froms
    dictionary.docid2date = dates

    logging.info('filter unimportant words...')
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=None)
    dictionary.compactify()

    logging.info('generate corpus...')
    dictionary.corpus = [dictionary.doc2bow(document) for document in new_documents]
    dictionary.id2token = revdict(dictionary.token2id)

    dictionary.save('data/dictionary/report_' + allowed_pos.pattern + '.dict')
