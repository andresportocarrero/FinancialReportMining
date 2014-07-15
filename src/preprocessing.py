"""
Preprocessing text (takes approximately 30 minutes for 9,000 financial reports)
"""
import logging
from gensim import corpora
from nltk import RegexpTokenizer
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.utils import lemmatize, revdict
import re
from module.text.compoundword import compound_words
from module.text.stopword import extended_stopwords
from utils.util import unpickle

__author__ = 'kensk8er'


def convert_compound(document):
    doc_string = " ".join(document)
    for compound_word in compound_words:
        doc_string = re.sub(compound_word, re.sub('\s', '-', compound_word), doc_string)

    return doc_string.split()


def clean_text(text):
    # TODO: a bit like black magic... simplify this.
    text = re.sub("(http(s)?://[A-Za-z0-9\'~+\-=_.,/%\?!;:@#\*&\(\)]+)", '', text)  # URL
    text = re.sub("(www\.[A-Za-z0-9\'~+\-=_.,/%\?!;:@#\*&\(\)]+)", '', text)  # URL
    text = re.sub("([A-Za-z0-9\'~+\-=_.,/%\?!;:@#\*&\(\)]+\.\w+)", '', text)  # URL
    text = re.sub("(\w+@[a-zA-Z_]+?\.[a-zA-Z]{2,6})", '', text)  # email
    text = re.sub("(-{2,})", '', text)  # hyphen
    text = re.sub("(\w+)(-)(\w+)", r'\1 \3', text)  # hyphen
    text = re.sub("(mailto:\w+)", r'\1 \3', text)  # hyphen
    text = re.sub("=\r\n", '', text)  # next-line
    text = re.sub("([A-Za-z0-9\'~+\-=_.,/%\?!;:@#\*&\(\)]{15,})", '', text)  # long characters
    text = re.sub("(\b)(\w)(\b)", r'\1 \3', text)  # short (single) characters
    text = re.sub("[0-9]", ' ', text)  # number
    text = re.sub("\w*([~+\-=_/%@#\*&\?!]+\w+)", '', text)  # special code
    return text


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

        # filter stop words
        document = [w for w in document if not w in stop_words and 2 <= len(w) <= 15]

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

    dictionary.save('data/dictionary/dictionary_' + allowed_pos.pattern + '.dict')
