"""
Lemmatize text
"""
import string
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import re
from module.text.stopword import extended_stopwords
from utils.util import unpickle, enpickle

__author__ = 'kensk8er'


def get_tokens(text):
    lowers = text.lower()

    # remove the punctuation using the character deletion step of translate
    no_punctuation = lowers.translate(None, string.punctuation)
    tokens = word_tokenize(no_punctuation)
    return tokens


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        # return NOUN if others
        return wordnet.NOUN


if __name__ == '__main__':
    # expand stopwords list
    stop_words = extended_stopwords

    # avoid URL and email address
    condition = re.compile('(^(http)\w+|^(mailto)\w+|\w{10,}(com)$)')
    noun_only = False

    print 'loading documents...'
    documents = unpickle('data/txt/documents.pkl')
    lemmatizer = WordNetLemmatizer()

    print 'lemmatizing...'
    count = 0
    doc_num = len(documents)
    new_documents = {}
    for index, document in documents.items():
        count += 1
        print '\r', count, '/', doc_num,
        text = document['text']
        from_name = document['from']
        date = document['date']
        tokens = get_tokens(text)
        filtered = [w for w in tokens if not w in stop_words]
        tagged = pos_tag(filtered)

        document = ''
        for word in tagged:
            if not condition.match(word[0]):
                tag = get_wordnet_pos(word[1])

                # process NOUN only if noun_only is True
                if noun_only is False or tag == wordnet.NOUN:
                    document += lemmatizer.lemmatize(word[0], pos=tag) + ' '

        new_documents[index] = {}
        new_documents[index]['text'] = document
        new_documents[index]['from'] = from_name
        new_documents[index]['date'] = date

    print 'saving documents...'
    if noun_only is True:
        enpickle(new_documents, 'data/txt/lemmatized_noun_documents.pkl')
    else:
        enpickle(new_documents, 'data/txt/lemmatized_documents.pkl')