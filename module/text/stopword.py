"""
Self defined domain-specific stop-word list.
"""
from sklearn.feature_extraction import stop_words

__author__ = 'kensk8er'

extra_stopwords = set(
    {'hsbc', 'view', 'click', 'mailto', 'nomura', 'message', 'april', 'march', 'may', 'june', 'jun', 'july', 'august'})
extended_stopwords = stop_words.ENGLISH_STOP_WORDS.union(extra_stopwords)
