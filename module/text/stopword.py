"""
Self defined domain-specific stop-word list.
"""
from sklearn.feature_extraction import stop_words

__author__ = 'kensk8er'

extra_stopwords = {'hsbc', 'view', 'click', 'mailto', 'nomura', 'message', 'january', 'february', 'march', 'april',
                   'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'jan', 'feb', 'mar',
                   'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'barclays', 'survey', 'invitation', 'daily',
                   'weekly', 'monthly', 'highlight', 'isi', 'report', 'portfolio', 'strategy', 'jpy', 'analysis',
                   'technical', 'register', 'reminder', 'survey', 'citi', 'merrill', 'lynch', 'goldman', 'sachs',
                   'morgan', 'stanley', 'chase', 'Lloyd', 'attach', 'nomura' 'norge', 'bank', 'recommendation',
                   'insight', 'comment', 'morning', 'gs', 'week', 'day', 'month', 'year', 'minute', 'second', 'hour',
                   'two', 'news', 'data', 'market', 'ubs', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                   'saturday', 'sunday', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun', 'reid', 'research', 'summary',
                   'ft', 'comment', 'wsj', 'outlook', 'update', 'reading', 'note', 'hfe', 'snapshot', 'thought', 'edt',
                   'gmt', 'natixis', 'research', 'intraday', 'chosen', 'running', 'cent', 'dollar', 'ordinary', 'ad',
                   'subscription', 'access', 'internet', 'web', 'email', 'mail', 'app', 'pc', 'product', 'analyst',
                   'use', 'jpmorgan', 'near', 'time', 'unsubscribe', 'subscription', 'form', 'read', 'hundred',
                   'thousand', 'million', 'billion', 'trillion', 'forecast', 'expect', 'expectation', 'detail',
                   'number', 'curve', 'end', 'learn', 'member', 'offer', 'estimate', 'index', 'line', 'ensure',
                   'person', 'book', 'material', 'state', 'smart', 'appearance', 'category', 'percent', 'performance',
                   'bloomberg', 'record', 'program', 'status', 'group', 'forward', 'let', 'thank', 'dear',
                   'distribution', 'position', 'night', 'phone', 'chart', 'buy', 'sell', 'target', 'access', 'land',
                   'communication', 'reference', 'document', 'attachment', 'recipient', 'calendar' 'confirmation',
                   'backlog', 'word', 'overview', 'yesterday', 'today', 'tomorrow', 'last', 'next', 'previous', 'final',
                   'summarize', 'quieter', 'conundrum', 'package', 'whilst', 'source', 'client', 'distribution', 'core',
                   'headline', 'increase', 'right', 'act', 'download', 'matrix', 'refer', 'fall', 'monitor', 'opinion',
                   'theme', 'short', 'area', 'power', 'security', 'address', 'browser', 'paste', 'copy', 'close',
                   'stop', 'start', 'sender', 'representation', 'buy', 'sell'}

extended_stopwords = stop_words.ENGLISH_STOP_WORDS.union(extra_stopwords)
