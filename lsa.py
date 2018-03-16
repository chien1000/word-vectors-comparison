import numpy as np
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from stop_words import ENGLISH_STOP_WORDS # stopword list defined by sklearn
from base import BaseWordVectorizer
from exceptions import *

class LsaWordVectorizer(BaseWordVectorizer):
    def __init__(self, vector_dim, count_normalization = None, min_df=1, stop_words = 'english'):
        
        super(LsaWordVectorizer, self).__init__()

        self.vector_dim = vector_dim
        self.min_df = min_df
        self.stop_words = stop_words
        if count_normalization is None:
            self.vectorizer = CountVectorizer(min_df = self.min_df, stop_words = self.stop_words)
        elif count_normalization == 'tfidf':
            self.vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', 
                                        min_df = self.min_df, stop_words = self.stop_words, 
                                        sublinear_tf=True, use_idf=True)
        else:
            raise ValueError("not a valid normalization method: {}".format(count_normalization))

    def fit_word_vectors(self, raw_documents):
        dtm = self.vectorizer.fit_transform(raw_documents)
        self.vocabulary_ = self.vectorizer.vocabulary_

        dtm = dtm.asfptype()
        svd = TruncatedSVD(self.vector_dim, algorithm = 'arpack')
        svd = svd.fit(dtm)
        self.svd = svd #components_ : n_components* n_features

        # dtm_svd = svd.fit_transform(dtm) # doc_len * vector_dim
        # dtm_svd = Normalizer(copy=False).fit_transform(dtm_svd) 

    def get_similarity(self, term1, term2):

        ind1 = self.vocabulary_.get(term1)
        ind2 = self.vocabulary_.get(term2)
        if not ind1:
            raise KeyError('term {} is not in the vocabulary'.format(term1))
        if not ind2:
            raise KeyError('term {} is not in the vocabulary'.format(term2))

        v1 = self.svd.components_[:, ind1].reshape(1, -1)
        v2 = self.svd.components_[:, ind2].reshape(1, -1)
        sim = cosine_similarity(v1, v2)[0][0]

        return sim

    def __getitem__(self, key):

        if not hasattr(self, 'svd'):
            raise NotFittedError('Raw documented needed be fed first to estimate word vectors before\
             acquiring specific word vector. Call fit_word_vectors(raw_documents)')

        ind = self.vocabulary_.get(key)
        if not ind:
            raise KeyError('term {} is not in the vocabulary'.format(key))

        word_vec =  self.svd.components_[:, ind]
        return word_vec

#normalization???