import numpy as np
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

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

        tdm = dtm.T
        tdm = tdm.asfptype()
        svd = TruncatedSVD(self.vector_dim, algorithm = 'arpack')
        tdm_svd = svd.fit_transform(tdm) # vocab_len * vector_dim
        # tdm_svd = Normalizer(copy=False).fit_transform(tdm_svd) 
        
        self.svd = svd #components_ : vector_dim* doc_len
        self.tdm_svd = tdm_svd

    def get_similarity(self, term1, term2):

        v1 = self.__getitem__(term1).reshape(1, -1)
        v2 = self.__getitem__(term2).reshape(1, -1)

        sim = cosine_similarity(v1, v2)[0][0]

        return sim

    def __getitem__(self, key):

        if not hasattr(self, 'svd'):
            raise NotFittedError('Raw documented needed be fed first to estimate word vectors before\
             acquiring specific word vector. Call fit_word_vectors(raw_documents)')

        ind = self.vocabulary_.get(key)
        if not ind:
            raise KeyError('term {} is not in the vocabulary'.format(key))

        word_vec =  self.tdm_svd[ind, :]
        return word_vec

#normalization???