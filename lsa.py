import numpy as np
import os
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
# from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from six import string_types
from gensim import matutils

from base import BaseWordVectorizer
from corpus import LineCorpus
from exceptions import *

class LsaWordVectorizer(BaseWordVectorizer):
    def __init__(self, vector_dim, count_normalization = None):
        
#         super(LsaWordVectorizer, self).__init__()

        self.vector_dim = vector_dim
#         self.min_df = min_df
        
        self.count_normalization = count_normalization
        if count_normalization is None:
            self.vectorizer = CountVectorizer(token_pattern=r'(?u)\b\S+\b',)
        elif count_normalization == 'tfidf':
            self.vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\S+\b', #標點都會被拿掉..
                                        sublinear_tf=True, use_idf=True)
        else:
            raise ValueError("not a valid normalization method: {}".format(count_normalization))
    
    def get_name(self):
        return 'LSA'

    def get_mid(self):
        count_norm = self.count_normalization or 'no_count_norm'
        # corpus_name = ''
        # if hasattr(self, 'corpus_path'): 

        mid = '{}_d{}_{}'.format(self.get_name(), self.vector_dim, count_norm)
        return mid
        
    def fit_word_vectors(self, corpus_path):
        docs = LineCorpus(corpus_path)
        dtm = self.vectorizer.fit_transform(docs)

        self.vocabulary = self.vectorizer.vocabulary_
        self.ind2word = [None] * len(self.vocabulary)
        for k, v in self.vocabulary.items():
            self.ind2word[v] = k

        tdm = dtm.T
        tdm = tdm.asfptype()
        svd = TruncatedSVD(self.vector_dim, algorithm = 'arpack')
        tdm_svd = svd.fit_transform(tdm) # vocab_len * vector_dim
        # tdm_svd = Normalizer(copy=False).fit_transform(tdm_svd) 
        
        self.svd = svd #components_ : vector_dim* doc_len
        self.word_vectors = tdm_svd
        self.init_sims(replace=True)

    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.
        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!
        """
        if getattr(self, 'word_vectors_norm', None) is None or replace:
            print("precomputing L2-norms of word weight vectors")
            if replace:
                self.word_vectors = normalize(self.word_vectors, norm='l2', axis=1, copy=False)
                self.word_vectors_norm =  self.word_vectors 
            else:
                self.word_vectors_norm = normalize(self.word_vectors, norm='l2', axis=1, copy=True)
    

    def get_similarity(self, term1, term2):

        v1 = self.get_word_vector(term1).reshape(1, -1)
        v2 = self.get_word_vector(term2).reshape(1, -1)

        sim = cosine_similarity(v1, v2)[0][0]

        return sim

    def most_similar(self, positive=None, negative=None, topn=10, restrict_vocab=None, indexer=None):
        """
        https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/keyedvectors.py

        Parameters
        ----------
        positive : :obj: `list` of :obj: `str`
            List of words that contribute positively.
        negative : :obj: `list` of :obj: `str`
            List of words that contribute negatively.
        topn : int
            Number of top-N similar words to return.
        restrict_vocab : int
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_vocab=10000 would
            only check the first 10000 word vectors in the vocabulary order. (This may be
            meaningful if you've sorted the vocabulary by descending frequency.)
        Returns
        -------
        :obj: `list` of :obj: `tuple`
            Returns a list of tuples (word, similarity)
        Examples
        --------
        >>> trained_model.most_similar(positive=['woman', 'king'], negative=['man'])
        [('queen', 0.50882536), ...]
        """
        if positive is None:
            positive = []
        if negative is None:
            negative = []

        self.init_sims()

        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [
            (word, 1.0) if isinstance(word, string_types + (np.ndarray,)) else word
            for word in positive
        ]
        negative = [
            (word, -1.0) if isinstance(word, string_types + (np.ndarray,)) else word
            for word in negative
        ]

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if isinstance(word, np.ndarray):
                mean.append(weight * word)
            else:
                mean.append(weight * self.get_word_vector(word, use_norm=True))
                ind = self.vocabulary.get(word)
                if  ind:
                    all_words.add(ind)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)

        if indexer is not None:
            return indexer.most_similar(mean, topn)

        # limited = self.word_vectors_norm if restrict_vocab is None else self.word_vectors_norm[:restrict_vocab]
        sims = np.dot(self.word_vectors_norm, mean) #cosine of two unit vectors = dot
        if not topn:
            return sims
        best = matutils.argsort(sims, topn=topn + len(all_words), reverse=True)
        # ignore (don't return) words from the input
        result = [(self.ind2word[ind], float(sims[ind])) for ind in best if ind not in all_words]
        return result[:topn]

    def get_word_vector(self, term, use_norm=False):
        if not hasattr(self, 'vocabulary') or not hasattr(self, 'word_vectors'):
            raise NotFittedError('Raw documents needed be fed first to estimate word vectors before\
             acquiring specific word vector. Call fit_word_vectors(corpus_path)')

        ind = self.vocabulary.get(term)
        if ind is None:
            raise KeyError('term {} is not in the vocabulary'.format(term))

        if use_norm:
            word_vec = self.word_vectors_norm[ind, :]
        else:
            word_vec = self.word_vectors[ind, :]

        return word_vec

    def __getitem__(self, key):

        word_vec = self.get_word_vector(key)
        return word_vec