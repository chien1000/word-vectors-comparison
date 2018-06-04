#!/usr/bin/python3
import re
import pickle
import six
from six import string_types

from collections import Mapping, defaultdict, Counter
import numbers
import numpy as np
import scipy.sparse as sp
import array
from operator import itemgetter
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import pairwise_distances
from gensim import matutils

# from stop_words import ENGLISH_STOP_WORDS
from base import BaseWordVectorizer
from corpus import LineCorpus
from exceptions import *

def _make_int_array():
    """Construct an array.array of a type suitable for scipy.sparse indices."""
    return array.array(str("i"))

class HalWordVectorizer(BaseWordVectorizer):
    """
    Parameters
    ----------
    window_size : context window size
    max_features : int or None, default=None
        If not None, conserving k context words(columns) with the 
        highest variance.
    vocabulary : Mapping or iterable, optional
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents. Indices
        in the mapping should not be repeated and should not have any gap
        between 0 and the largest index.
    dtype : type, optional
        Type of the matrix returned by fit_transform() or transform().
    """

    def __init__(self, window_size = 10, max_features=None, min_count=None,
                 dtype=np.int64):
        
        # super(HalWordVectorizer, self).__init__()

        self.window_size = window_size
        self.max_features = max_features
        if max_features is not None:
            if (not isinstance(max_features, numbers.Integral) or
                    max_features <= 0):
                raise ValueError(
                    "max_features=%r, neither a positive integer nor None"
                    % max_features)

        self.min_count = min_count or 0
        if min_count is not None:
            if not isinstance(min_count, numbers.Integral):
                raise ValueError(
                    "min_count=%r, neither a integer nor None"
                    % min_count)

        self.dtype = dtype
        self.use_sp_matrix = True
        
    def get_dim(self):
        return self.max_features
        
    def get_name(self):
        return 'HAL'

    def get_mid(self):
        mid =  '{}_d{}_window_{}'.format(self.get_name(), self.max_features, self.window_size)
        return mid

    def _count_cooccurence(self, docs):
        """Create sparse feature matrix, and vocabulary
        """
        vocabulary = self.vocabulary

        row = _make_int_array()
        col = _make_int_array()
        values = _make_int_array()

        window_size = self.window_size
        for doc in docs:
            doc = doc.split()
            doc_length = len(doc)

            for i, feature in enumerate(doc):
                try:
                    #左右window要分開計算！！！！！
                    feature_idx = vocabulary[feature]
                    for j in range(max(i - window_size, 0), i) :
                        context_word = doc[j]
                        context_idx = vocabulary[context_word]
                        row.append(feature_idx)
                        col.append(context_idx)
                        diff = i-j-1
                        values.append(window_size-diff)

                    for j in range(i+1, min(i + window_size, doc_length-1)+1):
                        context_word = doc[j]
                        context_idx = vocabulary[context_word] + len(vocabulary) 
                        row.append(feature_idx)
                        col.append(context_idx)
                        diff = j-i-1
                        values.append(window_size-diff)

                except KeyError:
                    # Ignore out-of-vocabulary items
                    continue

        values = np.frombuffer(values, dtype=np.intc)
        # print(len(vocabulary))
        # print(row.max())
        cooccurence_matrix = sp.coo_matrix((values, (row, col)), shape=(len(vocabulary), 
                                                                         len(vocabulary)*2)
                                           ,dtype=self.dtype)
        cooccurence_matrix = cooccurence_matrix.tocsc()
        # cooccurence_matrix.sort_indices()

#         print(cooccurence_matrix.toarray())
        return cooccurence_matrix

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
    
    def fit_word_vectors(self, corpus_path):

        docs = LineCorpus(corpus_path)

        # filter rare words according to self.min_count
        word_counter = Counter()
        for doc in docs:
            word_counter.update(doc.split())

        vocabulary = {}    
        freq_count = 0
        for w, c in word_counter.items():
            if c >= self.min_count:
                vocabulary[w] = freq_count
                freq_count+=1
        self.vocabulary = vocabulary
        self.ind2word = [None] * len(self.vocabulary)
        for k, v in self.vocabulary.items():
            self.ind2word[v] = k
        print('vocabulary size: {}'.format(len(vocabulary)))

        #count cooccurence
        cooccurence_matrix = self._count_cooccurence(docs)

        if self.max_features: #conserve top k cols with highest variance
            # compute variance 
            # E[X^2] - (E[X])^2 or np.var?
            squared = cooccurence_matrix.copy() 
            squared.data = np.power(squared.data, 2)
            mean_of_squared = squared.mean(0)
            squared_of_mean = np.power(cooccurence_matrix.mean(0), 2)
            variance = (mean_of_squared - squared_of_mean).A
            variance = np.squeeze(variance, axis = 0)
            del squared

            # conserve top k cols
            k = self.max_features
            topk_ind = np.sort(np.argsort(-variance)[:k])
            cooccurence_matrix = cooccurence_matrix[:, topk_ind]

            # reserved features
            vlen = len(self.ind2word)
            reserved_features = [(self.ind2word[i],'l') for i in topk_ind if i < vlen]
            reserved_features.extend([(self.ind2word[i-vlen],'r') for i in topk_ind if i >= vlen])
            self.reserved_features = reserved_features

        #normalize
        # cooccurence_matrix = normalize(cooccurence_matrix, norm='l2', axis=1, copy=True)

        self.word_vectors = cooccurence_matrix.tocsr()
        self.init_sims(replace=True)

        return self

    def get_similarity(self, term1, term2):


        v1 = self.get_word_vector(term1, use_norm=True)
        v2 = self.get_word_vector(term2, use_norm=True)

        distance = pairwise_distances(np.vstack((v1, v2)), metric='euclidean')[0,1]

        sim = 1 / (distance + 1)
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
        dists = pairwise_distances(X = self.word_vectors_norm, Y= [mean], metric='euclidean').squeeze() #distance!!

        if not topn:
            return 1/(dists+1)

        best = matutils.argsort(dists, topn=topn + len(all_words), reverse=False) #dist : small to large
        # ignore (don't return) words from the input
        result = [(self.ind2word[ind], 1/ (1+float(dists[ind]))) for ind in best if ind not in all_words]
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

        if sp.issparse(word_vec):
            word_vec = word_vec.toarray()
        word_vec = word_vec.squeeze()

        return word_vec

    def __getitem__(self, key):

        word_vec = self.get_word_vector(key)
        return word_vec
