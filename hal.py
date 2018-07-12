#!/usr/bin/python3
import pickle
from datetime import datetime
import os
import pickle

from collections import Mapping, defaultdict, Counter
import numbers
import numpy as np
import scipy.sparse as sp
import array
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import pairwise_distances

# from stop_words import ENGLISH_STOP_WORDS
from base import BaseWordVectorizer, get_vocabulary, MODEL_PATH
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

    def __init__(self, window_size = 10, max_features=None, min_count=0,
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

        self.min_count = min_count 
        if min_count is not None:
            if not isinstance(min_count, numbers.Integral):
                raise ValueError(
                    "min_count=%r, neither a integer nor None"
                    % min_count)

        self.dtype = dtype
        self.use_sp_matrix = True # for save_model 
        
    def get_dim(self):
        return self.max_features
        
    def get_name(self):
        return 'HAL'

    def get_mid(self):
        mid =  '{}_d{}_window_{}'.format(self.get_name(), self.max_features, self.window_size)
        return mid

    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.
        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!
        """
        if getattr(self, 'word_vectors_norm', None) is None or replace:
            print("precomputing L2-norms of word weight vectors")
            if replace:
                # normalize of sklearn can deal with sparse matrix
                self.word_vectors = normalize(self.word_vectors, norm='l2', axis=1, copy=False)
                self.word_vectors_norm =  self.word_vectors 
            else:
                self.word_vectors_norm = normalize(self.word_vectors, norm='l2', axis=1, copy=True)
    
    def _count_cooccurence(self, docs):
        """Create sparse feature matrix, and vocabulary
        """
        vocabulary = self.vocabulary

        row = _make_int_array()
        col = _make_int_array()
        values = _make_int_array()
        cooccurence_matrix = sp.csc_matrix((len(vocabulary), len(vocabulary)*2) ,dtype=self.dtype)

        window_size = self.window_size
        doc_id = 0
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

            
            batch_size = 10000
            if doc_id % batch_size ==0:
                values = np.frombuffer(values, dtype=np.intc)
                batch_matrix = sp.csc_matrix((values, (row, col)), shape=(len(vocabulary), 
                                                                        len(vocabulary)*2), dtype=self.dtype)
                cooccurence_matrix += batch_matrix
                # reset
                row = _make_int_array()
                col = _make_int_array()
                values = _make_int_array()

                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                print('processed #{} docs'.format(doc_id+1))

            doc_id +=1

        if len(values)  > 0: 
            values = np.frombuffer(values, dtype=np.intc)
            batch_matrix = sp.csc_matrix((values, (row, col)), shape=(len(vocabulary), 
                                                                    len(vocabulary)*2), dtype=self.dtype)
            cooccurence_matrix += batch_matrix

            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print('processed #{} docs'.format(doc_id+1))

        # cooccurence_matrix = cooccurence_matrix.tocsc()
        # cooccurence_matrix.sort_indices()

#         print(cooccurence_matrix.toarray())
        return cooccurence_matrix
    
    def fit_word_vectors(self, corpus_path):

        #count cooccurence
        corpus_name = os.path.splitext(os.path.basename(corpus_path))[0]
        save_com_path =  '{}_{}_mc{}_com.npz'.format(self.get_name(), corpus_name, self.min_count)
        save_com_path = os.path.join(MODEL_PATH, save_com_path)
        save_ind2word_path =  '{}_{}_mc{}_ind2word.bin'.format(self.get_name(), corpus_name, self.min_count)
        save_ind2word_path = os.path.join(MODEL_PATH, save_ind2word_path)

        try:
            cooccurence_matrix = sp.load_npz(save_com_path)
            with open(save_ind2word_path, 'rb') as fin:
                self.ind2word = pickle.load(fin)
                self.vocabulary = {w:i for i, w in enumerate(self.ind2word)}

                print('load existed cooccurence_matrix and vocab')
                print('vocabulary size: {}'.format(len(self.vocabulary)))
                
        except Exception as e:
            docs = LineCorpus(corpus_path)
            # filter rare words according to self.min_count
            self.ind2word, self.vocabulary = get_vocabulary(docs, self.min_count)
            print('vocabulary size: {}'.format(len(self.vocabulary)))

            cooccurence_matrix = self._count_cooccurence(docs)
            sp.save_npz(save_com_path, cooccurence_matrix)
            with open(save_ind2word_path, 'wb') as fout:
                pickle.dump(self.ind2word, fout)


        if self.max_features: #conserve top k cols with highest variance
            # compute variance 
            # E[X^2] - (E[X])^2 or np.var?
            squared_of_mean = np.square(cooccurence_matrix.mean(0))
            assert (squared_of_mean>=0).all()
            
            cooccurence_matrix.data = np.square(cooccurence_matrix.data)
            assert (cooccurence_matrix.data >= 0).all()
            mean_of_squared = cooccurence_matrix.mean(0)

            variance = (mean_of_squared - squared_of_mean).A
            variance = np.squeeze(variance, axis = 0)
            
            cooccurence_matrix.data = np.sqrt(cooccurence_matrix.data)

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
    
    def one2many_similarity(self, one_v, many_v, normalized=True):
        one_v = one_v.reshape(1, -1) # 1*dim
        many_v = many_v.reshape(-1, self.get_dim()) # n*dim

        if not normalized:
            l2_len = np.sqrt((many_v**2).sum(axis=1)).reshape(-1,1)
            many_v /= l2_len
            one_v /= np.sqrt((one_v ** 2).sum(-1))

        dists = pairwise_distances(X = many_v, Y= one_v, metric='euclidean').squeeze() #distance!!
        sim = 1 / (dists+1)
        return sim
