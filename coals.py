from collections import defaultdict, Counter
import six
from six import string_types
from datetime import datetime

from hal import HalWordVectorizer, _make_int_array
from corpus import LineCorpus
from stop_words import ENGLISH_CLOSED_CLASS_WORDS
from exceptions import *

import numbers
import numpy as np 
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from sklearn.decomposition import TruncatedSVD
from gensim import matutils

class CoalsWordVectorizer(HalWordVectorizer):
    """docstring for CoalsWordVectorizer"""
    def __init__(self, window_size = 4, max_features=None, svd_dim=None,
                        min_count=None, dtype=np.int64):

        self.window_size = window_size
        self.max_features = max_features
        self.svd_dim = svd_dim
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

        self.stop_words = set(ENGLISH_CLOSED_CLASS_WORDS)
        # self.stop_words = None

        self.dtype = dtype

    def get_dim(self):
        if self.svd_dim is not None:
            return self.svd_dim 
        else:
            return  self.max_features

    def get_name(self):
        return 'COALS'

    def get_mid(self):
        mid = '{}_d{}_{}_window_{}'.format(self.get_name(), self.svd_dim, self.max_features, self.window_size)
        return mid

    def _count_cooccurence(self, docs):
        """Create sparse feature matrix
        """
        vocabulary = self.vocabulary
       
        row = _make_int_array()
        col = _make_int_array()
        values = _make_int_array()
        cooccurence_matrix = sp.csc_matrix((len(vocabulary), len(vocabulary)), dtype=self.dtype)

        window_size = self.window_size
        for doc_id, doc in enumerate(docs):
            doc = [t for t in doc.split()]
            doc_length = len(doc)

            for i, feature in enumerate(doc):
                try:
                    feature_idx = vocabulary[feature]
                    for j in range(max(i - window_size, 0), min(i + window_size, doc_length-1)+1):
                        if j == i:
                            continue
                        context_word = doc[j]
                        context_idx = vocabulary[context_word]
                        row.append(feature_idx)
                        col.append(context_idx)
                        diff = abs(j-i)-1
                        values.append(window_size-diff)

                except KeyError:
                    # Ignore out-of-vocabulary items    
                    continue

            batch_size = 10000
            if doc_id % batch_size == 0:
                values = np.frombuffer(values, dtype=np.intc)
                batch_matrix = sp.csc_matrix((values, (row, col)), shape=(len(vocabulary), 
                                                                                 len(vocabulary)), dtype=self.dtype)
                cooccurence_matrix += batch_matrix
                # reset
                row = _make_int_array()
                col = _make_int_array()
                values = _make_int_array()

                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                print('processed #{} docs'.format(doc_id+1))

        if len(values) > 0: 
            values = np.frombuffer(values, dtype=np.intc)
            batch_matrix = sp.csc_matrix((values, (row, col)), shape=(len(vocabulary), 
                                                                             len(vocabulary)), dtype=self.dtype)
            cooccurence_matrix += batch_matrix

            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print('processed #{} docs'.format(doc_id+1))
        # cooccurence_matrix = cooccurence_matrix.tocsc()
        
#         print(cooccurence_matrix.toarray())

        return cooccurence_matrix  

    def fit_word_vectors(self, corpus_path):

        # self._validate_vocabulary()
        docs = LineCorpus(corpus_path)

        # filter rare words according to self.min_count
        word_counter = Counter()
        for doc in docs:
            word_counter.update(doc.split())

        vocabulary = {}    
        freq_count = 0
        for w, c in word_counter.items():
            if c >= self.min_count and w not in self.stop_words:
                vocabulary[w] = freq_count
                freq_count+=1
        self.vocabulary = vocabulary
        self.ind2word = [None] * len(self.vocabulary)
        for k, v in self.vocabulary.items():
            self.ind2word[v] = k
        print('vocabulary size: {}'.format(len(vocabulary)))

        cooccurence_matrix = self._count_cooccurence(docs)
        
        if self.max_features: #discard all but the k columns reflecting the most common open-class words
            k = self.max_features
            topk_words = word_counter.most_common(k+len(self.stop_words)) #
            topk_ind = [self.vocabulary[w] for w, c in topk_words if w in self.vocabulary]
            topk_ind = topk_ind[:k]
            cooccurence_matrix = cooccurence_matrix[:, topk_ind]

            #reserved features
            self.reserved_features = [self.ind2word[i] for i in topk_ind]

        #normalize
        ##convert counts to word pair correlations
        t_sum = cooccurence_matrix.sum()
        row_sum = cooccurence_matrix.sum(axis = 1)
        col_sum = cooccurence_matrix.sum(axis = 0)

        cooccurence_matrix = cooccurence_matrix.tocoo()

        multi_rsum_csum_value = np.multiply(col_sum.take(cooccurence_matrix.col), 
                                                            row_sum.take(cooccurence_matrix.row)).A.squeeze()
        assert (multi_rsum_csum_value >=0).all() #check overflow
        multi_rsum_csum = sp.coo_matrix((multi_rsum_csum_value, 
                                                        (cooccurence_matrix.row, cooccurence_matrix.col)))
    
        deno = t_sum*cooccurence_matrix.tocsr() - multi_rsum_csum.tocsr()

        row_d = np.multiply(np.sqrt(row_sum) , np.sqrt((t_sum - row_sum)))
        col_d = np.multiply(np.sqrt(col_sum ), np.sqrt((t_sum - col_sum)))
        assert (row_d >=0).all() #check overflow
        assert (col_d >=0).all() #check overflow
      
        col_d_target_value = col_d.take(cooccurence_matrix.col).A.squeeze()
        col_d_target = sp.coo_matrix((col_d_target_value, 
                                                    (cooccurence_matrix.row, cooccurence_matrix.col)))
        col_d_target.data = 1 / col_d_target.data

        row_d_target_value = row_d.take(cooccurence_matrix.row).A.squeeze()
        row_d_target = sp.coo_matrix((row_d_target_value, 
                                                    (cooccurence_matrix.row, cooccurence_matrix.col)))
        row_d_target.data = 1 / row_d_target.data

        cooccurence_matrix = deno.multiply(col_d_target.tocsr()).multiply(row_d_target.tocsr())
        
        ##set negative values to 0
        cooccurence_matrix[cooccurence_matrix < 0] = 0

        ##take square roots
        cooccurence_matrix = np.sqrt(cooccurence_matrix)

        #apply svd
        if self.svd_dim:
            #TODO : remove less frequent rows to accelerate computing speed of svd
            cooccurence_matrix = cooccurence_matrix.asfptype()
            svd = TruncatedSVD(self.svd_dim, algorithm = 'arpack')
            cooccurence_matrix = svd.fit_transform(cooccurence_matrix) # vocab_len * vector_dim
            self.svd = svd

        self.word_vectors = cooccurence_matrix
        self.init_sims()

        return self

    def init_sims(self, replace=False):
        #方法本身就已經normalize
        if getattr(self, 'word_vectors_norm', None) is None or replace:
            self.word_vectors_norm = self.word_vectors

    def get_similarity(self, term1, term2):


        v1 = self.get_word_vector(term1)
        v2 = self.get_word_vector(term2)

        sim =  np.corrcoef(v1, v2)[0, 1]
    
        return sim
    
    def one2many_similarity(self, one_v, many_v, normalized=True):
        one_v = one_v.reshape(1, -1) # 1*dim
        many_v = many_v.reshape(-1, self.get_dim()) # n*dim

        # NOTE! 
        # there are some zero vectors with std = 0, causing errors(ZeroDivision) when calculating  corrcoef!
        std = np.std(many_v, axis=1)
        nonzero_mask = std != 0
        n_vectors = many_v.shape[0]
        sims = np.zeros(n_vectors)
        vectors_with_std = many_v[nonzero_mask,]
        # sims_with_std = np.apply_along_axis(lambda x: np.corrcoef(x,mean)[0,1], 1,vectors_with_std)
        sims_with_std = (1 - cdist(one_v, vectors_with_std, metric='correlation')).squeeze() #faster!!
        sims[nonzero_mask] = sims_with_std

        return sims
