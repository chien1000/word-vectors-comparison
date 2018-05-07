from collections import defaultdict
import six
from six import string_types

from hal import HalWordVectorizer, _make_int_array
from corpus import LineCorpus
from stop_words import ENGLISH_CLOSED_CLASS_WORDS
from exceptions import *

import numbers
import numpy as np 
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from gensim import matutils

class CoalsWordVectorizer(HalWordVectorizer):
    """docstring for CoalsWordVectorizer"""
    def __init__(self, window_size = 10, max_features=None, svd_dim=None,
                        vocabulary=None, dtype=np.int64):

        self.window_size = window_size
        self.max_features = max_features
        self.svd_dim = svd_dim
        if max_features is not None:
            if (not isinstance(max_features, numbers.Integral) or
                    max_features <= 0):
                raise ValueError(
                    "max_features=%r, neither a positive integer nor None"
                    % max_features)

        self.stop_words = set(ENGLISH_CLOSED_CLASS_WORDS)
        # self.stop_words = None
        self.vocabulary = vocabulary
        self.dtype = dtype

    def get_name(self):
        return 'COALS'

    def get_mid(self):
        mid = '{}_d{}_{}_window_{}'.format(self.get_name(), self.svd_dim, self.max_features, self.window_size)
        return mid

    def _count_cooccurence(self, docs, fixed_vocab):
        """Create sparse feature matrix, and vocabulary where fixed_vocab=False
        """
        if fixed_vocab:
            vocabulary = self.vocabulary
        else:
            # Add a new value when a new vocabulary item is seen
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__ #自動幫新字產生index

        context_vocabulary = defaultdict()
        context_vocabulary.default_factory = context_vocabulary.__len__ #自動幫新字產生index

        #
        
        row = _make_int_array()
        col = _make_int_array()
        values = _make_int_array()

        window_size = self.window_size
        for doc in docs:
            doc = [t for t in doc.split(' ') if t not in self.stop_words]
            doc_length = len(doc)

            for i, feature in enumerate(doc):
                try:
                    feature_idx = vocabulary[feature]
                    for j in range(max(i - window_size, 0), min(i + window_size, doc_length-1)+1):
                        if j == i:
                            continue
                        context_word = doc[j]
                        context_idx = context_vocabulary[context_word]
                        row.append(feature_idx)
                        col.append(context_idx)
                        diff = abs(j-i)-1
                        values.append(window_size-diff)

                except KeyError:
                    # Ignore out-of-vocabulary items for fixed_vocab=True
                    continue

        if not fixed_vocab:
            # disable defaultdict behaviour
            vocabulary = dict(vocabulary) #不要自動幫新字產生index！！
            if not vocabulary:
                raise ValueError("empty vocabulary; perhaps the documents only"
                                 " contain stop words")
            context_vocabulary = dict(context_vocabulary)

            
        ###sort by alphebetic order
        sorted_features = sorted(six.iteritems(vocabulary))
        sorted_context = sorted(six.iteritems(context_vocabulary))

        map_index_v = np.empty(len(sorted_features), dtype=np.int32)
        for new_val, (term, old_val) in enumerate(sorted_features):
            vocabulary[term] = new_val
            map_index_v[old_val] = new_val

        map_index_c = np.empty(len(sorted_context), dtype=np.int32)
        for new_val, (term, old_val) in enumerate(sorted_context):
            context_vocabulary[term] = new_val
            map_index_c[old_val] = new_val

        row = map_index_v.take(row, mode='clip')
        col = map_index_c.take(col, mode='clip')

        values = np.frombuffer(values, dtype=np.intc)
        cooccurence_matrix = sp.coo_matrix((values, (row, col)), shape=(len(vocabulary), 
                                                                         len(context_vocabulary))
                                           ,dtype=self.dtype)
        cooccurence_matrix = cooccurence_matrix.tocsc()
        # cooccurence_matrix.sort_indices()

#         print(cooccurence_matrix.toarray())
        return vocabulary, context_vocabulary, cooccurence_matrix  

    def fit_word_vectors(self, corpus_path):

        # self._validate_vocabulary()
        docs = LineCorpus(corpus_path)
        vocabulary, context_vocabulary, cooccurence_matrix = self._count_cooccurence(docs, False)
        self.vocabulary = vocabulary
        self.ind2word = [None] * len(self.vocabulary)
        for k, v in self.vocabulary.items():
            self.ind2word[v] = k

        if self.max_features: #discard all but the k columns reflecting the most common open-class words
            freqs = np.sum(cooccurence_matrix, axis=0).A[0]
            k = self.max_features
            topk_ind = np.sort(np.argsort(-freqs)[:k])
            cooccurence_matrix = cooccurence_matrix[:, topk_ind]

            #update context vobabulary
            terms = list(context_vocabulary.keys())
            indices = np.array(list(context_vocabulary.values()))
            sort_ind = np.argsort(indices)
            inverse_context_vocabulary = [terms[ind] for ind in sort_ind]
            new_context_vocabulary = {inverse_context_vocabulary[ind]:new_ind for new_ind, ind in enumerate(topk_ind)}
            context_vocabulary = new_context_vocabulary

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

        row_d = np.multiply(row_sum , (t_sum - row_sum))
        col_d = np.multiply(col_sum , (t_sum - col_sum))
        assert (row_d >=0).all() #check overflow
        assert (col_d >=0).all() #check overflow
      
        col_d_target_value = col_d.take(cooccurence_matrix.col).A.squeeze()
        col_d_target = sp.coo_matrix((col_d_target_value, 
                                                    (cooccurence_matrix.row, cooccurence_matrix.col)))
        col_d_target.data = 1 / np.sqrt(col_d_target.data)

        row_d_target_value = row_d.take(cooccurence_matrix.row).A.squeeze()
        row_d_target = sp.coo_matrix((row_d_target_value, 
                                                    (cooccurence_matrix.row, cooccurence_matrix.col)))
        row_d_target.data = 1 / np.sqrt(row_d_target.data)

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

        self.context_vocabulary = context_vocabulary
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
        sims = np.apply_along_axis(lambda x: np.corrcoef(x,mean)[0,1], 1, self.word_vectors_norm)

        if not topn:
            return sims

        best = matutils.argsort(sims, topn=topn + len(all_words), reverse=True) 
        # ignore (don't return) words from the input
        result = [(self.ind2word[ind], 1/ (1+float(sims[ind]))) for ind in best if ind not in all_words]
        return result[:topn]