import numpy as np
import scipy.sparse as sp

import os
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
# from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from six import string_types
from gensim import matutils

from datetime import datetime
import pickle

from base import BaseWordVectorizer, get_vocabulary, MODEL_PATH
from corpus import LineCorpus
from exceptions import *

def word_entropy(vecs):
    if sp.issparse(vecs):
        vecs = vecs.toarray()

    vecs = vecs + 0.000001 # smoothing
    row_sum = vecs.sum(axis=1).reshape(-1, 1)

    vecs = vecs / row_sum
    assert (vecs>=0).all()
    vecs_log = np.log(vecs)

    H = -1 * np.multiply(vecs, vecs_log).sum(axis=1)

    return H

class LsaWordVectorizer(BaseWordVectorizer):
    def __init__(self, vector_dim, count_normalization=None, min_count=0):
        
#         super(LsaWordVectorizer, self).__init__()

        self.vector_dim = vector_dim
#         self.min_df = min_df
        
        self.count_normalization = count_normalization
        if count_normalization is not None:
            if count_normalization not in set(['entropy','tfidf']):
                raise ValueError("not a valid normalization method: {}".format(count_normalization))

        self.min_count = min_count

    def get_name(self):
        count_norm = self.count_normalization or 'no_count_norm'
        return 'LSA-{}'.format(count_norm)

    def get_mid(self):
        count_norm = self.count_normalization or 'no_count_norm'
        # corpus_name = ''
        # if hasattr(self, 'corpus_path'): 

        name = self.get_name().split('-')[0]
        mid = '{}_d{}_{}_mc{}'.format(name, self.vector_dim, count_norm, self.min_count)
        return mid
        
    def fit_word_vectors(self, corpus_path):
        docs = LineCorpus(corpus_path)
        self.ind2word, self.vocabulary = get_vocabulary(docs, self.min_count)
        print('vocabulary size: {}'.format(len(self.vocabulary)))

        if self.count_normalization is None:
            self.vectorizer = CountVectorizer(vocabulary=self.vocabulary,tokenizer=str.split)
        elif self.count_normalization == 'entropy':
            self.vectorizer = CountVectorizer(vocabulary=self.vocabulary,tokenizer=str.split)
        elif self.count_normalization == 'tfidf':
            self.vectorizer = TfidfVectorizer(vocabulary=self.vocabulary,tokenizer=str.split,
                                        sublinear_tf=True, use_idf=True)
        
        dtm = self.vectorizer.fit_transform(docs)

        tdm = dtm.T
        tdm = tdm.asfptype()

        if self.count_normalization == 'entropy':
            #apply entropy normalization
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print('apply entropy normalization')

            corpus_name = os.path.splitext(os.path.basename(corpus_path))[0]
            save_tdm_path =  '{}_mc{}_tdm.npz'.format(corpus_name, self.min_count)
            save_tdm_path = os.path.join(MODEL_PATH, save_tdm_path)
            save_ind2word_path =  '{}_mc{}_ind2word.bin'.format(corpus_name, self.min_count)
            save_ind2word_path = os.path.join(MODEL_PATH, save_ind2word_path)

            try:
                tdm = sp.load_npz(save_tdm_path)
                with open(save_ind2word_path, 'rb') as fin:
                    self.ind2word = pickle.load(fin)
                    self.vocabulary = {w:i for i, w in enumerate(self.ind2word)}

                print('load existed normalized tdm and vocab')

            except Exception as e:
                vlen = tdm.shape[0]
                H = np.zeros((vlen,1)) # row entropy
                step = 2000
                for i in range(0, vlen, step):
                    start, end = i, i+step
                    end = end if end < vlen else vlen
                    H[start:end,0] = word_entropy(tdm[start:end, ])
                    
                    if i % 2000 == 0:
                        print('finish computing entropy of {}/{} rows'.format(i, vlen))

                tdm.data = np.log(tdm.data+1)
                tdm = tdm.multiply(1/H)

                sp.save_npz(save_tdm_path, tdm)
                with open(save_ind2word_path, 'wb') as fout:
                    pickle.dump(self.ind2word, fout)

        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print('start performing svd')
        svd = TruncatedSVD(self.vector_dim, algorithm = 'arpack')
        tdm_svd = svd.fit_transform(tdm) # vocab_len * vector_dim (U * sigma)
        # tdm_svd = Normalizer(copy=False).fit_transform(tdm_svd) 
        
        self.svd = svd #components_ : vector_dim* doc_len (aka. transpose of T)
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

        v1 = self.get_word_vector(term1, use_norm=True)
        v2 = self.get_word_vector(term2, use_norm=True)

        sim = v1.dot(v2)

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
