#! python3

import os 
import pickle
import numpy as np
import scipy.sparse as sp
from collections import Counter

from gensim import matutils
from six import string_types

from exceptions import *
import traceback

MODEL_PATH = 'models'

def get_vocabulary(docs, min_count=0, sort_by_frequency=True):
    # filter rare words according to self.min_count
    word_counter = Counter()
    for doc in docs:
        word_counter.update(doc.split())

    vocabulary = {}    
    freq_count = 0
    if sort_by_frequency:
        for w,c in word_counter.most_common():
            if c >= min_count:
                vocabulary[w] = freq_count
                freq_count+=1
            else:
                break
    else:
        for w, c in word_counter.items():
            if c >= min_count:
                vocabulary[w] = freq_count
                freq_count+=1
    
    ind2word = [None] * len(vocabulary)
    for k, v in vocabulary.items():
        ind2word[v] = k

    return ind2word, vocabulary

class BaseWordVectorizer(object):
    """BaseWordVectorizer"""
    def __init__(self, vector_dim):
        self.vector_dim = vector_dim
        self.word_vectors = None
        self.vocabulary = {} # word to ind
        self.ind2word = []  # ind to word

    def get_dim(self):
        return self.vector_dim

    def get_name(self):
        # the derived classes need to override the method
        # define model name
        raise NotImplementedError()

    def get_mid(self):
        # the derived classes need to override the method
        # define model id decided by parameters
        raise NotImplementedError()

    def __contains__(self, w):
        if not hasattr(self, 'vocabulary') :
            raise NotFittedError('call fit_word_vectors first')

        if w in self.vocabulary:
            return True
        else:
            return False

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = MODEL_PATH
        mid =  self.get_mid()

        # save wv matrix    
        mfile = mid + '_wv.npz'
        mpath = os.path.join(model_path, mfile)
        if hasattr(self, 'use_sp_matrix') and self.use_sp_matrix and sp.issparse(self.word_vectors):
            # for sparse matrix
            sp.save_npz(mpath, self.word_vectors)

        else:
            np.savez(mpath, wv=self.word_vectors)

        #save other data
        mfile = mid + '_data.bin'
        mpath = os.path.join(model_path, mfile)
        with open(mpath, 'wb') as fout:
            dump_dict = {'ind2word':self.ind2word}
            pickle.dump(dump_dict, fout)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = MODEL_PATH
        mid =  self.get_mid()
    
        try:
            #load wv matrix
            mfile = mid + '_wv.npz'
            mpath = os.path.join(model_path, mfile)

            if hasattr(self, 'use_sp_matrix') and self.use_sp_matrix:
                self.word_vectors = sp.load_npz(mpath)
            else:
                self.word_vectors = np.load(mpath)['wv']

            #save other data
            mfile = mid + '_data.bin'
            mpath = os.path.join(model_path, mfile)
            with open(mpath, 'rb') as fin:
                dump_dict = pickle.load(fin)
                self.ind2word = dump_dict['ind2word']
                self.vocabulary = {w:i for i, w in enumerate(self.ind2word)}
                if hasattr(self, 'init_sims'):
                    self.init_sims()

        except (FileNotFoundError,IOError) as e:
            print('model loading fails: file does not exist')
            self.word_vectors = None

        except Exception as e:
            s = traceback.format_exc()
            print(s)

    def fit_word_vectors(self, corpus_path):
        # the derived classes need to override the method
        # estimate word_vectors from the corpus 
        raise NotImplementedError()

    def init_sims(self, replace=False):
        # the derived classes need to override the method
        # compute word_vectors_norm
        raise NotImplementedError()

    def get_word_vector(self, term, use_norm=False):
        if not hasattr(self, 'vocabulary') or not hasattr(self, 'word_vectors') or self.word_vectors is None:

            raise NotFittedError('corpus needed be fed first to estimate word vectors before\
             acquiring specific word vector. Call fit_word_vectors(corpus_path)')

        ind = self.vocabulary.get(term)
        if ind is None:
            raise KeyError('term {} is not in the vocabulary'.format(term))

        if use_norm:
            if not hasattr(self, 'word_vectors_norm'):
                self.init_sims()
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
    
    def get_similarity(self, term1, term2):
        # the derived classes need to override the method
        # compute similarity between vectors of  term1 and term2
        raise NotImplementedError()

    # def one2many_similarity(self, one_v, many_v):

    def most_similar(self, positive=None, negative=None, topn=10, ok_vocab=None, restrict_vocab=None):
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

        #TODO: use restrict_vocab : sort vocab by frequencies
        if ok_vocab is not None:
            original_vocab = self.vocabulary
            self.vocabulary = {w:self.vocabulary[w] for w in ok_vocab}

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

        limited_ind = list(self.vocabulary.values())

        if sp.issparse(self.word_vectors_norm):
            self.word_vectors_norm = self.word_vectors_norm.toarray()
        limited = np.take(self.word_vectors_norm, limited_ind, axis=0)

        # sims = np.dot(limited, mean) #cosine of two unit vectors = dot
        sims = self.one2many_similarity(mean, limited)
        if not topn:
            return sims
        best_of_limited = matutils.argsort(sims, topn=topn + len(all_words), reverse=True)
        best = [limited_ind[i] for i in best_of_limited]
        # ignore (don't return) words from the input
        result = []
        for ind, l_ind in zip(best, best_of_limited):
            if ind not in all_words: #all_words = positive + negative words 
                result.append((self.ind2word[ind], sims[l_ind]))
       
        #recover vocabulary
        if ok_vocab is not None:
            self.vocabulary = original_vocab

        return result[:topn]
