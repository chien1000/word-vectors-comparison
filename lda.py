import numpy as np
from gensim.models.ldamodel import LdaModel
from gensim import matutils
from gensim.corpora.textcorpus import TextCorpus
from sklearn.metrics.pairwise import cosine_similarity

from six import string_types
from six.moves import xrange

# from corpus import MyTextCorpus
from base import BaseWordVectorizer
from exceptions import *

class LdaWordVectorizer(BaseWordVectorizer):
    """docstring for LdaWordVectorizer"""
    def __init__(self, num_topics=50, alpha=1, eta=0.01, passes=1, random_state=None):
        # super(LdaWordVectorizer, self).__init__()
        self.num_topics = num_topics 
        self.alpha = alpha
        self.eta = eta
        self.passes = passes

        # self.id2word = id2word #https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/corpora/dictionary.py
        self.random_state = random_state

    def get_name(self):
        return 'LDA'

    def get_mid(self):
        mid = '{}_d{}_alpha_{:.4f}_beta_{}_pass_{}'.format(self.get_name(), self.num_topics, 
                                                                                        self.alpha, self.eta, self.passes)
        return mid

    def fit_word_vectors(self, corpus_path):
        # corpus = TextCorpus(corpus_path)
        corpus = TextCorpus(corpus_path,  token_filters=[]) #character_filters=[lambda x:x],
        id2word = corpus.dictionary #https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/corpora/dictionary.py
        
        self.model = LdaModel(corpus, num_topics=self.num_topics,
            alpha=self.alpha, eta=self.eta, passes=self.passes, 
            id2word=id2word, random_state=self.random_state)
        
        self.vocabulary = self.model.id2word.token2id
        self.ind2word =  self.model.id2word.id2token

        topic_word_dist = self.model.state.get_lambda()
        topic_word_dist = np.log(topic_word_dist)
        row_sum = topic_word_dist.sum(axis=0)

        self.word_vectors = topic_word_dist/row_sum
        self.word_vectors = self.word_vectors.transpose() # word * topic

        self.init_sims(replace=True)

    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.
        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!
        # Note that you **cannot continue training** after doing a replace. The model becomes
        # effectively read-only = you can call `most_similar`, `similarity` etc., but not `train`.
        """
        if getattr(self, 'word_vectors_norm', None) is None or replace:
            print("precomputing L2-norms of word weight vectors")
            if replace:
                for i in xrange(self.word_vectors.shape[0]):
                    self.word_vectors[i, :] /= np.sqrt((self.word_vectors[i, :] ** 2).sum(-1))
                self.word_vectors_norm = self.word_vectors
            else:
                self.word_vectors_norm = (self.word_vectors / np.sqrt((self.word_vectors ** 2).sum(-1))[..., np.newaxis]).astype(np.float32)

    def get_similarity(self, term1, term2):
        #cosine sim
        v1 = self.__getitem__(term1).reshape(1, -1)
        v2 = self.__getitem__(term2).reshape(1, -1)

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

        limited = self.word_vectors_norm if restrict_vocab is None else self.word_vectors_norm[:restrict_vocab]
        sims = np.dot(limited, mean) #cosine of two unit vectors = dot
        if not topn:
            return sims
        best = matutils.argsort(sims, topn=topn + len(all_words), reverse=True)
        # ignore (don't return) words from the input
        result = [(self.ind2word[ind], float(sims[ind])) for ind in best if ind not in all_words]
        return result[:topn]

    def get_word_vector(self, term, use_norm=False):
        if not hasattr(self, 'vocabulary') or not hasattr(self, 'word_vectors'):

            raise NotFittedError('corpus needed be fed first to estimate word vectors before\
             acquiring specific word vector. Call fit_word_vectors(corpus_path)')

        ind = self.vocabulary.get(term)
        if not ind:
            raise KeyError('term {} is not in the vocabulary'.format(term))

        if use_norm:
            word_vec = self.word_vectors_norm[ind, :]
        else:
            word_vec = self.word_vectors[ind, :]

        return word_vec

    def __getitem__(self, key):

        # if not hasattr(self, 'vocabulary') or not hasattr(self, 'word_vectors'):

        #     raise NotFittedError('corpus needed be fed first to estimate word vectors before\
        #      acquiring specific word vector. Call fit_word_vectors(corpus_path)')

        # ind = self.vocabulary.get(key)
        # if not ind:
        #     raise KeyError('term {} is not in the vocabulary'.format(key))

        # word_vec = self.word_vectors[ind, :]

        # return word_vec

        word_vec = self.get_word_vector(key)
        return word_vec
        