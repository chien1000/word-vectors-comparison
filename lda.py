import numpy as np
from gensim.models.ldamodel import LdaModel
from gensim.models.callbacks import PerplexityMetric, Callback
from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora.textcorpus import TextCorpus
from sklearn.metrics.pairwise import cosine_similarity

from corpus import LineCorpus, MyTextCorpus
from base import BaseWordVectorizer, get_vocabulary
from exceptions import *
import os
import logging

class LdaWordVectorizer(BaseWordVectorizer):
    """docstring for LdaWordVectorizer"""
    def __init__(self, num_topics=50, alpha=1, eta=0.01, passes=1, 
        random_state=None, min_count=0, max_vocab_size=None):
        # super(LdaWordVectorizer, self).__init__()
        self.num_topics = num_topics 
        self.alpha = alpha
        self.eta = eta
        self.passes = passes

        # self.id2word = id2word #https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/corpora/dictionary.py
        self.random_state = random_state

        self.min_count = min_count
        self.max_vocab_size = max_vocab_size

    def get_dim(self):
        return self.num_topics
        
    def get_name(self):
        return 'LDA'

    def get_mid(self):
        mid = '{}_d{}_alpha_{:.4f}_beta_{}_pass_{}'.format(self.get_name(), self.num_topics, 
                                                                                        self.alpha, self.eta, self.passes)
        return mid

    def fit_word_vectors(self, corpus_path, holdout_path=None):
        # logger 
        corpus_name = os.path.splitext(os.path.basename(corpus_path))[0]
        log_file = os.path.join('exp_results', 'log_{}_{}.txt'.format(corpus_name, self.get_mid()))
        logging.basicConfig(filename=log_file,
                    format="%(asctime)s:%(levelname)s:%(message)s",
                    level=logging.INFO)

        corpus = TextCorpus(corpus_path, tokenizer=str.split, token_filters=[])
        # corpus = MyTextCorpus(corpus_path, tokenizer=str.split,
        #     token_filters=[], min_count=self.min_count) #character_filters=[lambda x:x],
        id2word = corpus.dictionary #https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/corpora/dictionary.py
        
        self.ind2word, self.vocabulary = get_vocabulary(LineCorpus(corpus_path), self.min_count, sort_by_frequency=True)
        if self.max_vocab_size is not None:
            self.ind2word = self.ind2word[:self.max_vocab_size]
            self.vocabulary = {w:i for i, w in enumerate(self.ind2word)}

        id2word.token2id = self.vocabulary
        id2word.id2token = self.ind2word
        id2word.dfs = {} # useless here
        print('vocabulary size: {}'.format(len(self.vocabulary)))

        if holdout_path is not None:
            holdout_corpus = TextCorpus(holdout_path, tokenizer=str.split, token_filters=[])
            perplexity_logger = PerplexityMetric(corpus=holdout_corpus, logger='shell') 
            callbacks = [perplexity_logger]
        else:
            callbacks = None

        self.model = LdaModel(corpus, num_topics=self.num_topics,
            alpha=self.alpha, eta=self.eta, passes=self.passes, 
            id2word=id2word, random_state=self.random_state,
            callbacks=callbacks)
        
        # self.model = LdaMulticore(corpus, num_topics=self.num_topics,
        #     alpha=self.alpha, eta=self.eta, passes=self.passes, 
        #     id2word=id2word, random_state=self.random_state, workers=2)
        
        # self.vocabulary = self.model.id2word.token2id
        # self.ind2word =  self.model.id2word.id2token

        topic_word_dist = self.model.state.get_lambda()
        topic_word_dist = np.log(topic_word_dist)
        col_sum = topic_word_dist.sum(axis=0)

        self.word_vectors = topic_word_dist/col_sum
        self.word_vectors = self.word_vectors.transpose() # word * topic

        self.init_sims(replace=True)

    def init_sims(self, replace=False):
        """
        from gensim
        Precompute L2-normalized vectors.
        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!
        """
        if getattr(self, 'word_vectors_norm', None) is None or replace:
            print("precomputing L2-norms of word weight vectors")
            if replace:
                for i in range(self.word_vectors.shape[0]):
                    self.word_vectors[i, :] /= np.sqrt((self.word_vectors[i, :] ** 2).sum(-1))
                self.word_vectors_norm = self.word_vectors
            else:
                self.word_vectors_norm = (self.word_vectors / np.sqrt((self.word_vectors ** 2).sum(-1))[..., np.newaxis]).astype(np.float32)

    def get_similarity(self, term1, term2):
        #cosine sim
        v1 = self.get_word_vector(term1, use_norm=True)
        v2 = self.get_word_vector(term2, use_norm=True)

        sim = v1.dot(v2)

        return sim

    def one2many_similarity(self, one_v, many_v, normalized=True):
        vector_dim = self.get_dim()
        if normalized:
            many_v = many_v.reshape(-1, vector_dim) # n * dim
            one_v = one_v.reshape(-1, 1) # dim * 1
            sims = np.dot(many_v, one_v) # cosine of two unit vectors = dot
        else:
            many_v = many_v.reshape(-1, vector_dim) # n*dim
            one_v = one_v.reshape(1, -1) # 1*dim
            sims = cosine_similarity(many, one)

        return sims.squeeze()
        