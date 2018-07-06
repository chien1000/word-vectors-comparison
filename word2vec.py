from gensim.models.word2vec import Word2Vec, LineSentence
from gensim.models import KeyedVectors
from base import BaseWordVectorizer
from exceptions import *
import os

MAX_WORDS_IN_BATCH = 10000

class W2vWordVectorizer(BaseWordVectorizer):
    """docstring for W2vWordVectorizor"""
    def __init__(self, vector_dim, algorithm='skip-gram', window_size=5, min_count=1,
                            sample=0.001, hs=0, negative=5, iter_num=5):
        # super(W2vWordVectorizer, self).__init__()
        self.vector_dim = vector_dim
        if algorithm != 'skip-gram' and algorithm != 'cbow':
            raise ValueError('algorithm must be skip-gram or cbow')
        self.algorithm = algorithm

        self.window_size = window_size
        self.min_count = min_count
        self.sample = sample
        # If hs = 1, hierarchical softmax will be used for model training. 
        # If hs is set to 0, and negative is non-zero, negative sampling will be used.
        self.hs = hs
        # If negative > 0, negative sampling will be used, If set to 0, no negative sampling is used.
        # the int for negative specifies how many “noise words” should be drawn (usually between 5-20). 
        self.negative = negative
        self.iter_num = iter_num 

    def get_name(self):
        return self.algorithm.upper()

    def get_mid(self):
        mid = '{}_d{}_window_{}_mincount_{}_hs_{}_neg_{}_iter_{}'.format(self.get_name(), self.vector_dim,
                                self.window_size, self.min_count, self.hs, self.negative, self.iter_num)
        return mid

    def fit_word_vectors(self, corpus_path):
        sg = 1 if self.algorithm == 'skip-gram' else 0
        sentences = LineSentence(corpus_path)
        self.model = Word2Vec(sentences, size=self.vector_dim, sg=sg,
            window=self.window_size, min_count=self.min_count, sample=self.sample,
            hs=self.hs, negative=self.negative, iter=self.iter_num, batch_words=MAX_WORDS_IN_BATCH )
        
        self.word_vectors = self.model.wv #KeyedVectors
        self.vocabulary = self.model.wv.vocab
        self.ind2word = None #TODO

    def init_sims(self, replace=False):

        self.word_vectors.init_sims(replace)

    def get_similarity(self, term1, term2):
        if not hasattr(self, 'word_vectors'):
            raise NotFittedError('Raw documented needed be fed first. Call fit_word_vectors(corpus_path)')
        #cosine sim
        sim = self.word_vectors.similarity(term1, term2)

        return sim

    def most_similar(self, positive=None, negative=None, topn=10, restrict_vocab=None, indexer=None):
        if not hasattr(self, 'word_vectors'):
            raise NotFittedError('Raw documented needed be fed first. Call fit_word_vectors(corpus_path)')

        result = self.word_vectors.most_similar(positive, negative, topn, restrict_vocab, indexer)
        
        return result

    def get_word_vector(self, term, use_norm=False):
        if not hasattr(self, 'word_vectors'):
            raise NotFittedError('Raw documented needed be fed first to estimate word vectors before\
             acquiring specific word vector. Call fit_word_vectors(corpus_path)')

        word_vec = self.word_vectors.word_vec(term, use_norm=use_norm)
        return word_vec

    def __getitem__(self, key):
        if not hasattr(self, 'word_vectors'):
            raise NotFittedError('Raw documented needed be fed first to estimate word vectors before\
             acquiring specific word vector. Call fit_word_vectors(corpus_path)')

        word_vec = self.word_vectors.__getitem__(key)

        return word_vec

    def save_model(self, model_path='models'):
         mid = self.get_mid()
         mfile = mid + '_wv.bin'
         mpath = os.path.join(model_path, mfile)

         self.word_vectors.save(mpath)

    def load_model(self, model_path=None):
        mid =  self.get_mid()
        mfile = mid + '_wv.bin'
        mpath = os.path.join(model_path, mfile)

        try:
            self.word_vectors = KeyedVectors.load(mpath)
            self.vocabulary = self.word_vectors.vocab
            self.ind2word = None

        except FileNotFoundError as e:
            print('model loading fails: file does not exist')
            self.word_vectors = None
        