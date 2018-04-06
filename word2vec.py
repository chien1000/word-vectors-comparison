from gensim.models import Word2Vec
from base import BaseWordVectorizer

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

    def fit_word_vectors(self, raw_documents):
        sg = 1 if self.algorithm == 'skip-gram' else 0

        self.model = Word2Vec(raw_documents, size=self.vector_dim, sg=sg,
            window=self.window_size, min_count=self.min_count, sample=self.sample,
            hs=self.hs, negative=self.negative, iter=self.iter_num)

    def get_similarity(self, term1, term2):
        sim = self.model.wv.similarity(term1, term2)

        return sim

    def __getitem__(self, key):
        if not hasattr(self, 'model'):
            raise NotFittedError('Raw documented needed be fed first to estimate word vectors before\
             acquiring specific word vector. Call fit_word_vectors(raw_documents)')

        word_vec = self.model.wv.__getitem__(key)

        return word_vec


        