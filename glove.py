import subprocess
import sys
import os
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from base import BaseWordVectorizer
from exceptions import *

class GloveWordVectorizer(BaseWordVectorizer):
    """docstring for W2vWordVectorizor"""
    def __init__(self, vector_dim, bash_file='train.sh', window_size=5, min_count=1,max_iter=5):
        # super(W2vWordVectorizer, self).__init__()
        self.glove_dir = 'GloVe'
        self.vector_dim = vector_dim
        self.bash_path = os.path.join(self.glove_dir, bash_file)
        self.window_size = window_size
        self.min_count = min_count
        self.max_iter = max_iter 

    def get_name(self):
        return 'GLOVE'

    def get_mid(self):
        mid = '{}_d{}_window_{}_mincount_{}_iter_{}'.format(self.get_name(), self.vector_dim,
                                self.window_size, self.min_count, self.max_iter)

        return mid

    def fit_word_vectors(self, corpus_file):
        cwd = os.getcwd()
        corpus_path = os.path.join(cwd, corpus_file)
        corpus_name = os.path.splitext(os.path.basename(corpus_file))[0]
        bash_full_path = os.path.join(cwd, self.bash_path)
        vector_dim = self.vector_dim
        window_size = self.window_size
        min_count = self.min_count
        max_iter = self.max_iter
        save_file = 'glove_vectors_{}_dim_{}_mincount_{}_wsize_{}_iter_{}'.format(
                         corpus_name, vector_dim, min_count, window_size, max_iter)

        self.save_file = save_file

        # bash_args = [bash_full_path,'--corpus', corpus_path, '--vector_dim', str(vector_dim),
        #            '--min_count', str(min_count), '--window_size', str(window_size),
        #            '--max_iter', str(max_iter), '--save_file', save_file]
        bash_args = '{} --corpus {} --vector_dim {} --min_count {} --window_size {} --max_iter {} --save_file {}'.format(bash_full_path,
                            corpus_path, vector_dim, min_count, window_size, max_iter, save_file)
        print(bash_args)
        subprocess.check_output(args = bash_args, shell=True, cwd = self.glove_dir)

        #https://radimrehurek.com/gensim/scripts/glove2word2vec.html
        glove_file = datapath(os.path.join(cwd, self.glove_dir, save_file) + '.txt')
        tmp_file = get_tmpfile(os.path.join(cwd, self.glove_dir, save_file) + '_w2v.txt')
        glove2word2vec(glove_file, tmp_file)
        self.word_vectors = KeyedVectors.load_word2vec_format(tmp_file)
        self.vocabulary = self.word_vectors.vocab
        self.ind2word =self.word_vectors.index2word

    def init_sims(self, replace=False):

        self.word_vectors.init_sims(replace)

    def get_similarity(self, term1, term2):
        if not hasattr(self, 'word_vectors'):
            raise NotFittedError('Raw documented needed be fed first. Call fit_word_vectors(corpus_file)')
        #cosine sim
        sim = self.word_vectors.similarity(term1, term2)

        return sim

    def most_similar(self, positive=None, negative=None, topn=10, restrict_vocab=None, indexer=None):
        if not hasattr(self, 'word_vectors'):
            raise NotFittedError('Raw documented needed be fed first. Call fit_word_vectors(corpus_file)')

        result = self.word_vectors.most_similar(positive, negative, topn, restrict_vocab, indexer)
        
        return result

    def get_word_vector(self, term, use_norm=False):
        if not hasattr(self, 'word_vectors'):
            raise NotFittedError('Raw documented needed be fed first to estimate word vectors before\
             acquiring specific word vector. Call fit_word_vectors(corpus_file)')

        word_vec = self.word_vectors.word_vec(term, use_norm=use_norm)
        return word_vec
    
    def __getitem__(self, key):
        if not hasattr(self, 'word_vectors'):
            raise NotFittedError('Raw documented needed be fed first to estimate word vectors before\
             acquiring specific word vector. Call fit_word_vectors(corpus_file)')

        word_vec = self.word_vectors[key]

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
            self.ind2word = self.word_vectors.index2word

        except FileNotFoundError as e:
            print('model loading fails: file does not exist')
            self.word_vectors = None
