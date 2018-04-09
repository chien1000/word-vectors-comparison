import subprocess
import sys
import os
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from base import BaseWordVectorizer

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
        self.model = KeyedVectors.load_word2vec_format(tmp_file)
       
    def get_similarity(self, term1, term2):
        #cosine sim
        sim = self.model.similarity(term1, term2)

        return sim

    def __getitem__(self, key):
        if not hasattr(self, 'model'):
            raise NotFittedError('Raw documented needed be fed first to estimate word vectors before\
             acquiring specific word vector. Call fit_word_vectors(raw_documents)')

        word_vec = self.model[key]

        return word_vec


