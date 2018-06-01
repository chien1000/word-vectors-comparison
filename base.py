#! python3

import os 
import pickle

from exceptions import *
import traceback

MODEL_PATH = 'models'

class BaseWordVectorizer(object):
    """BaseWordVectorizer"""
    def __init__(self):
        super(BaseWordVectorizer, self).__init__()

    def get_dim(self):
        return self.vector_dim
  
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
        mfile = mid + '.bin'
        mpath = os.path.join(model_path, mfile)

        with open(mpath, 'wb') as fout:
            dump_dict = {'word_vectors':self.word_vectors, 
            'vocabulary':self.vocabulary, 'ind2word':self.ind2word}
            pickle.dump(dump_dict, fout)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = MODEL_PATH
        mid =  self.get_mid()
        mfile = mid + '.bin'
        mpath = os.path.join(model_path, mfile)

        try:
            with open(mpath, 'rb') as fin:
                dump_dict = pickle.load(fin)
                self.word_vectors = dump_dict['word_vectors']
                self.vocabulary = dump_dict['vocabulary']
                self.ind2word = dump_dict['ind2word']
                if hasattr(self, 'init_sims'):
                    self.init_sims()

        except FileNotFoundError as e:
            print('model loading fails: file does not exist')
            self.word_vectors = None

        except Exception as e:
            s = traceback.format_exc()
            print(s)

