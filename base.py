#! python3

import os 
import pickle
import numpy as np
import scipy.sparse as sp

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

