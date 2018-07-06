# coding: utf-8

from lsa import LsaWordVectorizer
from hal import HalWordVectorizer
from coals import CoalsWordVectorizer
from lda import LdaWordVectorizer
from word2vec import W2vWordVectorizer
from glove import GloveWordVectorizer


run_config = {}
# run_config['corpus_path'] = 'data/wikipedia/enwiki-20180101-p30304p88444-processed.txt'
run_config['corpus_path'] = 'data/preprocessed/reuters_docperline.txt'
# run_config['corpus_path'] = 'data/preprocessed/wiki_30mt.txt'
# run_config['corpus_path'] = 'data/preprocessed/wiki_100percent.txt'

vector_dim = 100
min_count = 0
run_config['vector_dim'] = vector_dim
run_config['min_count'] = min_count
run_config['models'] = [
		LsaWordVectorizer(vector_dim=vector_dim, min_count=min_count, count_normalization='entropy'),
		LsaWordVectorizer(vector_dim=vector_dim, min_count=min_count, count_normalization='tfidf'),
                                HalWordVectorizer(max_features=vector_dim, min_count=min_count, window_size=8 ),
                                CoalsWordVectorizer(window_size=4,  max_features=14000, svd_dim=vector_dim, min_count=min_count),
                                LdaWordVectorizer(num_topics=vector_dim, alpha=50/vector_dim, eta=0.01, passes=10, min_count=min_count),
                                W2vWordVectorizer(vector_dim, algorithm='cbow', min_count=min_count, window_size=5), 
                                W2vWordVectorizer(vector_dim, algorithm='skip-gram', min_count=min_count, window_size=5), 
                                GloveWordVectorizer(vector_dim, min_count=min_count, max_iter=50, window_size=5)
                               ]
run_config['output_dir'] = 'exp_results' 
run_config['spec'] = 'test2' #'test2' #'wiki_30mt_300_try2'   'wiki_10percent_mc5_300'
run_config['eval'] = ['anal'] #['sim', 'anal', 'ner']
