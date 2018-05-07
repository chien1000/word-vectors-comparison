# coding: utf-8

from lsa import LsaWordVectorizer
from hal import HalWordVectorizer
from coals import CoalsWordVectorizer
from lda import LdaWordVectorizer
from word2vec import W2vWordVectorizer
from glove import GloveWordVectorizer


run_config = {}
# run_config['corpus_path'] = 'data/wikipedia/enwiki-20180101-p30304p88444-processed.txt'
# run_config['corpus_path'] = 'data/preprocessed/reuters_docperline.txt'
run_config['corpus_path'] = 'data/preprocessed/wiki_part.txt'

vector_dim = 100
run_config['vector_dim'] = vector_dim
run_config['models'] = [LsaWordVectorizer(vector_dim=vector_dim, count_normalization='tfidf'),
                                HalWordVectorizer(max_features=vector_dim ),
                                # CoalsWordVectorizer(window_size=4,  max_features=14000, svd_dim=vector_dim),
                                LdaWordVectorizer(num_topics=vector_dim, alpha=50/vector_dim, eta=0.01, passes=10),
                                W2vWordVectorizer(vector_dim, algorithm='cbow', min_count=1), 
                                W2vWordVectorizer(vector_dim, algorithm='skip-gram', min_count=1), 
                                GloveWordVectorizer(vector_dim, min_count=1)
                               ]
run_config['output_dir'] = 'exp_results'
run_config['spec'] = 'wiki_part'
