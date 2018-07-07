# coding: utf-8

from config import run_config

import evaluations
from evaluations import evaluate_word_sims, evaluate_word_analogies
from ner_embedding_features.src import enner
from corpus import LineCorpus
from base import get_vocabulary

import logging
import os
import json
import traceback
import subprocess
import re

#logging

logger = logging.getLogger('compare')
logger.setLevel(logging.INFO)
# logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
log_formatter = logging.Formatter('%(asctime)s  %(message)s')
ch.setFormatter(log_formatter)

# add the handlers to logger
logger.addHandler(ch)


#prepare corpus
corpus_path = run_config['corpus_path']

#set output dir
output_dir = run_config['output_dir']
if run_config['spec']:
    output_dir = os.path.join(output_dir, run_config['spec'])
    try:
        os.mkdir(output_dir)
    except FileExistsError as e:
        pass

## save config
config_save_path = os.path.join(output_dir, 'config.txt')
tmp = run_config.copy()
tmp['models']  = [m.get_mid() for m in tmp['models']] #for json dump
with open(config_save_path, 'a') as fout:
    json.dump(tmp, fout, indent=0)


#outputs
corpus_name = os.path.splitext(os.path.basename(corpus_path))[0]
output_file = '{}_dim_{}.txt'.format(corpus_name, run_config['vector_dim'])
output_path = os.path.join(output_dir, output_file)
if os.path.exists(output_path):
    import time
    ts = time.time()
    output_path = '{}_{}.txt'.format(output_path.split('.')[0], ts )
logger.info('#========= output file: {} ========='.format(output_path))
fh = logging.FileHandler(output_path)
fh.setLevel(logging.INFO) #   >info > debug

log_formatter = logging.Formatter('# %(asctime)s  \n%(message)s')
fh.setFormatter(log_formatter)

logger.addHandler(fh)
evaluations.logger = logger #TODO: use a seperate file for logging??


#evaluations
def eval_log_sim(m):
    wordsim353 = 'data/evaluations/wordsim353/combined.csv'
    rg = 'data/evaluations/rg_sim.csv'
    sim_datasets = [wordsim353, rg]
    sim_dataset_names = ['WordSim353', 'Rubenstein and Goodenough']

    for dataset, dataset_name in zip(sim_datasets, sim_dataset_names):
        logger.warning('# ========= {} ========='.format(dataset_name))
        pearson, spearman, oov_ratio = evaluate_word_sims(m, m.get_name(), dataset,  delimiter=',')
        
        logger.warning('!model,pearson, spearman, oov_ratio')
        logger.warning('!{},{:.4f},{:.4f},{:.4f}'.format(m.get_name(), pearson[0], spearman[0], oov_ratio))

ind2word=None
def eval_log_anal(m):
    google_anal = 'data/evaluations/google_analogies.txt'
    logger.warning('# ========= Google Analogies =========')
    
    restrict_vocab = 300000
    corpus = LineCorpus(corpus_path)
    global ind2word
    if ind2word is None:
        ind2word, vocab = get_vocabulary(corpus, min_count=run_config['min_count'], sort_by_frequency=True)
    ok_vocab = set(ind2word[:restrict_vocab])

    print('restrict_vocab = {}'.format(restrict_vocab))
    analogies_score, sections, oov_ratio = evaluate_word_analogies(m, m.get_name(), google_anal, 
        ok_vocab=ok_vocab, restrict_vocab=restrict_vocab, case_insensitive=True, dummy4unknown=False)
    
    semantic_correct, semantic_incorrect = 0, 0
    syntactic_correct, syntactic_incorrect = 0, 0
    for sec in sections:
        if 'Total' in sec['section']:
            continue

        if 'gram' in sec['section']:
            syntactic_correct += len(sec['correct'])
            syntactic_incorrect += len(sec['incorrect'])
        else:
            semantic_correct += len(sec['correct'])
            semantic_incorrect += len(sec['incorrect'])
    semantic_score = semantic_correct / (semantic_correct+semantic_incorrect) 
    syntactic_score = syntactic_correct / (syntactic_correct+syntactic_incorrect) 
    print('semantic #{}'.format(semantic_correct+semantic_incorrect))
    print('syntactic #{}'.format(syntactic_correct+syntactic_incorrect))

    logger.warning('!model, analogies_score, semantic_score, syntactic_score, oov_ratio')
    logger.warning('!{},{:.4f},{:.4f},{:.4f},{:.4f}'.format(m.get_name(), 
                                            analogies_score, semantic_score, syntactic_score, oov_ratio))

def eval_log_ner(m):
    logger.warning('# ========= CoNLL 2003 NER task =========')

    ner_train_data = 'ner_embedding_features/data/ner/eng.train'
    # ner_dev_data 
    ner_test_data = 'ner_embedding_features/data/ner/eng.test' 
    cost = 0.2 

    ner_dir = os.path.join(output_dir, 'ner')
    try:
        os.mkdir(ner_dir)
    except FileExistsError as e:
        pass

    logger.info('# {} : generating ner features'.format(m.get_name()))
    ner_train_features = os.path.join(ner_dir, 'train_features_' + m.get_name() + '.txt')
    ner_test_features = os.path.join(ner_dir, 'test_features_' + m.get_name() + '.txt')
    if not os.path.exists(ner_train_features):
        enner.generate_crfsuite_features(ner_train_data, ner_train_features,
                                          emb_type='de', wv=m)
    if not os.path.exists(ner_test_features):
        enner.generate_crfsuite_features(ner_test_data, ner_test_features,
                                          emb_type='de', wv=m)

    ner_model = os.path.join(ner_dir, 'model_' + m.get_name())

    # crfsuite training
    logger.info('# training crf model')
    bash_args = '~/local/bin/crfsuite learn -m {} -p feature.possible_states=1 \
    -p feature.possible_transitions=1 -a l2sgd -p c2={} {}'.format(ner_model, cost, ner_train_features)
    # bash_args = ['~/local/bin/crfsuite learn','-m', ner_model,
    #              '-p', 'feature.possible_states=1', '-p', 'feature.possible_transitions=1',
    #              '-a', 'l2sgd', '-p', 'c2={}'.format(cost), ner_train_features]
    # print(bash_args)

    process = subprocess.Popen(bash_args, shell=True, 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logger.debug(stdout.decode('utf-8'))
    logger.warning(stderr.decode('utf-8'))

    # crfsuite tagging
    logger.info('# tagging')
    bash_args = '~/local/bin/crfsuite tag -qt -m {} {}'.format(ner_model, ner_test_features)
    process = subprocess.Popen(bash_args, shell=True, 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logger.warning(stdout.decode('utf-8'))
    logger.warning(stderr.decode('utf-8'))

    performance_pattern = r'Macro-average precision, recall, F1: \((.*)\)'
    performance = re.findall(performance_pattern, stdout.decode('utf-8'))[0].replace(' ','')
    logger.warning('!model, precision, recall, F1')
    logger.warning('!{},{}'.format(m.get_name(), performance))


#model
models = run_config['models']
logger.info('#========= Evaluate models: =========')
for m in models:
    logger.info('# ' + m.get_name())

while len(models)>0:
    m = models.pop(0)
    try:
        logger.info('# --- loading {} ---\n '.format(m.get_name()))
        m.load_model(output_dir)
        if m.word_vectors is None:
            logger.info('# --- training {} ---\n '.format(m.get_name()))
            m.fit_word_vectors(corpus_path)
            m.save_model(output_dir)

        #evalutaions
        if 'sim' in run_config['eval']:
            eval_log_sim(m)
        if 'anal' in run_config['eval']:
            eval_log_anal(m)
        if 'ner' in run_config['eval']:
            eval_log_ner(m)

    except Exception as e:
        s = traceback.format_exc()
        logger.error('# error occurred when training and evaluating {}'.format(m.get_name()))
        logger.error(s)


