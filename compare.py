# coding: utf-8

from config import run_config

import evaluations
from evaluations import evaluate_word_sims, evaluate_word_analogies
from ner_embedding_features.src import enner

import logging
import os
import json
import traceback
import subprocess

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
evaluations.logger = logger

#model
models = run_config['models']
logger.info('#========= Evaluate models: =========')
for m in models:
    logger.info('# ' + m.get_name())

for m in models:
    try:
        logger.info('# --- loading {} ---\n '.format(m.get_name()))
        m.load_model(output_dir)
        if m.word_vectors is None:
            logger.info('# --- training {} ---\n '.format(m.get_name()))
            m.fit_word_vectors(corpus_path)
            m.save_model(output_dir)

    except Exception as e:
        s = traceback.format_exc()
        logger.error('# error occurred when training {}'.format(m.get_name()))
        logger.error(s)
    
 
if 'sim' in run_config['eval']:

    wordsim353 = 'data/evaluations/wordsim353/combined.csv'
    rg = 'data/evaluations/rg_sim.csv'
    sim_datasets = [wordsim353, rg]
    sim_dataset_names = ['WordSim353', 'Rubenstein and Goodenough']

    for dataset, dataset_name in zip(sim_datasets, sim_dataset_names):
        logger.warning('# ========= {} ========='.format(dataset_name))
        logger.warning('!model,pearson, spearman, oov_ratio')
        for m in models:
            try:
                pearson, spearman, oov_ratio = evaluate_word_sims(m, m.get_name(), dataset,  delimiter=',')
                logger.warning('!{},{:.4f},{:.4f},{:.4f}'.format(m.get_name(), pearson[0], spearman[0], oov_ratio))

            except Exception as e:
                s = traceback.format_exc()
                logger.error(s)


if 'anal' in run_config['eval']:

    google_anal = 'data/evaluations/google_analogies.txt'
    logger.warning('# ========= Google Analogies =========')
    logger.warning('!model, analogies_score, oov_ratio')

    for m in models:
        try:
            analogies_score, sections, oov_ratio = evaluate_word_analogies(m, m.get_name(), google_anal, restrict_vocab=300000, case_insensitive=True, dummy4unknown=False)
            logger.warning('!{},{:.4f},{:.4f}'.format(m.get_name(), analogies_score, oov_ratio))

        except Exception as e:
            s = traceback.format_exc()
            logger.error(s)
    

## NER testing
if 'ner' in run_config['eval']:
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

    for m in models:
        try:
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

        except Exception as e:
            s = traceback.format_exc()
            logger.error(s)
    
        



