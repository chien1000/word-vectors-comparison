# coding: utf-8

from config import run_config

from evaluations import evaluate_word_sims, evaluate_word_analogies

import logging
import os
import json

#logging

logger = logging.getLogger('compare')
logger.setLevel(logging.DEBUG)
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
with open(config_save_path, 'w') as fout:
    json.dump(tmp, fout, indent=0)


#model
models = run_config['models']
logger.info('#========= Evaluate models: =========')
for m in models:
    logger.info(m.get_name())

for m in models:
    logger.info('# --- training {} ---\n '.format(m.get_name()))
    m.load_model(output_dir)
    if not m.model:
        m.fit_word_vectors(corpus_path)
        m.save_model(output_dir)
    # if m.get_name().lower() == 'skip-gram' or m.get_name().lower() == 'cbow':
    #     max_sentence_length = MAX_WORDS_IN_BATCH
    #     m.fit_word_vectors(docs.get_texts(max_sentence_length=max_sentence_length))
    # elif m.get_name().lower() == 'lsa':
    #     m.fit_word_vectors(docs.get_text_str())
    # elif m.get_name().lower() == 'lda':
    #     m.fit_word_vectors(docs)
    # elif m.get_name().lower() == 'glove':
    #     m.fit_word_vectors(corpus_path)
    # else:
    #     m.fit_word_vectors(docs.get_texts())

# lsa_wv.fit_word_vectors(docs.get_text_str())
# hal_wv.fit_word_vectors(docs.get_texts())
# coals_wv.fit_word_vectors(docs.get_texts())
# lda_wv.fit_word_vectors(docs)
# max_sentence_length = MAX_WORDS_IN_BATCH
# cbow_wv.fit_word_vectors(docs.get_texts(max_sentence_length=max_sentence_length))
# sg_wv.fit_word_vectors(docs.get_texts(max_sentence_length=max_sentence_length))
# glove_wv.fit_word_vectors(corpus_path)


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
fh.setLevel(logging.WARNING) #   >info > debug
logger.addHandler(fh)



wordsim353 = 'data/evaluations/wordsim353/combined.csv'
rg = 'data/evaluations/rg_sim.csv'
sim_datasets = [wordsim353, rg]
sim_dataset_names = ['WordSim353', 'Rubenstein and Goodenough']

for dataset, dataset_name in zip(sim_datasets, sim_dataset_names):
    logger.warning('# ========= {} ========='.format(dataset_name))
    logger.warning('model,pearson, spearman, oov_ratio')
    for m in models:
        pearson, spearman, oov_ratio = evaluate_word_sims(m, m.get_name(), dataset,  delimiter=',')
        logger.warning('{},{:.4f},{:.4f},{:.4f}'.format(m.get_name(), pearson, spearman, oov_ratio))

google_anal = 'data/evaluations/google_analogies.txt'
logger.warning('# ========= Google Analogies =========')
logger.warning('model, analogies_score, oov_ratio')

for m in models:
    analogies_score, sections, oov_ratio = evaluate_word_analogies(m, m.get_name(), google_anal, restrict_vocab=300000, case_insensitive=True, dummy4unknown=False)
    logger.warning('{},{:.4f},{:.4f}'.format(m.get_name(), analogies_score, oov_ratio))

