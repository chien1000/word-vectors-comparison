#! python3
import subprocess
import os 
import re
import logging

from ner_embedding_features.src import enner

# logger = logging.getLogger('__name__')
# logger.setLevel(logging.INFO)

def run_crf(algorithm, params, model_name, save_path, train_features, dev_features):
    '''
        params: list of tuples, ex: [('c1',0), ('c2',1)]
    '''
    params_lst = list(sum(params, ()))
    params_str = '_'.join(map(str, params_lst))

    ner_model = os.path.join(save_path, 'model_{}_{}'.format(model_name, params_str))

    # crfsuite training

    params_combined = list(map(lambda p: '-p {}={}'.format(p[0],p[1]), params))
    params_combined_str = ' '.join(params_combined)

    print('training: {}'.format(params_combined_str))
    bash_args = '~/local/bin/crfsuite learn -m {} -p feature.possible_states=1 \
    -p feature.possible_transitions=1 -a {} {} {}'.format(ner_model, 
        algorithm, params_combined_str, train_features)

    process = subprocess.Popen(bash_args, shell=True, stderr=subprocess.PIPE)
    stderr = process.communicate()[1]
    print(stderr.decode('utf-8'))

    # crfsuite tagging
    print('tagging: {}'.format(params_combined_str))

    bash_args = '~/local/bin/crfsuite tag -qt -m {} {}'.format(ner_model, dev_features)
    process = subprocess.Popen(bash_args, shell=True, 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(stdout.decode('utf-8'))
    print(stderr.decode('utf-8'))

    performance_pattern = r'Macro-average precision, recall, F1: \((.*)\)'
    performance = re.findall(performance_pattern, stdout.decode('utf-8'))[0].replace(' ','')

    performance = list(map(float, performance.split(',')))
    
    return performance
    
def tune_lbfgs(m, output_dir):
    validate_performance = ['parameters, precision, recall, F1']

    ner_trainp_data = 'ner_embedding_features/data/ner/eng.train_p'
    ner_dev_data = 'ner_embedding_features/data/ner/eng.dev' 
    
    ner_dir = os.path.join(output_dir, 'ner')
    ner_tuning_dir = os.path.join(output_dir, 'ner/tuning')
    try:
        os.mkdir(ner_dir)
    except FileExistsError as e:
        pass
    try:
        os.mkdir(ner_tuning_dir)
    except FileExistsError as e:
        pass

    print('# {} : generating ner features'.format(m.get_name()))
    ner_trainp_features = os.path.join(ner_tuning_dir, 'trainp_' + m.get_name() + '_features.txt')
    ner_dev_features = os.path.join(ner_tuning_dir, 'dev_' + m.get_name() + '_features.txt')
    if not os.path.exists(ner_trainp_features):
        enner.generate_crfsuite_features(ner_trainp_data, ner_trainp_features,
                                          emb_type='de', wv=m)
    if not os.path.exists(ner_dev_features):
        enner.generate_crfsuite_features(ner_dev_data, ner_dev_features,
                                          emb_type='de', wv=m)

    c1_candidates = [1, 0.1, 0.01, 0.001]
    c2_candidates = [1, 0.1, 0.01, 0.001]
    algorithm = 'lbfgs'

    best_c2 = None
    best_f1 = 0
    for c2 in c2_candidates:
        params = [('c1',0), ('c2', c2), ('max_iterations', 500)]
        performance = run_crf(algorithm, params, m.get_name(), 
            ner_tuning_dir, ner_trainp_features, ner_dev_features)
        
        # record performance
        params_lst = list(sum(params, ()))
        params_str = '_'.join(map(str, params_lst))
        s = '{},{p[0]},{p[1]},{p[2]}'.format(params_str, p=performance)
        validate_performance.append(s)

        f1 = performance[2]

        if f1 > best_f1:
            best_f1 = f1
            best_c2 = c2

    print('====================================')
    print('best c2 = {}'.format(best_c2))

    best_c1 = None
    best_f1 = 0
    for c1 in c1_candidates:
        params = [('c1',c1), ('c2', best_c2)]
        performance = run_crf(algorithm, params, m.get_name(), 
            ner_tuning_dir, ner_trainp_features, ner_dev_features)
        
        # record performance
        params_lst = list(sum(params, ()))
        params_str = '_'.join(map(str, params_lst))
        s = '{},{p[0]},{p[1]},{p[2]}'.format(params_str, p=performance)
        validate_performance.append(s)

        f1 = performance[2]

        if f1 > best_f1:
            best_f1 = f1
            best_c1 = c1

    print('====================================')
    s = 'best_c1 = {}, best_c2 = {}'.format(best_c1, best_c2)
    print(s)
    validate_performance.append(s)
    
    return best_c1, best_c2 , validate_performance


def tune_l2sgd(m, output_dir):
    validate_performance = ['parameters, precision, recall, F1']

    ner_trainp_data = 'ner_embedding_features/data/ner/eng.train_p'
    ner_dev_data = 'ner_embedding_features/data/ner/eng.dev' 
    
    ner_dir = os.path.join(output_dir, 'ner')
    ner_tuning_dir = os.path.join(output_dir, 'ner/tuning')
    try:
        os.mkdir(ner_dir)
    except FileExistsError as e:
        pass
    try:
        os.mkdir(ner_tuning_dir)
    except FileExistsError as e:
        pass

    print('# {} : generating ner features'.format(m.get_name()))
    ner_trainp_features = os.path.join(ner_tuning_dir, 'trainp_' + m.get_name() + '_features.txt')
    ner_dev_features = os.path.join(ner_tuning_dir, 'dev_' + m.get_name() + '_features.txt')
    if not os.path.exists(ner_trainp_features):
        enner.generate_crfsuite_features(ner_trainp_data, ner_trainp_features,
                                          emb_type='de', wv=m)
    if not os.path.exists(ner_dev_features):
        enner.generate_crfsuite_features(ner_dev_data, ner_dev_features,
                                          emb_type='de', wv=m)

    c2_candidates = [10, 1, 0.1, 0.01, 0.001]
    algorithm = 'l2sgd'

    best_c2 = None
    best_f1 = 0
    for c2 in c2_candidates:
        params = [('c2', c2), ('max_iterations', 1000)]
        performance = run_crf(algorithm, params, m.get_name(), 
            ner_tuning_dir, ner_trainp_features, ner_dev_features)
        
        # record performance
        params_lst = list(sum(params, ()))
        params_str = '_'.join(map(str, params_lst))
        s = '{},{p[0]},{p[1]},{p[2]}'.format(params_str, p=performance)
        validate_performance.append(s)

        f1 = performance[2]

        if f1 > best_f1:
            best_f1 = f1
            best_c2 = c2

    print('====================================')
    s = 'best c2 = {}'.format(best_c2)
    print(s)
    validate_performance.append(s)
    
    return best_c2 , validate_performance


def main():
    pass

if __name__ == '__main__':
    main()