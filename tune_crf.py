#! python3
import subprocess
import os 
import re
import logging

from ner_embedding_features.src import enner

# logger = logging.getLogger('__name__')
# logger.setLevel(logging.INFO)

def tune_crf(m, output_dir):
    validate_performance = ['model, precision, recall, F1']

    ner_trainp_data = 'ner_embedding_features/data/ner/eng.train_p'
    ner_dev_data = 'ner_embedding_features/data/ner/eng.dev' 
    
    c1_candidates = [1, 0.1, 0.01, 0.001]
    c2_candidates = [1, 0.1, 0.01, 0.001]
    
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

    best_c2 = None
    best_f1 = 0
    for c2 in c2_candidates:
        ner_model = os.path.join(ner_tuning_dir, 'model_' + m.get_name() + '_c1_0_c2_{}'.format(c2) )

        # crfsuite training
        print('training: c1=0, c2={}'.format(c2))
        bash_args = '~/local/bin/crfsuite learn -m {} -p feature.possible_states=1 \
        -p feature.possible_transitions=1 -a lbfgs -p max_iterations=500 -p c1=0 -p c2={} {}'.format(ner_model, c2, ner_trainp_features)

        process = subprocess.Popen(bash_args, shell=True, stderr=subprocess.PIPE)
        stderr = process.communicate()[1]
        print(stderr.decode('utf-8'))

        # crfsuite tagging
        print('tagging: c1=0, c2={}'.format(c2))

        bash_args = '~/local/bin/crfsuite tag -qt -m {} {}'.format(ner_model, ner_dev_features)
        process = subprocess.Popen(bash_args, shell=True, 
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(stdout.decode('utf-8'))
        print(stderr.decode('utf-8'))

        performance_pattern = r'Macro-average precision, recall, F1: \((.*)\)'
        performance = re.findall(performance_pattern, stdout.decode('utf-8'))[0].replace(' ','')
        s = '{},{}'.format(os.path.basename(ner_model), performance)
        validate_performance.append(s)

        f1 = float(performance.split(',')[2])
        if f1>best_f1:
            best_f1 = f1
            best_c2 = c2

    print('====================================')
    print('best c2 = {}'.format(best_c2))

    best_c1 = None
    best_f1 = 0
    for c1 in c1_candidates:
        ner_model = os.path.join(ner_tuning_dir, 'model_' + m.get_name() + '_c1_{}_c2_{}'.format(c1, best_c2) )

        # crfsuite training
        print('training: c1={}, c2={}'.format(c1, best_c2))
        bash_args = '~/local/bin/crfsuite learn -m {} -p feature.possible_states=1 \
        -p feature.possible_transitions=1 -a lbfgs -p max_iterations=500 -p c1={} -p c2={} {}'.format(ner_model, c1, best_c2, ner_trainp_features)

        process = subprocess.Popen(bash_args, shell=True, stderr=subprocess.PIPE)
        stderr = process.communicate()[1]
        print(stderr.decode('utf-8'))

        # crfsuite tagging
        print('tagging: c1={}, c2={}'.format(c1, best_c2))

        bash_args = '~/local/bin/crfsuite tag -qt -m {} {}'.format(ner_model, ner_dev_features)
        process = subprocess.Popen(bash_args, shell=True, 
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(stdout.decode('utf-8'))
        print(stderr.decode('utf-8'))

        performance_pattern = r'Macro-average precision, recall, F1: \((.*)\)'
        performance = re.findall(performance_pattern, stdout.decode('utf-8'))[0].replace(' ','')
        s='{},{}'.format(os.path.basename(ner_model), performance)
        validate_performance.append(s)


        f1 = float(performance.split(',')[2])
        if f1>best_f1:
            best_f1 = f1 
            best_c1 = c1

    print('====================================')
    s = 'best_c1 = {}, best_c2 = {}'.format(best_c1, best_c2)
    print(s)
    validate_performance.append(s)
    
    return best_c1, best_c2 , validate_performance

def main():
    pass

if __name__ == '__main__':
    main()