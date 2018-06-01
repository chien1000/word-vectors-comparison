#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: ./tag.sh [de|bi|ce|proto|baseline]"
    exit 1
fi

root_dir=data/ner
inst_dir=$root_dir/instances
model_dir=$root_dir/models
model=$model_dir/$1/$1.model

predict_dir=$root_dir/predict
if [ ! -d $predict_dir ]; then
    mkdir $predict_dir
fi

test_corpora=$inst_dir/test.$1.inst

test_predict=$predict_dir/test.$1.predict

echo "[test_corpora] => "$test_corpora
echo "[model]        => "$model

eval_dir=$root_dir/eval
if [ ! -d $eval_dir ]; then
    mkdir $eval_dir
fi

echo "[EXECUTE]:     => crfsuite tag -m $model -r $test_corpora > $test_predict"
~/local/bin/crfsuite tag -m $model -r $test_corpora > $test_predict
gold_test=$root_dir/eng.test
paste $gold_test $test_predict > $eval_dir/test.$1.eval
sed -i 's/^\t//g' $eval_dir/test.$1.eval

echo "----Performance(Test)----"
perl src/conlleval.pl -d "\t" < $eval_dir/test.$1.eval

~/local/bin/crfsuite tag -qt -m $model  $test_corpora
#which one is right???

