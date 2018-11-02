#!/bin/bash
# Usage: ./Main-wus-sync.sh DATA_PREFIX RESULT_FOLDER_NAME NMT_ENSEMBLES BEAM MODEL_TYPE NMT_SEED(if not ensemble)
# Usage: ./Main-wus-sync.sh wusmod wus_phase1/wusmod_sync 1 3 wc 1
# Usage: ./Main-wus-sync.sh wusmod wus_phase1/wusinit_sync 1 3 wc 1

#Configuration options:
# w  - use lm over words(trained on the target data)
# c  - use lm over chars(trained on the extra target data)
# we - use lm over words(trained on the target and extra target data)
# ce - use lm over chars(trained on the target and extra target data)
# cw - use lm over words(trained on the target) and lm over chars(trained on the extra target data)
# cwe - use lm over words(trained on the target and extra target data) and lm over chars(trained on the target and extra target data)

###########################################
## POINTERS TO WORKING AND DATA DIRECTORIES
###########################################
#

export PF=$1
export DIR=/home/tanja/uzh-corpuslab-normalization

# data paths
export DATA=$DIR/data/wus/$1
export EXTRADATA=/home/massimo/cmt/wus/sms_word_aligned_lc.txt

export TRAINDATA=$DATA/train.txt
export DEVDATA=$DATA/tune.txt
export TESTDATA=$DATA/test.txt

#LM paths
export LD_LIBRARY_PATH=/home/christof/Chintang/swig-srilm:$LD_LIBRARY_PATH
export PYTHONPATH=/home/christof/Chintang/swig-srilm:$PYTHONPATH
export PATH=/home/christof/Chintang/SRILM/bin:/home/christof/Chintang/SRILM/bin/i686-m64:$PATH

#MERT path
export MERT=/home/christof/Chintang/uzh-corpuslab-morphological-segmentation/zmert_v1.50

#Pretrained NMT model
export MODEL=$DIR/results/wus_phase1

export BEAM=$4

export CONFIG=$5


# ensemble model
if [ -z $6 ];
then
    export NMT_ENSEMBLES=$3

    # results folder
    mkdir -p $DIR/results/$2/ensemble/${CONFIG}
    export RESULTS=$DIR/results/$2/ensemble/${CONFIG}

    # pretrained models
    nmt_predictors="nmt"
    nmt_path="$MODEL/${PF}_nmt_1"
    if [ $NMT_ENSEMBLES -gt 1 ]; then
    while read num; do nmt_predictors+=",nmt"; done < <(seq $(($NMT_ENSEMBLES-1)))
    while read num; do nmt_path+=",$MODEL/${PF}_nmt_$num"; done < <(seq 2 $NMT_ENSEMBLES)
    fi
    echo "$nmt_path"
# individual model
else
    export NMT_SEED=$6

    # results folder
    mkdir -p $DIR/results/$2/individual/${CONFIG}/${NMT_SEED}
    export RESULTS=$DIR/results/$2/individual/${CONFIG}/${NMT_SEED}

    # pretrained models
    nmt_predictors="nmt"
    nmt_path="$MODEL/${PF}_nmt_${NMT_SEED}"
    echo "$nmt_path"
fi

#
###########################################
## PREPARATION - src/trg splits and vocabulary
###########################################
#

# Prepare target and source dictionaries
cp $MODEL/${PF}_nmt_1/vocab.txt $RESULTS/vocab.trg
cp $MODEL/${PF}_nmt_1/vocab.txt $RESULTS/vocab.src


# Prepare train set
cut -f1 $TRAINDATA | grep . > $RESULTS/train.src
cut -f2 $TRAINDATA | grep . > $RESULTS/train.trg

# Prepare test set
cut -f1 $TESTDATA | grep . > $RESULTS/test.src
cut -f2 $TESTDATA | grep . > $RESULTS/test.trg

# Prepare validation set
cut -f1 $DEVDATA | grep . > $RESULTS/dev.src
cut -f2 $DEVDATA | grep . > $RESULTS/dev.trg

# Prepare training target file based on the extra data
cut -f2 $EXTRADATA > $RESULTS/extra.train.trg
# Extend training set
cat $RESULTS/train.trg $RESULTS/extra.train.trg > $RESULTS/train_ext.trg

##########################################
# TRAINING NMT
##########################################

### TO BE REPLACED WITH DYNET TRAINING
if [[ $CONFIG == "train" ]]; then # Train nmt models
    echo "TO BE REPLACED WITH DYNET TRAINING"

############################################
# DECODING NMT + EVALUATION on dev and test
############################################

elif [[ $CONFIG == "nmt" ]]; then # Only evaluate ensembles of nmt models

    PYTHONIOENCODING=utf8 python $DIR/src/norm_soft.py ensemble_test ${nmt_path} --test_path=$TESTDATA --beam=$BEAM --pred_path=test.out $RESULTS 

    # evaluate on tokens - detailed output
    PYTHONIOENCODING=utf8 python $DIR/src/accuracy-det.py eval $TRAINDATA $TESTDATA $RESULTS/test.out.predictions $RESULTS/Accuracy_test_det.txt $RESULTS/Errors_test.txt

else # nmt + LM

##########################################
# LM over words
##########################################

# Use target extended data for language model over words
if [[ $CONFIG == *"e"* ]]; then
    # Build vocab over morphemes
    PYTHONIOENCODING=utf8  python vocab_builder.py build $RESULTS/train_ext.trg $RESULTS/morph_vocab.txt --segments 
    # Apply vocab mapping
    PYTHONIOENCODING=utf8  python vocab_builder.py apply $RESULTS/train_ext.trg $RESULTS/morph_vocab.txt $RESULTS/train_ext.morph.itrg  --segments
    # train LM
    (ngram-count -text $RESULTS/train_ext.morph.itrg -lm $RESULTS/morfs.lm -order 3 -write $RESULTS/morfs.lm.counts -kndiscount -interpolate ) || { echo "Backup to ukn "; (ngram-count -text $RESULTS/train_ext.morph.itrg -lm $RESULTS/morfs.lm -order 3 -write $RESULTS/morfs.lm.counts -ukndiscount -interpolate );} || { echo "Backup to wb "; (ngram-count -text $RESULTS/train_ext.morph.itrg -lm $RESULTS/morfs.lm -order 3 -write $RESULTS/morfs.lm.counts -wbdiscount -interpolate );}

# Use only target train data for language model over words
else
    # Build vocab over morphemes
    PYTHONIOENCODING=utf8  python vocab_builder.py build $RESULTS/train.trg $RESULTS/morph_vocab.txt --segments 
    # Apply vocab mapping
    PYTHONIOENCODING=utf8  python vocab_builder.py apply $RESULTS/train.trg $RESULTS/morph_vocab.txt $RESULTS/train.morph.itrg  --segments
    # train LM
    (ngram-count -text $RESULTS/train.morph.itrg -lm $RESULTS/morfs.lm -order 3 -write $RESULTS/morfs.lm.counts -kndiscount -interpolate ) || { echo "Backup to ukn "; (ngram-count -text $RESULTS/train.morph.itrg -lm $RESULTS/morfs.lm -order 3 -write $RESULTS/morfs.lm.counts -ukndiscount -interpolate );} || { echo "Backup to wb "; (ngram-count -text $RESULTS/train.morph.itrg -lm $RESULTS/morfs.lm -order 3 -write $RESULTS/morfs.lm.counts -wbdiscount -interpolate );}

fi


##########################################
# LM over chars
##########################################

# Use target extended data for language model over chars
if [[ $CONFIG == *"e"* ]]; then
    # Apply vocab mapping
    PYTHONIOENCODING=utf8  python vocab_builder.py apply $RESULTS/train_ext.trg $RESULTS/vocab.trg $RESULTS/train_ext.char.itrg 
    # train LM
    (ngram-count -text $RESULTS/train_ext.char.itrg -lm $RESULTS/chars.lm -order 7 -write $RESULTS/chars.lm.counts -kndiscount -interpolate  -gt3min 1 -gt4min 1 -gt5min 1 -gt6min 1 -gt7min 1 ) || { echo "Backup to ukn "; (ngram-count -text $RESULTS/train_ext.char.itrg -lm $RESULTS/chars.lm -order 7 -write $RESULTS/chars.lm.counts -ukndiscount -interpolate  -gt3min 1 -gt4min 1 -gt5min 1 -gt6min 1 -gt7min 1);} || { echo "Backup to wb "; (ngram-count -text $RESULTS/train_ext.char.itrg -lm $RESULTS/chars.lm -order 7 -write $RESULTS/chars.lm.counts -wbdiscount -interpolate  -gt3min 1 -gt4min 1 -gt5min 1 -gt6min 1 -gt7min 1 );}

# Use only target train data for language model over chars
else
    # Apply vocab mapping
    PYTHONIOENCODING=utf8  python vocab_builder.py apply $RESULTS/extra.train.trg $RESULTS/vocab.trg $RESULTS/extra.train.char.itrg 
    # train LM
    (ngram-count -text $RESULTS/extra.train.char.itrg -lm $RESULTS/chars.lm -order 7 -write $RESULTS/chars.lm.counts -kndiscount -interpolate  -gt3min 1 -gt4min 1 -gt5min 1 -gt6min 1 -gt7min 1 ) || { echo "Backup to ukn "; (ngram-count -text $RESULTS/extra.train.char.itrg -lm $RESULTS/chars.lm -order 7 -write $RESULTS/chars.lm.counts -ukndiscount -interpolate  -gt3min 1 -gt4min 1 -gt5min 1 -gt6min 1 -gt7min 1);} || { echo "Backup to wb "; (ngram-count -text $RESULTS/extra.train.char.itrg -lm $RESULTS/chars.lm -order 7 -write $RESULTS/chars.lm.counts -wbdiscount -interpolate  -gt3min 1 -gt4min 1 -gt5min 1 -gt6min 1 -gt7min 1 );}

fi

##########################################
# MERT for NMT & LM + EVALUATION
##########################################

mkdir $RESULTS/mert
export MERTEXPER=$RESULTS/mert

cd $MERTEXPER

# NMT + Language Model over chars
if [[ $CONFIG == "c" ]] || [[ $CONFIG == "ce" ]]; then
    # passed to zmert: commands to decode n-best list from dev file
    echo "PYTHONIOENCODING=utf8 python $DIR/src/statistical_syncdecode.py ${nmt_path} $RESULTS --beam=$BEAM --test_path=$DEVDATA --pred_path=$MERTEXPER/nbest.out --lm_predictors=srilm_char --lm_orders=7 --lm_paths=$RESULTS/chars.lm --output_format=1"> SDecoder_cmd

    # passed to zmert: commands to decode 1-best list from test file
    echo "PYTHONIOENCODING=utf8 python $DIR/src/statistical_syncdecode.py ${nmt_path} $RESULTS --beam=$BEAM --test_path=$TESTDATA --pred_path=$MERTEXPER/test.out --lm_predictors=srilm_char --lm_orders=7 --lm_paths=$RESULTS/chars.lm" > SDecoder_cmd_test

    echo -e "cands_file=nbest.txt\ncands_per_sen=12\ntop_n=12\n\nnmt 1\nlm 0.001" > SDecoder_cfg.txt

    echo -e "nmt\t|||\t1\tFix\t0\t+1\t0\t+1\nlm\t|||\t0.001\tOpt\t0\t+Inf\t0\t+1\nnormalization = none" > params.txt



# NMT + Language Model over words
elif [[ $CONFIG == "w" ]] || [[ $CONFIG == "we" ]]; then
    # passed to zmert: commands to decode n-best list from dev file
    echo "PYTHONIOENCODING=utf8 python $DIR/src/statistical_syncdecode.py ${nmt_path} $RESULTS --beam=$BEAM --test_path=$DEVDATA --pred_path=$MERTEXPER/nbest.out --lm_predictors=srilm_morph --lm_orders=3 --lm_paths=$RESULTS/morfs.lm --output_format=1 --morph_vocab=$RESULTS/morph_vocab.txt"> SDecoder_cmd

    # passed to zmert: commands to decode 1-best list from test file
    echo "PYTHONIOENCODING=utf8 python $DIR/src/statistical_syncdecode.py ${nmt_path} $RESULTS --beam=$BEAM --test_path=$TESTDATA --pred_path=$MERTEXPER/test.out --lm_predictors=srilm_morph --lm_orders=3 --lm_paths=$RESULTS/morfs.lm --morph_vocab=$RESULTS/morph_vocab.txt" > SDecoder_cmd_test

    echo -e "cands_file=nbest.txt\ncands_per_sen=12\ntop_n=12\n\nnmt 1\nlm 0.1" > SDecoder_cfg.txt

    echo -e "nmt\t|||\t1\tFix\t0\t+1\t0\t+1\nlm\t|||\t0.1\tOpt\t0\t+Inf\t0\t+1\nnormalization = none" > params.txt


# NMT + Language Model over chars + Language Model over words
elif [[ $CONFIG == "cw" ]] || [[ $CONFIG == "cwe" ]]; then
    # passed to zmert: commands to decode n-best list from dev file
    echo "PYTHONIOENCODING=utf8 python $DIR/src/statistical_syncdecode.py ${nmt_path} $RESULTS --beam=$BEAM --test_path=$DEVDATA --pred_path=$MERTEXPER/nbest.out --lm_predictors=srilm_char,srilm_morph --lm_orders=7,3 --lm_paths=$RESULTS/chars.lm,$RESULTS/morfs.lm --output_format=1 --morph_vocab=$RESULTS/morph_vocab.txt" > SDecoder_cmd

    # passed to zmert: commands to decode 1-best list from test file
    echo "PYTHONIOENCODING=utf8 python $DIR/src/statistical_syncdecode.py ${nmt_path} $RESULTS --beam=$BEAM --test_path=$TESTDATA --pred_path=$MERTEXPER/test.out --lm_predictors=srilm_char,srilm_morph --lm_orders=7,3 --lm_paths=$RESULTS/chars.lm,$RESULTS/morfs.lm --morph_vocab=$RESULTS/morph_vocab.txt" > SDecoder_cmd_test

    echo -e "cands_file=nbest.txt\ncands_per_sen=12\ntop_n=12\n\nnmt 1\nlm1 0.1\nlm2 0.001" > SDecoder_cfg.txt

    echo -e "nmt\t|||\t1\tFix\t0\t+1\t0\t+1\nlm1\t|||\t0.1\tOpt\t0\t+Inf\t0\t+1\nlm2\t|||\t0.001\tOpt\t0\t+Inf\t0\t+1\nnormalization = none" > params.txt

else
 echo -e "Uknown configuration!"

fi

cp $DIR/src/ZMERT_cfg.txt $MERTEXPER
cp $RESULTS/dev.trg $MERTEXPER
cp $RESULTS/test.src $MERTEXPER

wait

java -cp $MERT/lib/zmert.jar ZMERT -maxMem 500 ZMERT_cfg.txt

## copy test out file - for analysis
cp test.out.predictions $RESULTS/test_out_mert.txt
cp test.out.eval $RESULTS/test.eval
#
## copy n-best file for dev set with optimal weights - for analysis
cp nbest.out.predictions $RESULTS/nbest_dev_mert.out
cp nbest.out.eval $RESULTS/dev.eval
#
cp SDecoder_cfg.txt.ZMERT.final $RESULTS/params-mert-ens.txt
#
#
##evaluate on tokens - detailed output for the test set
PYTHONIOENCODING=utf8 python $DIR/src/accuracy-det.py eval $TRAINDATA $TESTDATA $RESULTS/test_out_mert.txt $RESULTS/Accuracy_test_det.txt $RESULTS/Errors_test.txt

#rm -r $MERTEXPER

fi
