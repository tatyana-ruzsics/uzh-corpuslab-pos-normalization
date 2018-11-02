#!/usr/bin/env python
# -*- coding: utf-8 -*
"""Synchronized decoding for combining soft attention models trained over chars with language model over segments(i.e. words/morphemes).

Usage:
  statistical_syncdecode.py [--dynet-mem MEM] [--beam=BEAM] [--pred_path=PRED_FILE]
  ED_MODEL_FOLDER MODEL_FOLDER --test_path=TEST_FILE [--input_format=INPUT_FORMAT]
  [--lm_predictors=LM_TYPES] [--lm_paths=LM_PATHS] [--lm_orders=LM_ORDERS]
  [--predictor_weights=WEIGHTS] [--output_format=FORMAT] [--lowercase=LOW] [--morph_vocab=MORPH_VOCAB] [--nmt_type=NMT_TYPE] [--verbose] [--indices=INDICES]
  
Arguments:
  ED_MODEL_FOLDER  ED model(s) folder, possibly relative to RESULTS_FOLDER, comma-separated
  MODEL_FOLDER     results folder, possibly relative to RESULTS_FOLDER

Options:
  -h --help                     show this help message and exit
  --dynet-mem MEM               allocates MEM bytes for DyNET [default: 500]
  --dev_path=DEV_FILE           dev set path, possibly relative to DATA_FOLDER, only for training
  --beam=BEAM                   beam width [default: 1]
  --test_path=TEST_FILE         test set path, possibly relative to DATA_FOLDER, only for evaluation
  --pred_path=PRED_FILE         name for predictions file in the test mode [default: 'best.test']
  --input_format=INPUT_FORMAT   coma-separated list of input, output columns [default: 0,1]
  --lowercase=LOW               use lowercased data [default: True]
  --lm_predictors=LM_TYPES      comma-separated type of the provided language models. E.g.: srilm_char, srilm_morph for SRILM character language model or SRILM morpheme language model
  --lm_paths=LM_PATHS           comma-separated paths of the language models in the same order as described in lm_predictors
  --lm_orders=LM_ORDERS         comma-separated orders of each of the language models in the same order as described in lm_predictors
  --predictor_weights=WEIGHTS   comma-separated weights of the nmt and language models (in the same order as described in lm_predictors)
  --output_format=FORMAT        format of the output: 0 - only predictions, 1 - n-best form with scores [default: 0]
  --morph_vocab=MORPH_VOCAB     mapping from morphs to int, needed for LM over morphs
  --nmt_type=NMT_TYPE           nmt model type: norm_soft, norm_soft_pos [default: 'norm_soft']
  --verbose                     verbose decoding
  --indices=INDICES             coma-seprated list of indices in the test data to be decoded
"""

from __future__ import division
from docopt import docopt
import os
import sys
import codecs
import random
import progressbar
import time
from collections import Counter, defaultdict

import math
import re

import dynet as dy
import numpy as np
import os
from itertools import izip
import copy

from common import BEGIN_CHAR,STOP_CHAR,UNK_CHAR, BOUNDARY_CHAR, SRC_FOLDER,RESULTS_FOLDER,DATA_FOLDER,check_path, write_pred_file, write_param_file, write_eval_file
from vocab_builder import build_vocabulary, Vocab
from norm_soft import SoftDataSet, SoftAttention
from norm_soft_pos import SoftDataSetFeat
from norm_soft_pos import SoftAttention as SoftAttentionFeat
from statistical_lm import SRILM_char_lm_loader, SRILM_morpheme_lm_loader
MAX_PRED_SEQ_LEN = 50 # option

def _compute_scores(lm_models, lm_states, weights, segment, nmt_vocab, STOP, BOUNDARY, UNK, eow=False, verbose=False):
    """compute scores of language model """
    
    lm_scores = []
    for i,(m,s) in enumerate(zip(lm_models,lm_states)):
    	# the segment is a morpheme
        if m.__class__.__name__ == 'SRILM_morpheme_lm_loader':
            #segment=" ".join([str(s) for s in segment])
#            print "morpheme:", segment
            # mapping of artifacts to unk
            if segment=='':
                segment = '<unk>'
            if verbose:
                print u'segment, state,eow: {},{},{}'.format(segment, s, eow)
            segment_enc = m.vocab.w2i.get(segment, UNK)
            m.set_state(s)
            if eow:
                lm_score = m.score(segment_enc, eow=1)
                if verbose:
                    print 'eow score: {}'.format(lm_score)
            else:
                lm_score = m.score(segment_enc)
                if verbose:
                    print 'score: {}'.format(lm_score)
            m.consume(segment_enc)
#            print "morpheme score:", lm_score
            lm_scores.append(-lm_score) #lm_scores.append(temp_lm_score)
            lm_states[i] = m.get_state()

	# the segment is a character
        elif m.__class__.__name__ == 'SRILM_char_lm_loader':
            temp_lm_score = 0
            new_state=s
            if verbose:
                print u'segment, state,eow: {},{},{}'.format(segment, s, eow)
            segment_enc = [nmt_vocab.w2i.get(c, UNK) for c in segment]
            if verbose:
                print u'segment encoded: {}'.format(segment_enc)

            for c in segment_enc:
                m.set_state(new_state)
                char_score = m.score(c)
                m.consume(c)
                temp_lm_score += -char_score
                new_state = m.get_state()
                if verbose:
                    print 'char, score: {}, {}'.format(nmt_vocab.i2w.get(c, UNK), char_score)
            m.set_state(new_state)
            if eow:
                char_score = m.score(STOP)
                m.consume(STOP)
                if verbose:
                    print 'char, score: {}, {}'.format(nmt_vocab.i2w.get(STOP, UNK), char_score)
            else:
                char_score = m.score(BOUNDARY)
                m.consume(BOUNDARY)
                if verbose:
                    print 'char, score: {}, {}'.format(nmt_vocab.i2w.get(BOUNDARY, UNK), char_score)
            lm_states[i]=m.get_state()
            temp_lm_score += -char_score
		
            lm_scores.append(temp_lm_score)
    if verbose:
        print lm_scores
    return np.array(lm_scores), lm_states
            
    
def predict_syncbeam(input, nmt_models, lm_models, weights, beam = 1, verbose = False, features=None):
    """predicts a string of characters performing synchronous beam-search."""
    dy.renew_cg()
    for nmt_model in nmt_models:
        if features:
            nmt_model.param_init(input,features)
        else:
            nmt_model.param_init(input)
    for lm_model in lm_models:
        lm_model.param_init()
    nmt_vocab = nmt_models[0].vocab # same vocab file for all nmt_models
    BEGIN   = nmt_vocab.w2i[BEGIN_CHAR]
    STOP   = nmt_vocab.w2i[STOP_CHAR]
    UNK       = nmt_vocab.w2i[UNK_CHAR]
    BOUNDARY = nmt_vocab.w2i[BOUNDARY_CHAR]

    m_hypos = [([m.s for m in nmt_models],[m.get_state() for m in lm_models] ,0.,np.array([0. for m in lm_models]), '', '')] # hypos to be expanded by morphemes
    m_complete_hypos = [] # hypos which end with STOP
    m_pred_length = 0 # number of morphemes
#        max_score = np.inf
    while m_pred_length <= MAX_PRED_SEQ_LEN and len(m_complete_hypos) < beam:# continue expansion while we don't have beam closed hypos #todo: MAX_PRED_SEQ_LEN should be changed to max morphemes number per word
        m_expansion = [] # beam * m_hypos expansion to be collected on the current iteration
        
        for m_hypo in m_hypos:
            hypos = [m_hypo] # hypos to be expanded by chars
            complete_hypos = [] # hypos which end with STOP or BOUNDARY
            pred_length = 0 # number of chars per morpheme
        
            while pred_length <= MAX_PRED_SEQ_LEN and len(hypos) > 0: # continue expansion while there is a hypo to expand
                expansion = [] # beam * m_hypos expansion to be collected on the current iteration
                for s_nmt, s_lm, nmt_log_p, lm_log_p, word, segment in hypos:
                    log_probs = np.array([-dy.log_softmax(m.predict_next_(s, scores=True)).npvalue() for m,s in zip(nmt_models,s_nmt)])
#                    print log_probs
                    log_probs = np.sum(log_probs, axis=0)
#                    print log_probs
                    top = np.argsort(log_probs,axis=0)[:beam]
                    expansion.extend(( (s_nmt, [copy.copy(s) for s in s_lm], nmt_log_p + log_probs[pred_id], lm_log_p, copy.copy(word), copy.copy(segment), pred_id) for pred_id in top ))
                hypos = []
                expansion.extend(complete_hypos)
                complete_hypos = []
                expansion.sort(key=lambda e: e[2])
                if verbose:
                    print u'expansion: {}'.format([(w+nmt_vocab.i2w.get(pred_id,UNK_CHAR),nmt_log_p,lm_log_p,pred_id) for _,_,nmt_log_p,lm_log_p,w,_,pred_id in expansion[:beam]])
                for e in expansion[:beam]:
                    s_nmt, s_lm, nmt_log_p, lm_log_p, word, segment, pred_id = e
                    if pred_id == STOP or pred_id == BOUNDARY:
                        complete_hypos.append((s_nmt,s_lm, nmt_log_p,lm_log_p,word, segment, pred_id))
                    else:
                        pred_char = nmt_vocab.i2w.get(pred_id,UNK_CHAR)
                        word+=pred_char
                        segment+=pred_char
                        hypos.append(([m.consume_next_(s, pred_id) for m,s in zip(nmt_models,s_nmt)],s_lm, nmt_log_p,lm_log_p,word, segment))
                pred_length += 1
            complete_hypos_new = []
            for e in complete_hypos:
                s_nmt, s_lm, nmt_log_p, lm_log_p, word, segment, pred_id = e
                if pred_id == STOP:
                    lm_score_new,s_lm_new = _compute_scores(lm_models, s_lm, weights, segment, nmt_vocab, STOP, BOUNDARY, UNK, True, verbose)
                    complete_hypos_new.append((s_nmt, s_lm_new, nmt_log_p, lm_log_p+lm_score_new, word, segment, pred_id))
                elif pred_id == BOUNDARY:
                    lm_score_new,s_lm_new = _compute_scores(lm_models, s_lm, weights, segment, nmt_vocab, STOP,BOUNDARY, UNK, False, verbose)
                    complete_hypos_new.append((s_nmt, s_lm_new, nmt_log_p, lm_log_p+lm_score_new, word, segment, pred_id))
            m_expansion.extend(complete_hypos_new)

        m_hypos = []
        m_expansion.extend(m_complete_hypos)
        m_complete_hypos = []
#        print m_expansion
#        print np.array(weights)
        m_expansion.sort(key=lambda e: e[2]*weights[0]+np.dot(e[3],np.array(weights[1:])))
        if verbose:
            print u'm_expansion: {}'.format([(w+nmt_vocab.i2w.get(pred_id,UNK_CHAR),nmt_log_p,lm_log_p) for _,_,nmt_log_p,lm_log_p,w,_,pred_id in m_expansion[:beam]])
        for e in m_expansion[:beam]:
            s_nmt, s_lm, nmt_log_p,lm_log_p, word, segment, pred_id = e
            if pred_id == STOP:
                m_complete_hypos.append(e)
            else: #BOUNDARY
                pred_char = nmt_vocab.i2w.get(pred_id, UNK_CHAR)
                word+=pred_char

                m_hypos.append(([m.consume_next_(s,pred_id) for m,s in zip(nmt_models,s_nmt)],s_lm, nmt_log_p,lm_log_p, word,''))
        m_pred_length += 1
            
    if not m_complete_hypos:
        # nothing found
        m_complete_hypos = [(nmt_log_p, word) for s_nmt, s_lm, nmt_log_p, lm_log_p, word, _ in m_hypos]
            
#    m_complete_hypos.sort(key=lambda e: e[2]+sum(e[3]))
    m_complete_hypos.sort(key=lambda e: e[2]*weights[0]+np.dot(e[3],np.array(weights[1:])))
    final_hypos = []
    for _,_, nmt_log_p,lm_log_p, word,_,_ in m_complete_hypos[:beam]:
        final_hypos.append((nmt_log_p,lm_log_p, word, nmt_log_p*weights[0] + np.dot(lm_log_p,np.array(weights[1:]))))
    return final_hypos

def evaluate_syncbeam_nofeat(data, ed_models, lm_models, weights, beam, format, verbose =False):
        # data is a list of tuples (an instance of SoftDataSet with iter method applied)
    correct = 0.
    final_results = []
    for i,(input,output) in enumerate(data):
        predictions = predict_syncbeam(input, ed_models, lm_models, weights, beam, verbose)
        try:
            prediction = predictions[0][2]
        except:
            print i, input, output, predictions
#        print i,predictions
        if prediction == output:
            correct += 1
#        else:
#        if i < 5:
#            print u'{}, input: {}, pred: {}, true: {}'.format(i, input, prediction, output)
#            print predictions
        if verbose:
            print u'{}, input: {}, pred: {}, true: {}'.format(i, input, prediction, output)
            print predictions
        if format == 0:
            final_results.append((input,prediction))
        else:
            final_results.append((input,predictions )) # input, (nmt_score, lm_score, pred, weighted_score)
    accuracy = correct / len(data)
    return accuracy, final_results

def evaluate_syncbeam_feat(data, ed_models, lm_models, weights, beam, format, verbose =False):
    # data is a list of tuples (an instance of SoftDataSet with iter method applied)
    correct = 0.
    final_results = []
    for i,(input,output,feat) in enumerate(data):
        predictions = predict_syncbeam(input, ed_models, lm_models, weights, beam, verbose, feat)
        try:
            prediction = predictions[0][2]
        except:
            print i, input, output, predictions
        #        print i,predictions
        if prediction == output.lower():
            correct += 1
#        if i < 5:
#            print u'{}, input: {}, pred: {}, true: {}'.format(i, input, prediction, output)
        if verbose:
            print u'{}, input: {}, pred: {}, true: {}'.format(i, input, prediction, output)
            print predictions
        if format == 0:
            final_results.append((input,prediction))
        else:
            final_results.append((input,predictions )) # input, (nmt_score, lm_score, pred, weighted_score)
    accuracy = correct / len(data)
    return accuracy, final_results

if __name__ == "__main__":
    arguments = docopt(__doc__)
    print arguments
    
    np.random.seed(123)
    random.seed(123)

    model_folder = check_path(arguments['MODEL_FOLDER'], 'MODEL_FOLDER', is_data_path=False)

    print '=========EVALUATION ONLY:========='
    # requires test path, model path of pretrained path and results path where to write the results to
    assert arguments['--test_path']!=None
    
    print 'Loading data...'
    test_path = check_path(arguments['--test_path'], '--test_path')
    if arguments['--nmt_type']=='norm_soft_pos':
        data_set = SoftDataSetFeat
        evaluate_syncbeam = evaluate_syncbeam_feat
    else:
        data_set = SoftDataSet
        evaluate_syncbeam = evaluate_syncbeam_nofeat
    input_format = [int(col) for col in arguments['--input_format'].split(',')]
    test_data = data_set.from_file(test_path,input_format, arguments['--lowercase'])
    print 'Test data has {} examples'.format(test_data.length)
    
    print 'Checking if any special symbols in data...'
    data = set(test_data.inputs + test_data.outputs)
    for c in [BEGIN_CHAR, STOP_CHAR, UNK_CHAR]:
        assert c not in data
    print 'Test data does not contain special symbols'
    
    pc = dy.ParameterCollection()
    
    weights = [float(w) for w in arguments['--predictor_weights'].split(',')]

    ed_models= []
    ed_model_params = []
    ## loading the nmt models
    for i,path in enumerate(arguments['ED_MODEL_FOLDER'].split(',')):
        print '...Loading nmt model {}'.format(i)
        ed_model_folder =  check_path(path, 'ED_MODEL_FOLDER_{}'.format(i), is_data_path=False)
        best_model_path  = ed_model_folder + '/bestmodel.txt'
        hypoparams_file_reader = codecs.open(ed_model_folder + '/best.dev', 'r', 'utf-8')
        hyperparams_dict = dict([line.strip().split(' = ') for line in hypoparams_file_reader.readlines()])
        model_hyperparams = {'INPUT_DIM': int(hyperparams_dict['INPUT_DIM']),
            'HIDDEN_DIM': int(hyperparams_dict['HIDDEN_DIM']),
                'LAYERS': int(hyperparams_dict['LAYERS'])}
        # vocab folder is taken from the first nmt folder
        vocab_path = check_path(arguments['ED_MODEL_FOLDER'].split(',')[0], 'ED_MODEL_FOLDER_0', is_data_path=False) + '/vocab.txt'
        model_hyperparams['VOCAB_PATH'] = vocab_path
        ed_model_params.append(pc.add_subcollection('ed{}'.format(i)))
        if arguments['--nmt_type']=='norm_soft_pos':
            feat_vocab_path = check_path(arguments['ED_MODEL_FOLDER'].split(',')[0], 'ED_MODEL_FOLDER_0', is_data_path=False) + '/feat_vocab.txt'
            model_hyperparams['FEAT_VOCAB_PATH'] = feat_vocab_path
            model_hyperparams['FEAT_INPUT_DIM'] = int(hyperparams_dict['FEAT_INPUT_DIM'])
            ed_model =  SoftAttentionFeat(ed_model_params[i], model_hyperparams,best_model_path)
        else:
            ed_model =  SoftAttention(ed_model_params[i], model_hyperparams,best_model_path)
        
        ed_models.append(ed_model)
    ensemble_number = len(ed_models)

    lm_models= []
    lm_model_params = []
    ## loading the language models
    for i,(lm_type,path,order) in enumerate(\
		zip(\
		    arguments['--lm_predictors'].split(','),\
		    arguments["--lm_paths"].split(','),\
		    [int(o) for o in arguments["--lm_orders"].split(',')]\
		)):
        lm_model_folder =  check_path(path, 'LM_MODEL_FOLDER_{}'.format(i), is_data_path=False)
        if lm_type=="srilm_char":
            print '...Loading lm model {} from path {}'.format(i,lm_model_folder)
            lm_model =  SRILM_char_lm_loader(lm_model_folder, order)
        elif lm_type=="srilm_morph":
            lm_model = SRILM_morpheme_lm_loader(lm_model_folder,order)
            assert arguments['--morph_vocab'] != None
            lm_model.vocab = Vocab.from_file(check_path(arguments['--morph_vocab'], 'morph_vocab', is_data_path=False))
        else:
            print "WARNING -- Could not load language model. Unknown type",lm_type,". Use 'srilm_char' or 'srilm_morph'"
        lm_models.append(lm_model)
    lm_number  = len(lm_models)

    output_file_path = os.path.join(model_folder,arguments['--pred_path'])

    # save best dev model parameters and predictions
    print 'Evaluating on test..'
    t = time.clock()
#    accuracy, test_results = evaluate_syncbeam(test_data.iter(indeces=1), ed_models, lm_models, weights, int(arguments['--beam']), int(arguments['--output_format']), verbose =True)
#    accuracy, test_results = evaluate_syncbeam(test_data.iter(), ed_models, lm_models, weights, int(arguments['--beam']), int(arguments['--output_format']))
    if arguments['--indices']:
        indices = [int(ind) for ind in arguments['--indices'].split(',')]
        accuracy, test_results = evaluate_syncbeam(test_data.iter(indices), ed_models, lm_models, weights, int(arguments['--beam']), int(arguments['--output_format']), verbose=arguments['--verbose'])
    else:
        accuracy, test_results = evaluate_syncbeam(test_data.iter(), ed_models, lm_models, weights, int(arguments['--beam']), int(arguments['--output_format']), verbose=arguments['--verbose'])
    print 'Time: {}'.format(time.clock()-t)
    print 'accuracy: {}'.format(accuracy)
    write_pred_file(output_file_path, test_results, int(arguments['--output_format']))
    write_eval_file(output_file_path, accuracy, test_path)
