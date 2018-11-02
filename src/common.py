#!/usr/bin/env python
# -*- coding: utf-8 -*

import os
import codecs

# Default paths
SRC_FOLDER = os.path.dirname(__file__)
RESULTS_FOLDER = os.path.join(SRC_FOLDER, '../results')
DATA_FOLDER = os.path.join(SRC_FOLDER, '../data/')


# Model defaults
BEGIN_CHAR   = u'<s>'
STOP_CHAR   = u'</s>'
UNK_CHAR = u'<unk>'
BOUNDARY_CHAR = u' '

### IO handling and evaluation

def check_path(path, arg_name, is_data_path=True): #common
    if not os.path.exists(path):
        prefix = DATA_FOLDER if is_data_path else RESULTS_FOLDER
        tmp = os.path.join(prefix, path)
        if os.path.exists(tmp):
            path = tmp
        else:
            if is_data_path:
                print '%s incorrect: %s and %s' % (arg_name, path, tmp)
                raise ValueError
            else: #results path
                print tmp
                os.makedirs(tmp)
                path = tmp
    return path

def write_pred_file(output_file_path, final_results, format = 0):
    
    print 'len of predictions is {}'.format(len(final_results))
    if format == 0:
        predictions_path = output_file_path + '.predictions'
        with codecs.open(predictions_path, 'w', encoding='utf8') as predictions:
            for input, prediction in final_results:
                predictions.write(u'{}\t{}\n'.format(input, prediction))
    elif format == 1:
        id = 0
        predictions_path = output_file_path + '.predictions'
        with codecs.open(predictions_path, 'w', encoding='utf8') as predictions:
            for input, beam_predictions  in final_results:
                for beam_prediction in beam_predictions:
                    nmt_score,lm_scores,prediction, weighted_score = beam_prediction
                    predictions.write(u'{} ||| {} ||| {} ||| {}\n'.format(id, prediction, u' '.join([str(-nmt_score)] + [str(-s) for s in lm_scores]), -weighted_score))
                id +=1

    return

def write_param_file(output_file_path, hyper_params):
    
    with codecs.open(output_file_path, 'w', encoding='utf8') as f:
        for param in hyper_params:
            f.write(param + ' = ' + str(hyper_params[param]) + '\n')
    
    return

def write_eval_file(output_file_path, result, test_file_path, measure='Prediction Accuracy'):
    
    f = codecs.open(output_file_path + '.eval', 'w', encoding='utf8')
    f.write('File path = ' + str(test_file_path) + '\n')
    f.write('{} = {}\n'.format(measure, result))
    
    return
