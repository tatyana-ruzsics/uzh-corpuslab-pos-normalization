#!/usr/bin/env python
# -*- coding: utf-8 -*
""" This file evaluates results with details over seen/unseen segments(morphemes/words)

Usage:
  accuracy-det.py eval [--input_format=INPUT_FORMAT] [--lowercase=LOW] [--extended_train_data=EXT_FILE]
  TRAIN_DATA TEST_DATA PREDICTIONS RESULTS_FILE RESULTS_ERRORS_FILE
  accuracy-det.py eval_baseline [--input_format=INPUT_FORMAT] [--lowercase=LOW] [--error_file=ERR_FILE]
  TRAIN_DATA TEST_DATA
  accuracy-det.py eval_ambiguity [--input_format=INPUT_FORMAT] [--lowercase=LOW]
  TRAIN_DATA TEST_DATA PREDICTIONS RESULTS_FILE RESULTS_ERRORS_FILE
  accuracy-det.py eval_ambiguity_baseline [--input_format=INPUT_FORMAT] [--lowercase=LOW] [--error_file=ERR_FILE]
  TRAIN_DATA TEST_DATA
  

Arguments:
TRAIN_DATA          train file path
TEST_DATA           test file path
PREDICTIONS         path for the predictions for the test data
RESULTS_FILE        path to save the evaluation
RESULTS_ERRORS_FILE path to save errors

Options:
  -h --help                         show this help message and exit
  --input_format=INPUT_FORMAT       coma-separated list of input, output, features columns [default: 0,1]
  --lowercase=LOW                   use lowercased data [default: True]
  --extended_train_data=EXT_FILE    extended data used for LM training, one-column format
  --error_file=ERR_FILE             file to write the errors of baseline evaluation
"""

from __future__ import division
__author__ = 'Tanya'
from docopt import docopt
import codecs
import unicodedata
from collections import defaultdict



def evaluate_baseline(trainin,gold,input_format,lowercase=False, file_out_errors=None):
    train_lexicon_m = defaultdict(int)
    test_lexicon_m = defaultdict(int)
    train_lexicon_w = defaultdict(int)
    test_lexicon_w = defaultdict(int)
    test_dict = {} # test_word: dict(predict:freq)
    train_dict = {} # train_word: dict(predict:freq)
    
    input = input_format[0]
    pred = input_format[1]
    
    # Read lexicon of training set
    trainin_f = codecs.open(trainin,'r','utf-8')
    for i,line in enumerate(trainin_f):
        #if i < 40:
        #        print line
        if len(line.strip()) != 0:
            line = line.strip().lower() if lowercase else line.strip()
            lineitems = line.split('\t')
            word = lineitems[input]
            segm = lineitems[pred]
            train_lexicon_w[word] += 1
            morfs = lineitems[pred].split(' ')
            for m in morfs:
                train_lexicon_m[m] += 1
            if not word in train_dict.keys():
                train_dict[word] = {}
                train_dict[word][segm]=1
            else:
                if segm not in train_dict[word].keys():
                    train_dict[word][segm]=1
                else:
                    train_dict[word][segm]+=1

#    print train_dict['unds']

    # Read lexicon of test set
    gold_f = codecs.open(gold,'r','utf-8')
    for i,line in enumerate(gold_f):
        #        if i < 40:
        if len(line.strip()) != 0:
            #            print line
            line = line.strip().lower() if lowercase else line.strip()
            lineitems = line.split('\t')
            word = lineitems[input]
            segm = lineitems[pred]
            test_lexicon_w[word] += 1
            morfs = lineitems[pred].split(' ')
            for m in morfs:
                test_lexicon_m[m] += 1
            if not word in test_dict.keys():
                test_dict[word] = {}
                test_dict[word][segm]=1
            else:
                if segm not in test_dict[word].keys():
                    test_dict[word][segm]=1
                else:
                    test_dict[word][segm]+=1

#    print test_dict['unds']

    amb_segm_test_candidates = {k:v for k,v in test_dict.items() if k in train_dict.keys()} #the values are test frequencies - for statistics
    amb_segm_train = {k:train_dict[k] for k,v in amb_segm_test_candidates.items() if len(train_dict[k])>1} # the values are train frequencies - for prediction
    amb_segm_test = {k:v for k,v in amb_segm_test_candidates.items() if len(train_dict[k])>1} # the values are test frequencies - for statistics
#    print amb_segm_train.items()[:10]
    amb_segm_test_freq = {k:sum(v.values()) for k,v in amb_segm_test.items()}  # the values are test frequencies - for statistics
    amb = sum(amb_segm_test_freq.values())
    corr_amb = 0 # number of correct ambigous

    amb_segm_test_tie_candidates = {k:v.values() for k,v in amb_segm_train.items()} # the values are train frequencies - for tie candidates filtering based on train set
    amb_segm_test_tie_check = {k:v for k,v in amb_segm_test_tie_candidates.items() if v.count(v[0]) == len(v)} # the values are train frequencies - for printing
#    print amb_segm_test_tie_check.items()
    amb_segm_test_tie = {k:sum(test_dict[k].values()) for k,v in amb_segm_test_tie_candidates.items() if v.count(v[0]) == len(v)}   # the values are test frequencies - for statistics
    amb_tie = sum(amb_segm_test_tie.values())
    corr_amb_tie = 0 # number of correct ambigous with tie
    amb_notie = amb - amb_tie
    corr_amb_notie = 0 # number of correct ambigous with tie


    
    not_amb_segm_test = {k:v for k,v in test_dict.items() if k not in amb_segm_test.keys()}
    seen_freq = {k:sum(v.values()) for k,v in not_amb_segm_test.items() if k in train_lexicon_w.keys()}
    seen = sum(seen_freq.values())
    corr_seen = 0 # number of correct seen words
    
    unseen_freq = {k:sum(v.values()) for k,v in not_amb_segm_test.items() if not k in train_lexicon_w.keys()}
    unseen = sum(unseen_freq.values())
    corr_unseen = 0
    
    unseen_m_freq = {k:sum(v.values()) for k,v in not_amb_segm_test.items() if ( not k in train_lexicon_w.keys() and not all(m in train_lexicon_m.keys() for m in v.keys()[0].split(' ')) )}
    unseen_m = sum(unseen_m_freq.values())
    corr_unseen_m = 0 # number of correct unseen words - new morphs
    
    unseen_new_comb = unseen - unseen_m
    corr_unseen_comb = 0 # number of correct unseen words - new combinations
    allc = 0
    corr = 0

    # baseline statistics
    gold_f.seek(0)
    errors = {}
    for i,line in enumerate(gold_f):
        #if i < 5:
        if len(line.strip()) !=0:
            try:
                line = line.strip().lower() if lowercase else line.strip()
                lineitems = line.split('\t')
                w = lineitems[input]
                w_segm = lineitems[pred]
                lines = str(i)
                w_segm_morfs = w_segm.split(' ')
            
            except:
                print i, line
            allc +=1
            
            # seen and ambigous
            if w in amb_segm_test.keys():
                w_preds = amb_segm_train[w]
                w_baseline_pred = max(w_preds.keys(), key=lambda k: w_preds[k])
                if w_baseline_pred == w_segm:
                    corr +=1
                    corr_amb +=1
                    if w in amb_segm_test_tie.keys():
                        corr_amb_tie +=1
                    else:
                        corr_amb_notie +=1
            else:
                #new
                if w not in train_lexicon_w.keys():
                    w_baseline_pred = w
                    # new - old morphemes but new combination
                    if all(m in train_lexicon_m.keys() for m in w_segm_morfs):
                        if w_baseline_pred == w_segm:
                            corr +=1
                            corr_unseen +=1
                            corr_unseen_comb +=1
                    else:
                        # new - new morphemes
                        if w_baseline_pred == w_segm:
                            corr +=1
                            corr_unseen +=1
                            corr_unseen_m +=1

                #seen and unique
                else:
                    w_baseline_pred = train_dict[w].keys()[0]
                    if w_baseline_pred == w_segm:
                        corr +=1
                        corr_seen +=1
            # collect errors
            if not w_baseline_pred == w_segm:
                if (w,w_baseline_pred, w_segm) not in errors.keys():
                    errors[(w,w_baseline_pred, w_segm)] = [lines]
                else:
                    errors[(w,w_baseline_pred, w_segm)].append(lines)


    total_w_pred = sum(test_lexicon_w.values())
    print
    print "{:<50} {:>10} {:>10}\n".format("DATA","TRAIN", "TEST")
    print "{:<50} {:12d} {:8d}\n".format("# of target segment tokens:", sum(train_lexicon_m.values()), sum(test_lexicon_m.values()))
    print "{:<50} {:12d} {:8d}\n".format("# of source word tokens:", sum(train_lexicon_w.values()), total_w_pred)
    print "{:<50} {:21d} {:8.2f}%\n".format("# of ambigous source word tokens:", amb, amb/total_w_pred*100)
    print "{:<50} {:21d} {:8.2f}%\n".format("# of ambigous source word tokens - ties:", amb_tie, amb_tie/total_w_pred*100)
    print "{:<50} {:21d} {:8.2f}%\n".format("# of ambigous source word tokens - no ties:", amb_notie, amb_notie/total_w_pred*100)
    print "{:<50} {:21d} {:8.2f}%\n".format("# of unique source word tokens:", seen, seen/total_w_pred*100)
    print "{:<50} {:21d} {:8.2f}%\n".format("# of new source word tokens:", unseen, unseen/total_w_pred*100)
    print "{:<50} {:21d} {:8.2f}%\n".format("# of new source word tokens - new target segments:", unseen_m, unseen_m/total_w_pred*100)
    print "{:<50} {:21d} {:8.2f}%\n".format("# of new source word tokens - new combination:", unseen_new_comb, unseen_new_comb/total_w_pred*100)

    print "\nPERFORMANCE:\n"
    print "{:>60} {:11d}\n".format("Number of predictions total:", allc)
    print "{:>60} {:11d} {:8.2f}%\n".format("Number of correct predictions total:", corr, corr/allc*100)
    print "{:>60} {:11d} {:8.2f}%\n".format("- ambigous:", corr_amb, corr_amb/amb*100)
    print "{:>60} {:11d} {:8.2f}%\n".format("- ambigous(ties):", corr_amb_tie, corr_amb_tie/amb_tie*100)
    print "{:>60} {:11d} {:8.2f}%\n".format("- ambigous(no ties):", corr_amb_notie, corr_amb_notie/amb_notie*100)
    if seen !=0:
        print "{:>60} {:11d} {:8.2f}%\n".format("- unique:", corr_seen, corr_seen/seen*100)
    print "{:>60} {:11d} {:8.2f}%\n".format("- new:", corr_unseen, corr_unseen/unseen*100)
    print "{:>60} {:11d} {:8.2f}%\n".format("- new (new morphemes):", corr_unseen_m, corr_unseen_m/unseen_m*100)
    print "{:>60} {:11d} {:8.2f}%\n".format("- new (new combination):", corr_unseen_comb, corr_unseen_comb/unseen_new_comb*100)

    if file_out_errors:
        with codecs.open(file_out_errors,'w','utf-8') as f:
            f.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format("word","prediction", "gold", "err_freq", "ambigous?", "new?", "unique?", "lines(test)"))
            orderd_w = sorted(errors.keys(), key=lambda v: v[1], reverse=True)
            for (w,pred,true_pred) in orderd_w:
                w_new,w_unique,w_amb_tie,w_amb = False, False, False, False
                if w in amb_segm_test.keys():
                    w_amb = True
                    if w in amb_segm_test_tie.keys():
                        w_amb_tie = True
                else:
                    if w not in train_lexicon_w.keys():
                        w_new = True
                    else:
                        w_unique = True
                f.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(w, pred, true_pred, len(errors[(w,pred,true_pred)]), w_amb, w_new, w_unique, ", ".join(errors[(w,pred,true_pred)])))

def evaluate_ambiguity(trainin,gold,input_format,lowercase=False,file_out_errors =None, predict=None,file_out=None):
    train_lexicon_m = defaultdict(int)
    test_lexicon_m = defaultdict(int)
    train_lexicon_w = defaultdict(int)
    test_lexicon_w = defaultdict(int)
    train_dict = {} # train_word: dict(pos:dict(predict:freq))
    train_seg_dict = {} # train_word: dict(predict:freq)
    test_dict = {} # test_word: dict(pos:dict(predict:freq))
    
    input = input_format[0]
    pred = input_format[1]
    pos_col = input_format[2]
    
    # Read lexicon of training set
    trainin_f = codecs.open(trainin,'r','utf-8')
    for i,line in enumerate(trainin_f):
        #if i < 40:
        #        print line
        if len(line.strip()) != 0:
            line = line.strip()
            lineitems = line.split('\t')
            word = lineitems[input].lower() if lowercase else line.strip()
            segm = lineitems[pred].lower() if lowercase else line.strip()
            pos = lineitems[pos_col]
            train_lexicon_w[word] += 1
            morfs = lineitems[pred].split(' ')
            for m in morfs:
                train_lexicon_m[m] += 1
            
            if not word in train_seg_dict.keys():
                train_seg_dict[word] = {}
                train_seg_dict[word][segm]=1
            else:
                if segm not in train_seg_dict[word].keys():
                    train_seg_dict[word][segm]=1
                else:
                    train_seg_dict[word][segm]+=1
    
            if not word in train_dict.keys():
                train_dict[word] = {}
                train_dict[word][pos]={}
                train_dict[word][pos][segm] = 1
            else:
                if pos not in train_dict[word].keys():
                    train_dict[word][pos]={}
                    train_dict[word][pos][segm]=1
                else:
                    if segm not in train_dict[word][pos].keys():
                        train_dict[word][pos][segm]=1
                    else:
                        train_dict[word][pos][segm]+=1


    # Read lexicon of test set
    gold_f = codecs.open(gold,'r','utf-8')
    for i,line in enumerate(gold_f):
        #        if i < 40:
        if len(line.strip()) != 0:
            #            print line
            line = line.strip()
            lineitems = line.split('\t')
            word = lineitems[input].lower() if lowercase else line.strip()
            segm = lineitems[pred].lower() if lowercase else line.strip()
            pos = lineitems[pos_col]
            test_lexicon_w[word] += 1
            morfs = lineitems[pred].split(' ')
            for m in morfs:
                test_lexicon_m[m] += 1
            if not word in test_dict.keys():
                test_dict[word] = {}
                test_dict[word][pos]={}
                test_dict[word][pos][segm] = 1
            else:
                if pos not in test_dict[word].keys():
                    test_dict[word][pos]={}
                    test_dict[word][pos][segm]=1
                else:
                    if segm not in test_dict[word][pos].keys():
                        test_dict[word][pos][segm]=1
                    else:
                        test_dict[word][pos][segm]+=1
    if predict:
        # Collect predictions
        predict_f = codecs.open(predict,'r','utf-8')
        pred_dict_ext = {}
        for j, line in enumerate(predict_f):
            line = line.strip().lower() if lowercase else line.strip()
            w, w_segm = line.split('\t')
            pred_dict_ext[(w,j+1)] = w_segm
    

    amb_segm_test_candidates = {k:v for k,v in test_dict.items() if k in train_seg_dict.keys()} #the keys are test words seen in the train, the values are test frequencies - for statistics
    amb_segm_train = {k:train_dict[k] for k,v in amb_segm_test_candidates.items() if len(train_seg_dict[k])>1} # the values are train frequencies - for prediction
    amb_segm_test = {k:v for k,v in amb_segm_test_candidates.items() if len(train_seg_dict[k])>1} # the values are test frequencies - for statistics
    amb = 0
    for k,v in amb_segm_test.items():
        for pos,pos_v in v.items():
            amb += sum(pos_v.values())
    corr_amb = 0 # number of correct ambigous


    pos_disamb_test_candidates = {k:train_dict[k] for k,v in amb_segm_test.items()} # the keys are ambigous test words, the values are train frequencies - for candidates filtering based on train set
    pos_disamb_test = {} # the keys are ambigous test words, the values are {(w,pos) : {segs:freqs}}
    pos_disamb = 0
    pos_nodisamb_test = {}
    pos_nodisamb = 0
    pos_nodisamb_new_test = {}
    pos_nodisamb_new = 0
    
    
    pos_nodisamb_tie_test = {}
    pos_nodisamb_tie = 0
    pos_nodisamb_notie_test = {}
    pos_nodisamb_notie = 0
    
    for w, w_v in pos_disamb_test_candidates.items():
        train_pos = w_v.keys()
        u = [pos for pos,pos_v in w_v.items() if len(pos_v)>1]

        for pos,pos_v in test_dict[w].items():
            freq = sum(test_dict[w][pos].values())
            if pos not in train_pos:
                # cannot be disambiguated with pos - new (w,pos) pairs
                pos_nodisamb += freq
                pos_nodisamb_new += freq
                for pos_,pos_v_ in train_dict[w].items():
                    for seg, freq_ in pos_v_.items():
                        if (w,pos_) not in pos_nodisamb_test.keys():
                            pos_nodisamb_test[(w,pos_)] = {}
                            pos_nodisamb_test[(w,pos_)][seg] = freq_
                        else:
                            pos_nodisamb_test[(w,pos_)][seg] = freq_
                        if (w,pos_) not in pos_nodisamb_new_test.keys():
                            pos_nodisamb_new_test[(w,pos_)] = {}
                            pos_nodisamb_new_test[(w,pos_)][seg] = freq_
                        else:
                            pos_nodisamb_new_test[(w,pos_)][seg] = freq_
            else:
                # can be disambiguated with pos - unique (w,pos) pairs
                if pos not in u:
                    pos_disamb += freq
                    for seg, freq_ in train_dict[w][pos].items():
                        if (w,pos) not in pos_disamb_test.keys():
                            pos_disamb_test[(w,pos)] = {}
                            pos_disamb_test[(w,pos)][seg] = freq_
                        else:
                            pos_disamb_test[(w,pos)][seg] = freq_
                else:
                    # cannot be disambiguated with pos - not unique (w,pos) pairs
                    pos_nodisamb += freq
                    
                    # collect freq statistics based on the train set
                    f = train_dict[w][pos].values()
                    if f.count(f[0]) == len(f):
                        # tie
#                        print w, train_dict[w], test_dict[w], pos
                        pos_nodisamb_tie += freq
                        for seg, freq_ in train_dict[w][pos].items():
                            if (w,pos) not in pos_nodisamb_tie_test.keys():
                                pos_nodisamb_tie_test[(w,pos)] = {}
                                pos_nodisamb_tie_test[(w,pos)][seg] = freq_
                            else:
                                pos_nodisamb_tie_test[(w,pos)][seg] = freq_
                            
                            if (w,pos_) not in pos_nodisamb_test.keys():
                                pos_nodisamb_test[(w,pos)] = {}
                                pos_nodisamb_test[(w,pos)][seg] = freq_
                            else:
                                pos_nodisamb_test[(w,pos)][seg] = freq_
                    else:
                        # no tie
                        pos_nodisamb_notie += freq
                        for seg, freq_ in train_dict[w][pos].items():
                            if (w,pos) not in pos_nodisamb_notie_test.keys():
                                pos_nodisamb_notie_test[(w,pos)] = {}
                                pos_nodisamb_notie_test[(w,pos)][seg] = freq_
                            else:
                                pos_nodisamb_notie_test[(w,pos)][seg] = freq_

                            if (w,pos) not in pos_nodisamb_test.keys():
                                pos_nodisamb_test[(w,pos)] = {}
                                pos_nodisamb_test[(w,pos)][seg] = freq_
                            else:
                                pos_nodisamb_test[(w,pos)][seg] = freq_
    
    if predict:
        errors = {}
        corr = 0
        corr_amb = 0 # number of correct ambigous
        corr_pos_disamb = 0 # number of correct ambigous which can be disambiguated with pos
        corr_pos_nodisamb = 0 # number of correct ambigous which cannot be disambiguated with pos
        corr_pos_nodisamb_new = 0 # number of correct ambigous which cannot be disambiguated with pos - new
        corr_pos_nodisamb_tie = 0 # number of correct ambigous which cannot be disambiguated with pos - tie
        corr_pos_nodisamb_notie = 0 # number of correct ambigous which cannot be disambiguated with pos - noties

    allc = 0
    errors_b = {}
    corr_b = 0
    corr_amb_b = 0 # number of correct ambigous
    corr_pos_disamb_b = 0 # number of correct ambigous which can be disambiguated with pos
    corr_pos_nodisamb_b = 0 # number of correct ambigous which cannot be disambiguated with pos
    corr_pos_nodisamb_new_b = 0 # number of correct ambigous which cannot be disambiguated with pos - new
    corr_pos_nodisamb_tie_b = 0 # number of correct ambigous which cannot be disambiguated with pos - tie
    corr_pos_nodisamb_notie_b = 0 # number of correct ambigous which cannot be disambiguated with pos - noties

    gold_f.seek(0)
    for i,line in enumerate(gold_f):
        #if i < 5:
        if len(line.strip()) !=0:
            try:
                line = line.strip()
                lineitems = line.split('\t')
                w = lineitems[input].lower() if lowercase else line.strip()
                w_segm = lineitems[pred].lower() if lowercase else line.strip()
                pos = lineitems[pos_col]
                lines = str(i)
                w_segm_morfs = w_segm.split(' ')
                
            except:
                print i, line
            
            allc += 1
            
            # baseline
            if w in amb_segm_test.keys():
                # unique (w,pos)
                if (w,pos) in pos_disamb_test.keys():
                    if not len(train_dict[w][pos].keys()) == 1:
                        print pos, train_dict[w], test_dict[w]
                    seg = train_dict[w][pos].keys()[0]
                    w_baseline_pred = seg
                    if w_baseline_pred == w_segm:
                        corr_pos_disamb_b +=1
                        corr_amb_b +=1
                        corr_b +=1
                else:
                    # if pos is new, take the seg with highest frequency
                    if (w,pos) in pos_nodisamb_new_test.keys():
                        w_preds = {seg:freq for pos,pos_v in train_dict[w].items() for seg,freq in pos_v.items()}
                        w_baseline_pred = max(w_preds.keys(), key=lambda k: w_preds[k])
                        if w_baseline_pred == w_segm:
                            corr_pos_nodisamb_new_b +=1
                            corr_pos_nodisamb_b +=1
                            corr_amb_b +=1
                            corr_b +=1
                    #if pos in ties, choose randomely from the train segs corresponding to these pos
                    elif (w,pos) in pos_nodisamb_tie_test.keys():
                        w_preds = train_dict[w][pos]
                        w_baseline_pred = max(w_preds.keys(), key=lambda k: w_preds[k])
                        if w_baseline_pred == w_segm:
                            corr_pos_nodisamb_tie_b +=1
                            corr_pos_nodisamb_b +=1
                            corr_amb_b +=1
                            corr_b +=1
                    #if pos not in ties, choose from the train segs corresponding to these pos with highest frequency
                    elif (w,pos) in pos_nodisamb_notie_test.keys():
                        w_preds = train_dict[w][pos]
                        w_baseline_pred = max(w_preds.keys(), key=lambda k: w_preds[k])
                        if w_baseline_pred == w_segm:
                            corr_pos_nodisamb_notie_b +=1
                            corr_pos_nodisamb_b +=1
                            corr_amb_b +=1
                            corr_b +=1
            
            else:
                #new
                if w not in train_lexicon_w.keys():
                    w_baseline_pred = w
                    # new - old morphemes but new combination
                    if all(m in train_lexicon_m.keys() for m in w_segm_morfs):
                        if w_baseline_pred == w_segm:
                            corr_b +=1
                    else:
                        # new - new morphemes
                        if w_baseline_pred == w_segm:
                            corr_b +=1
                #seen and unique
                else:
                    w_baseline_pred = train_seg_dict[w].keys()[0]
                    if w_baseline_pred == w_segm:
                        corr_b +=1

            if not w_baseline_pred == w_segm:
                if (w,w_baseline_pred, w_segm, pos) not in errors_b.keys():
                    errors_b[(w,w_baseline_pred, w_segm, pos)] = [lines]
                else:
                    errors_b[(w,w_baseline_pred, w_segm, pos)].append(lines)
        
            if predict:
                # system predictions
                
                if pred_dict_ext[(w,allc)] == w_segm:
                    corr += 1
            
                    if w in amb_segm_test.keys():
                        corr_amb += 1
                        if (w,pos) in pos_disamb_test.keys():
                            corr_pos_disamb +=1
                        else:
                            corr_pos_nodisamb +=1
                            if (w,pos) in pos_nodisamb_new_test.keys():
                                corr_pos_nodisamb_new += 1
                            else:
                                if (w,pos) in pos_nodisamb_tie_test.keys():
                                    corr_pos_nodisamb_tie +=1
                                else:
                                    corr_pos_nodisamb_notie +=1

                else:
        
                    if (w,pred_dict_ext[(w,allc)], w_segm, pos) not in errors.keys():
                        errors[(w,pred_dict_ext[(w,allc)], w_segm, pos)] = [lines]
                    else:
                        errors[(w,pred_dict_ext[(w,allc)], w_segm, pos)].append(lines)

    if predict:
        with codecs.open(file_out,'w','utf-8') as f:
            f.write("\nDATA:\n")
            f.write("{:<50} {:21d} {:8.2f}%\n".format("# of ambigous source word tokens:", amb, amb/sum(test_lexicon_w.values())*100))
            f.write("{:<50} {:21d} {:8.2f}%\n".format(" - can be POS disambiguated:", pos_disamb, pos_disamb/amb*100))
            f.write("{:<50} {:21d} {:8.2f}%\n".format(" - cannot be POS disambiguated:", pos_nodisamb, pos_nodisamb/amb*100))
            f.write("{:<50} {:21d} {:8.2f}%\n".format(" - cannot be POS disambiguated - new POS:", pos_nodisamb_new, pos_nodisamb_new/amb*100))
            f.write("{:<50} {:21d} {:8.2f}%\n".format(" - cannot be POS disambiguated - seen POS but ties:", pos_nodisamb_tie, pos_nodisamb_tie/amb*100))
            f.write("{:<50} {:21d} {:8.2f}%\n".format(" - cannot be POS disambiguated - seen POS, no ties:", pos_nodisamb_notie, pos_nodisamb_notie/amb*100))

            f.write("\nPERFORMANCE:\n")
            f.write("{:>60} {:11d}\n".format("Number of predictions total:", allc))
            f.write("{:>60} {:11d} {:8.2f}%\n".format("Number of correct predictions total:", corr, corr/allc*100))
            f.write("{:>60} {:11d} {:8.2f}%\n".format("- ambigous:", corr_amb, corr_amb/amb*100))
            f.write("{:>60} {:11d} {:8.2f}%\n".format("- ambigous with POS disambig:", corr_pos_disamb, corr_pos_disamb/pos_disamb*100))
            f.write("{:>60} {:11d} {:8.2f}%\n".format("- ambigous with no POS disambig:", corr_pos_nodisamb, corr_pos_nodisamb/pos_nodisamb*100))
            f.write("{:>60} {:11d} {:8.2f}%\n".format("- ambigous with no POS disambig - new POS:", corr_pos_nodisamb_new, corr_pos_nodisamb_new/pos_nodisamb_tie*100))
            f.write("{:>60} {:11d} {:8.2f}%\n".format("- ambigous with no POS disambig - ties:", corr_pos_nodisamb_tie, corr_pos_nodisamb_tie/pos_nodisamb_tie*100))
            f.write("{:>60} {:11d} {:8.2f}%\n".format("- ambigous with no POS disambig - no ties:", corr_pos_nodisamb_notie, corr_pos_nodisamb_notie/pos_nodisamb_notie*100))
            
            with codecs.open(file_out_errors,'w','utf-8') as f:
                #f.write("\n\nERRORS:\n")
                f.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format("word","prediction", "gold", "err_freq", "pos", "ambigous?", "can be POS disamb?", "tie?", "lines(test)"))
                orderd_w = sorted(errors.keys(), key=lambda v: v[1], reverse=True)
                for (w,pred,true_pred,pos) in orderd_w:
                    amb_type, pos_disamb_type,w_amb_tie =  False, False, False
                    amb_type = w in amb_segm_test.keys()
                    pos_disamb_type = (w,pos) in pos_disamb_test.keys()
                    if not pos_disamb_type:
                        if (w,pos) in pos_nodisamb_tie_test.keys():
                            w_amb_tie = True
                    f.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(w, pred, true_pred, len(errors[(w,pred,true_pred,pos)]), pos, amb_type, pos_disamb_type, w_amb_tie,  ", ".join(errors[(w,pred,true_pred,pos)])))

    else:
        print "\nDATA:\n"
        print "{:<50} {:21d} {:8.2f}%\n".format("# of ambigous source word tokens:", amb, amb/sum(test_lexicon_w.values())*100)
        print "{:<50} {:21d} {:8.2f}%\n".format(" - can be POS disambiguated:", pos_disamb, pos_disamb/amb*100)
        print "{:<50} {:21d} {:8.2f}%\n".format(" - cannot be POS disambiguated:", pos_nodisamb, pos_nodisamb/amb*100)
        print "{:<50} {:21d} {:8.2f}%\n".format(" - cannot be POS disambiguated - new POS:", pos_nodisamb_new, pos_nodisamb_new/amb*100)
        print "{:<50} {:21d} {:8.2f}%\n".format(" - cannot be POS disambiguated - seen POS but ties:", pos_nodisamb_tie, pos_nodisamb_tie/amb*100)
        print "{:<50} {:21d} {:8.2f}%\n".format(" - cannot be POS disambiguated - seen POS, no ties:", pos_nodisamb_notie, pos_nodisamb_notie/amb*100)

        print "\nPERFORMANCE - BASELINE:\n"
        print "{:>60} {:11d}\n".format("Number of predictions total:", allc)
        print "{:>60} {:11d} {:8.2f}%\n".format("Number of correct predictions total:", corr_b, corr_b/allc*100)
        print "{:>60} {:11d} {:8.2f}%\n".format("- ambigous:", corr_amb_b, corr_amb_b/amb*100)
        print "{:>60} {:11d} {:8.2f}%\n".format("- ambigous with POS disambig:", corr_pos_disamb_b, corr_pos_disamb_b/pos_disamb*100)
        print "{:>60} {:11d} {:8.2f}%\n".format("- ambigous with no POS disambig:", corr_pos_nodisamb_b, corr_pos_nodisamb_b/pos_nodisamb*100)
        print "{:>60} {:11d} {:8.2f}%\n".format("- ambigous with no POS disambig - new POS:", corr_pos_nodisamb_new_b, corr_pos_nodisamb_new_b/pos_nodisamb_tie*100)
        print "{:>60} {:11d} {:8.2f}%\n".format("- ambigous with no POS disambig - ties:", corr_pos_nodisamb_tie_b, corr_pos_nodisamb_tie_b/pos_nodisamb_tie*100)
        print "{:>60} {:11d} {:8.2f}%\n".format("- ambigous with no POS disambig - no ties:", corr_pos_nodisamb_notie_b, corr_pos_nodisamb_notie_b/pos_nodisamb_notie*100)

        if file_out_errors:
            with codecs.open(file_out_errors,'w','utf-8') as f:
                #f.write("\n\nERRORS:\n")
                f.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format("word","prediction", "gold", "err_freq", "pos", "ambigous?", "can be POS disamb?", "tie?", "lines(test)"))
                orderd_w = sorted(errors_b.keys(), key=lambda v: v[1], reverse=True)
                for (w,pred,true_pred,pos) in orderd_w:
                    amb_type, pos_disamb_type,w_amb_tie =  False, False, False
                    amb_type = w in amb_segm_test.keys()
                    pos_disamb_type = (w,pos) in pos_disamb_test.keys()
                    if not pos_disamb_type:
                        if (w,pos) in pos_nodisamb_tie_test.keys():
                            w_amb_tie = True
                    f.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(w, pred, true_pred, len(errors_b[(w,pred,true_pred,pos)]), pos, amb_type, pos_disamb_type, w_amb_tie,  ", ".join(errors_b[(w,pred,true_pred,pos)])))


def evaluate(trainin,gold,predict,file_out,file_out_errors,input_format,lowercase=False, ext_trainin=None):
    train_lexicon_m = defaultdict(int)
    test_lexicon_m = defaultdict(int)
    train_lexicon_w = defaultdict(int)
    test_lexicon_w = defaultdict(int)
    test_dict = {} # test_word: dict(predict:freq)
    train_dict = {} # train_word: dict(predict:freq)

    input = input_format[0]
    pred = input_format[1]
                 
    # Read lexicon of training set
    trainin_f = codecs.open(trainin,'r','utf-8')
    for i,line in enumerate(trainin_f):
        #if i < 40:
#        print line
        if len(line.strip()) != 0:
            line = line.strip().lower() if lowercase else line.strip()
            lineitems = line.split('\t')
            word = lineitems[input]
            segm = lineitems[pred]
            train_lexicon_w[word] += 1
            morfs = lineitems[pred].split(' ')
            for m in morfs:
                train_lexicon_m[m] += 1
            if not word in train_dict.keys():
                train_dict[word] = {}
                train_dict[word][segm]=1
            else:
                if segm not in train_dict[word].keys():
                    train_dict[word][segm]=1
                else:
                    train_dict[word][segm]+=1

    if ext_trainin: # extra train data for LM training, one-colum
        trainin_ext_f = codecs.open(ext_trainin,'r','utf-8')
        for i,line in enumerate(trainin_ext_f):
            if len(line.strip()) != 0:
                line = line.strip().lower() if lowercase else line.strip()
                morfs = line.split(' ')
                for m in morfs:
                    train_lexicon_m[m] += 1
    

    # Read lexicon of test set
    gold_f = codecs.open(gold,'r','utf-8')
    for i,line in enumerate(gold_f):
#        if i < 40:
        if len(line.strip()) != 0:
#            print line
            line = line.strip().lower() if lowercase else line.strip()
            lineitems = line.split('\t')
            word = lineitems[input]
            segm = lineitems[pred]
            test_lexicon_w[word] += 1
            morfs = lineitems[pred].split(' ')
            for m in morfs:
                test_lexicon_m[m] += 1
            if not word in test_dict.keys():
                test_dict[word] = {}
                test_dict[word][segm]=1
            else:
                if segm not in test_dict[word].keys():
                    test_dict[word][segm]=1
                else:
                    test_dict[word][segm]+=1

    errors = {}

    # Collect predictions
    predict_f = codecs.open(predict,'r','utf-8')
    pred_dict_ext = {}
    for j, line in enumerate(predict_f):
        line = line.strip().lower() if lowercase else line.strip()
        w, w_segm = line.split('\t')
        pred_dict_ext[(w,j+1)] = w_segm

    #LM Evaluation
    allc = 0 # total number of predictions  (that is the number of words in the input (gold))
    corr = 0 # total number of correct predictions
    
    amb_segm_test_candidates = {k:v for k,v in test_dict.items() if k in train_dict.keys()} #the values are test frequencies - for statistics
    amb_segm_train = {k:train_dict[k] for k,v in amb_segm_test_candidates.items() if len(train_dict[k])>1} # the values are train frequencies - for prediction
    amb_segm_test = {k:v for k,v in amb_segm_test_candidates.items() if len(train_dict[k])>1} # the values are test frequencies - for statistics
    #    print amb_segm_train.items()[:10]
    amb_segm_test_freq = {k:sum(v.values()) for k,v in amb_segm_test.items()}  # the values are test frequencies - for statistics
    amb = sum(amb_segm_test_freq.values())
    corr_amb = 0 # number of correct ambigous
    
    amb_segm_test_tie_candidates = {k:v.values() for k,v in amb_segm_train.items()} # the values are train frequencies - for tie candidates filtering based on train set
    amb_segm_test_tie_check = {k:v for k,v in amb_segm_test_tie_candidates.items() if v.count(v[0]) == len(v)} # the values are train frequencies - for printing
    #    print amb_segm_test_tie_check.items()
    amb_segm_test_tie = {k:sum(test_dict[k].values()) for k,v in amb_segm_test_tie_candidates.items() if v.count(v[0]) == len(v)}   # the values are test frequencies - for statistics
    amb_tie = sum(amb_segm_test_tie.values())
    corr_amb_tie = 0 # number of correct ambigous with tie
    amb_notie = amb - amb_tie
    corr_amb_notie = 0 # number of correct ambigous with tie
    
    
    
    not_amb_segm_test = {k:v for k,v in test_dict.items() if k not in amb_segm_test.keys()}
    seen_freq = {k:sum(v.values()) for k,v in not_amb_segm_test.items() if k in train_lexicon_w.keys()}
    seen = sum(seen_freq.values())
    corr_seen = 0 # number of correct seen words
    
    unseen_freq = {k:sum(v.values()) for k,v in not_amb_segm_test.items() if not k in train_lexicon_w.keys()}
    unseen = sum(unseen_freq.values())
    corr_unseen = 0
    
    unseen_m_freq = {k:sum(v.values()) for k,v in not_amb_segm_test.items() if ( not k in train_lexicon_w.keys() and not all(m in train_lexicon_m.keys() for m in v.keys()[0].split(' ')) )}
    unseen_m = sum(unseen_m_freq.values())
    corr_unseen_m = 0 # number of correct unseen words - new morphs
    
    unseen_new_comb = unseen - unseen_m
    corr_unseen_comb = 0 # number of correct unseen words - new combinations

    gold_f.seek(0)
    for i,line in enumerate(gold_f):
        #if i < 5:
        if len(line.strip()) !=0:
            try:
                line = line.strip().lower() if lowercase else line.strip()
                lineitems = line.split('\t')
                w = lineitems[input]
                w_segm = lineitems[pred]
                lines = str(i)
                w_segm_morfs = w_segm.split(' ')
            
            except:
                print i, line
            
#                # remove diacritic
#                if unicodedata.combining(w[0]):
#                    w = w[1:]

            allc += 1
            if pred_dict_ext[(w,allc)] == w_segm:
                corr += 1
                
                if w in amb_segm_test.keys():
                    corr_amb += 1
                    if w in amb_segm_test_tie.keys():
                        corr_amb_tie +=1
                    else:
                        corr_amb_notie +=1
                else:
                    if w not in train_lexicon_w.keys():
                        corr_unseen += 1
                        if all(m in train_lexicon_m.keys() for m in w_segm_morfs):
                            corr_unseen_comb += 1
                        else:
                            corr_unseen_m += 1
                    else:
                        corr_seen += 1
            else:
                if (w,pred_dict_ext[(w,allc)], w_segm) not in errors.keys():
                    errors[(w,pred_dict_ext[(w,allc)], w_segm)] = [lines]
                else:
                    errors[(w,pred_dict_ext[(w,allc)], w_segm)].append(lines)
                    
                    
    with codecs.open(file_out,'w','utf-8') as f:
        
        # Print statistics

        total_w_pred = sum(test_lexicon_w.values())
        f.write("{:<50} {:>10} {:>10}\n".format("DATA","TRAIN", "TEST"))
        f.write("{:<50} {:12d} {:8d}\n".format("# of target segment tokens:", sum(train_lexicon_m.values()), sum(test_lexicon_m.values())))
        f.write("{:<50} {:12d} {:8d}\n".format("# of source word tokens:", sum(train_lexicon_w.values()), total_w_pred))
        f.write("{:<50} {:21d} {:8.2f}%\n".format("# of ambigous source word tokens:", amb, amb/total_w_pred*100))
        f.write("{:<50} {:21d} {:8.2f}%\n".format("# of ambigous source word tokens - ties:", amb_tie, amb_tie/total_w_pred*100))
        f.write("{:<50} {:21d} {:8.2f}%\n".format("# of ambigous source word tokens - no ties:", amb_notie, amb_notie/total_w_pred*100))
        f.write("{:<50} {:21d} {:8.2f}%\n".format("# of unique source word tokens:", seen, seen/total_w_pred*100))
        f.write("{:<50} {:21d} {:8.2f}%\n".format("# of new source word tokens:", unseen, unseen/total_w_pred*100))
        f.write("{:<50} {:21d} {:8.2f}%\n".format("# of new source word tokens - new target segments:", unseen_m, unseen_m/total_w_pred*100))
        f.write("{:<50} {:21d} {:8.2f}%\n".format("# of new source word tokens - new combination:", unseen_new_comb, unseen_new_comb/total_w_pred*100))
    
        f.write("\nPERFORMANCE:\n")
        f.write("{:>60} {:11d}\n".format("Number of predictions total:", allc))
        f.write("{:>60} {:11d} {:8.2f}%\n".format("Number of correct predictions total:", corr, corr/allc*100))
        if amb !=0:
            f.write("{:>60} {:11d} {:8.2f}%\n".format("- ambigous:", corr_amb, corr_amb/amb*100))
        if amb_tie !=0:
            f.write("{:>60} {:11d} {:8.2f}%\n".format("- ambigous(ties):", corr_amb_tie, corr_amb_tie/amb_tie*100))
        if amb_notie !=0:
            f.write("{:>60} {:11d} {:8.2f}%\n".format("- ambigous(no ties):", corr_amb_notie, corr_amb_notie/amb_notie*100))
        if seen !=0:
            f.write("{:>60} {:11d} {:8.2f}%\n".format("- unique:", corr_seen, corr_seen/seen*100))
        f.write("{:>60} {:11d} {:8.2f}%\n".format("- new:", corr_unseen, corr_unseen/unseen*100))
        f.write("{:>60} {:11d} {:8.2f}%\n".format("- new (new morphemes):", corr_unseen_m, corr_unseen_m/unseen_m*100))
        f.write("{:>60} {:11d} {:8.2f}%\n".format("- new (new combination):", corr_unseen_comb, corr_unseen_comb/unseen_new_comb*100))

    with codecs.open(file_out_errors,'w','utf-8') as f:
        #f.write("\n\nERRORS:\n")
        f.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format("word","prediction", "gold", "err_freq", "ambigous?", "new?", "unique?", "new morphemes?", "lines(test)"))
        orderd_w = sorted(errors.keys(), key=lambda v: v[1], reverse=True)
        for (w,pred,true_pred) in orderd_w:
#            seen_w = w in train_lexicon_w.keys()
#            seen_w, new_m = 'NA','NA'
            w_new,w_unique,w_amb,w_new_m = False, False, False,False
            amb_type = w in amb_segm_test.keys()
            if not amb_type:
                if w not in train_lexicon_w.keys():
                    w_new = True
                    w_new_m = not all(m in train_lexicon_m.keys() for m in true_pred.split(' '))
                else:
                    w_unique = True
            f.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(w, pred, true_pred, len(errors[(w,pred,true_pred)]), amb_type, w_new, w_unique, w_new_m, ", ".join(errors[(w,pred,true_pred)])))

if __name__ == "__main__":
    arguments = docopt(__doc__)
    print arguments
    
    trainin = arguments['TRAIN_DATA']
    gold = arguments['TEST_DATA']
    predict = arguments['PREDICTIONS']
    file_out = arguments['RESULTS_FILE']
    file_out_errors = arguments['RESULTS_ERRORS_FILE']
    input_format_arg = arguments['--input_format']
    input_format=[int(col) for col in input_format_arg.split(',')]
    
    if arguments['eval']:
        evaluate(trainin,gold,predict,file_out,file_out_errors, input_format,arguments['--lowercase'],arguments['--extended_train_data'])
    elif arguments['eval_baseline']:
        evaluate_baseline(trainin,gold, input_format,arguments['--lowercase'],arguments['--error_file'])
    elif arguments['eval_ambiguity']:
        evaluate_ambiguity(trainin,gold, input_format,arguments['--lowercase'],file_out_errors,predict,file_out)
    elif arguments['eval_ambiguity_baseline']:
        evaluate_ambiguity(trainin,gold, input_format,arguments['--lowercase'],arguments['--error_file'])
    else:
        print "Unknown option"
