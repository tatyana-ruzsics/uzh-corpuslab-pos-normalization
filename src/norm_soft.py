#!/usr/bin/env python
# -*- coding: utf-8 -*
"""Trains encoder-decoder model with soft attention.

Usage:
  norm_soft.py train [--dynet-seed SEED] [--dynet-mem MEM] [--input_format=INPUT_FORMAT]  [--lowercase=LOW]
    [--input=INPUT] [--hidden=HIDDEN] [--layers=LAYERS] [--vocab_path=VOCAB_PATH]
    [--dropout=DROPOUT] [--epochs=EPOCHS] [--patience=PATIENCE] [--optimization=OPTIMIZATION]
    MODEL_FOLDER --train_path=TRAIN_FILE --dev_path=DEV_FILE
  norm_soft.py test [--dynet-mem MEM] [--beam=BEAM] [--pred_path=PRED_FILE] [--input_format=INPUT_FORMAT]
    MODEL_FOLDER --test_path=TEST_FILE [--lowercase=LOW]
  norm_soft.py ensemble_test [--dynet-mem MEM] [--beam=BEAM] [--pred_path=PRED_FILE] [--input_format=INPUT_FORMAT]
    ED_MODEL_FOLDER MODEL_FOLDER --test_path=TEST_FILE [--lowercase=LOW]
    

Arguments:
MODEL_FOLDER  save/read model folder where also eval results are written to, possibly relative to RESULTS_FOLDER
ED_MODEL_FOLDER  ED model(s) folder, possibly relative to RESULTS_FOLDER, coma-separated

Options:
  -h --help                     show this help message and exit
  --dynet-seed SEED             DyNET seed
  --dynet-mem MEM               allocates MEM bytes for DyNET [default: 500]
  --input=INPUT                 input vector dimensions [default: 100]
  --hidden=HIDDEN               hidden layer dimensions [default: 200]
  --layers=LAYERS               amount of layers in LSTMs  [default: 1]
  --dropout=DROPOUT             amount of dropout in LSTMs [default: 0]
  --epochs=EPOCHS               number of training epochs   [default: 30]
  --patience=PATIENCE           patience for early stopping [default: 10]
  --optimization=OPTIMIZATION   chosen optimization method ADAM/SGD/ADAGRAD/MOMENTUM/ADADELTA [default: SGD]
  --train_path=TRAIN_FILE       train set path, possibly relative to DATA_FOLDER, only for training
  --dev_path=DEV_FILE           dev set path, possibly relative to DATA_FOLDER, only for training
  --test_path=TEST_FILE         test set path, possibly relative to DATA_FOLDER, only for evaluation
  --vocab_path=VOCAB_PATH       vocab path, possibly relative to RESULTS_FOLDER [default: vocab.txt]
  --beam=BEAM                   beam width [default: 1]
  --pred_path=PRED_FILE         name for predictions file in the test mode [default: 'best.test']
  --input_format=INPUT_FORMAT   coma-separated list of input, output columns [default: 0,1]
  --lowercase=LOW               use lowercased data [default: True]
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

import time
import copy
import dynet as dy
import numpy as np
import os

from common import BEGIN_CHAR,STOP_CHAR,UNK_CHAR, SRC_FOLDER,RESULTS_FOLDER,DATA_FOLDER,check_path, write_pred_file, write_param_file, write_eval_file
from vocab_builder import build_vocabulary, Vocab

MAX_PRED_SEQ_LEN = 50 # option
OPTIMIZERS = {'ADAM'    : lambda m: dy.AdamTrainer(m, lam=0.0, alpha=0.0001, #common
                                                   beta_1=0.9, beta_2=0.999, eps=1e-8),
    'SGD'     : dy.SimpleSGDTrainer,
        'ADADELTA': dy.AdadeltaTrainer}


### IO handling and evaluation

def load_data(filename, input_format, lowercase=False):
    """ Load data from file
        
        filename (str):   file containing input/output data, structure (tab-separated):
        input    output
        return tuple (output, input) where each element is a list
        where each element in the list is one example
        """
    
    print 'loading data from file:', filename
    
    input_col, output_col = input_format
    inputs, outputs = [], []
    
    with codecs.open(filename, encoding='utf8') as f:
        for i,line in enumerate(f):
            if not len(line.strip())==0:
                try:
                    splt = line.strip().split('\t')
                    inputs.append(splt[input_col].lower() if lowercase else splt[input_col])
                    outputs.append(splt[output_col].lower() if lowercase else splt[output_col])
                except:
                    print "bad line: {}, {}".format(i,line)

    tup = (inputs, outputs)
    print 'found', len(outputs), 'examples'
    return tup

def log_to_file(log_file_name, e, avg_train_loss, train_accuracy, dev_accuracy):
    # if first write, add headers
    if e == 0:
        log_to_file(log_file_name, 'epoch', 'avg_train_loss', 'train_accuracy', 'dev_accuracy')
    
    with open(log_file_name, "a") as logfile:
        logfile.write("{}\t{}\t{}\t{}\n".format(e, avg_train_loss, train_accuracy, dev_accuracy))

# class to handle data
class SoftDataSet(object):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.dataset = self.inputs, self.outputs
        self.length = len(self.outputs)
    
    def iter(self, indices=None, shuffle=False):
        zipped = zip(*self.dataset)
        if indices or shuffle:
            if not indices:
                indices = range(self.length)
            elif isinstance(indices, int):
                indices = range(indices)
            else:
                assert isinstance(indices, (list, tuple))
            if shuffle:
                random.shuffle(indices)
            zipped = [zipped[i] for i in indices]
        return zipped
    
    @classmethod
    def from_file(cls, path, input_format, *args, **kwargs):
        # returns a `SoftDataSet` with fields: inputs, outputs
        inputs, outputs = load_data(path, input_format, *args, **kwargs)
        return cls(inputs, outputs)

class SoftAttention(object):
    def __init__(self, pc, model_hyperparams, best_model_path=None):
        
        self.hyperparams = model_hyperparams
        
        print 'Loading vocabulary from {}:'.format(self.hyperparams['VOCAB_PATH'])
        self.vocab = Vocab.from_file(self.hyperparams['VOCAB_PATH'])
        self.BEGIN   = self.vocab.w2i[BEGIN_CHAR]
        self.STOP   = self.vocab.w2i[STOP_CHAR]
        self.UNK       = self.vocab.w2i[UNK_CHAR]
        self.hyperparams['VOCAB_SIZE'] = self.vocab.size()
        
        self.build_model(pc, best_model_path)
        
        print 'Model Hypoparameters:'
        for k, v in self.hyperparams.items():
            print '{:20} = {}'.format(k, v)
        print
        
    def build_model(self, pc, best_model_path):
        
        if best_model_path:
            print 'Loading model from: {}'.format(best_model_path)
            self.fbuffRNN, self.bbuffRNN, self.VOCAB_LOOKUP, self.decoder, self.R, self.bias, self.W_c, self.W__a, self.U__a,  self.v__a = dy.load(best_model_path, pc)
        else:
            # BiLSTM for input
            self.fbuffRNN  = dy.CoupledLSTMBuilder(self.hyperparams['LAYERS'], self.hyperparams['INPUT_DIM'], self.hyperparams['HIDDEN_DIM'], pc)
            self.bbuffRNN  = dy.CoupledLSTMBuilder(self.hyperparams['LAYERS'], self.hyperparams['INPUT_DIM'], self.hyperparams['HIDDEN_DIM'], pc)
            
            # embedding lookups for vocabulary
            self.VOCAB_LOOKUP  = pc.add_lookup_parameters((self.hyperparams['VOCAB_SIZE'], self.hyperparams['INPUT_DIM']))

            # decoder LSTM
            self.decoder = dy.CoupledLSTMBuilder(self.hyperparams['LAYERS'], self.hyperparams['INPUT_DIM'], self.hyperparams['HIDDEN_DIM'], pc)

            # softmax parameters
            self.R = pc.add_parameters((self.hyperparams['VOCAB_SIZE'], 3 * self.hyperparams['HIDDEN_DIM']))
            self.bias = pc.add_parameters(self.hyperparams['VOCAB_SIZE'])
            
            # attention MLPs - Loung-style with extra v_a from Bahdanau
            
            # concatenation layer for h (hidden dim), c (2 * hidden_dim)
            self.W_c = pc.add_parameters((3 * self.hyperparams['HIDDEN_DIM'], 3 * self.hyperparams['HIDDEN_DIM']))
            
            # attention MLP's - Bahdanau-style
            # concatenation layer for h_input (2*hidden_dim), h_output (hidden_dim)
            self.W__a = pc.add_parameters((self.hyperparams['HIDDEN_DIM'], self.hyperparams['HIDDEN_DIM']))
            
            # concatenation layer for h (hidden dim), c (2 * hidden_dim)
            self.U__a = pc.add_parameters((self.hyperparams['HIDDEN_DIM'], 2 * self.hyperparams['HIDDEN_DIM']))
            
            # concatenation layer for h_input (2*hidden_dim), h_output (hidden_dim)
            self.v__a = pc.add_parameters((1, self.hyperparams['HIDDEN_DIM']))
        
        
        print 'Model dimensions:'
        print ' * VOCABULARY EMBEDDING LAYER: IN-DIM: {}, OUT-DIM: {}'.format(self.hyperparams['VOCAB_SIZE'], self.hyperparams['INPUT_DIM'])
        print
        print ' * ENCODER biLSTM: IN-DIM: {}, OUT-DIM: {}'.format(2*self.hyperparams['INPUT_DIM'], 2*self.hyperparams['HIDDEN_DIM'])
        print ' * DECODER LSTM: IN-DIM: {}, OUT-DIM: {}'.format(self.hyperparams['INPUT_DIM'], self.hyperparams['HIDDEN_DIM'])
        print ' All LSTMs have {} layer(s)'.format(self.hyperparams['LAYERS'])
        print
        print ' * SOFTMAX: IN-DIM: {}, OUT-DIM: {}'.format(self.hyperparams['HIDDEN_DIM'], self.hyperparams['VOCAB_SIZE'])
        print

    def save_model(self, best_model_path):
        dy.save(best_model_path, [self.fbuffRNN, self.bbuffRNN, self.VOCAB_LOOKUP, self.decoder, self.R, self.bias, self.W_c, self.W__a, self.U__a,  self.v__a])


    def bilstm_transduce(self, encoder_frnn, encoder_rrnn, input_char_vecs):
        
        # BiLSTM forward pass
        s_0 = encoder_frnn.initial_state()
        s = s_0
        frnn_outputs = []
        for c in input_char_vecs:
            s = s.add_input(c)
            frnn_outputs.append(s.output())
        
        # BiLSTM backward pass
        s_0 = encoder_rrnn.initial_state()
        s = s_0
        rrnn_outputs = []
        for c in reversed(input_char_vecs):
            s = s.add_input(c)
            rrnn_outputs.append(s.output())
        
        # BiLTSM outputs
        blstm_outputs = []
        for i in xrange(len(input_char_vecs)):
            blstm_outputs.append(dy.concatenate([frnn_outputs[i], rrnn_outputs[len(input_char_vecs) - i - 1]]))
        
        return blstm_outputs

    def transduce(self, input, _true_output=None, feats=None):
        
        # convert _true_output string to list of vocabulary indeces
        if _true_output:
            try:
                true_output = [self.vocab.w2i[a] for a in _true_output]
            except:
                print a
                print _true_output
            true_output += [self.STOP]
            true_output = list(reversed(true_output))
        
        R = dy.parameter(self.R)   # hidden to vocabulary
        bias = dy.parameter(self.bias)
        W_c = dy.parameter(self.W_c)
        W__a = dy.parameter(self.W__a)
        U__a = dy.parameter(self.U__a)
        v__a = dy.parameter(self.v__a)
        
        
        # biLSTM encoder of input string
        input = [BEGIN_CHAR] + [c for c in input] + [STOP_CHAR]

        input_emb = []
        for char_ in reversed(input):
            char_id = self.vocab.w2i.get(char_, self.UNK)
            char_embedding = self.VOCAB_LOOKUP[char_id]
            input_emb.append(char_embedding)
        biencoder = self.bilstm_transduce(self.fbuffRNN, self.bbuffRNN, input_emb)
        
        losses = []
        output = []
        pred_history = [self.BEGIN] # <
        s = self.decoder.initial_state()
        
        while not len(pred_history) == MAX_PRED_SEQ_LEN:
            # compute probability over vocabulary and choose a prediction
            # either from the true prediction at train time or based on the model at test time
            
            # decoder next state
            prev_pred_id = pred_history[-1]
            s = s.add_input(self.VOCAB_LOOKUP[prev_pred_id])
            
            # soft attention vector
            scores = [v__a * dy.tanh(W__a * s.output() + U__a * h_input) for h_input in biencoder]
            alphas = dy.softmax(dy.concatenate(scores))
            c = dy.esum([h_input * dy.pick(alphas, j) for j, h_input in enumerate(biencoder)])
            
            # softmax over vocabulary
            h_output = dy.tanh(W_c * dy.concatenate([s.output(), c]))
            probs = dy.softmax(R * h_output + bias)

            if _true_output is None:
                pred_id = np.argmax(probs.npvalue())
            else:
                pred_id = true_output.pop()
                
            losses.append(-dy.log(dy.pick(probs, pred_id)))
            pred_history.append(pred_id)
            
            if pred_id == self.STOP:
                break
            else:
                pred_char = self.vocab.i2w.get(pred_id,UNK_CHAR)
                output.append(pred_char)

        output = u''.join(output)
        return ((dy.average(losses) if losses else None), output)


    def evaluate(self, data, beam):
        # data is a list of tuples (an instance of SoftDataSet with iter method applied)
        correct = 0.
        final_results = []
        for i,(input,output) in enumerate(data):
            dy.renew_cg()
            predictions = self.predict(input, beam)
            prediction = predictions[0][1]
#            print i, input, predictions
            if prediction == output:
                correct += 1
#            else:
#                print u'{}, input: {}, pred: {}, true: {}'.format(i, input, prediction, output)
#                print predictions
            final_results.append((input,prediction))  # pred expected as list
        accuracy = correct / len(data)
        return accuracy, final_results


    def param_init(self, input): #initialize parameters for current cg with the current input
    
        R = dy.parameter(self.R)   # from parameters to expressions
        bias = dy.parameter(self.bias)
        W_c = dy.parameter(self.W_c)
        W__a = dy.parameter(self.W__a)
        U__a = dy.parameter(self.U__a)
        v__a = dy.parameter(self.v__a)
        
        self.cg_params = (R, bias, W_c, W__a, U__a, v__a) # params for current cg and input
    
        # biLSTM encoder of input string
        input = [BEGIN_CHAR] + [c for c in input] + [STOP_CHAR]
        
        input_emb = []
        for char_ in reversed(input):
            char_id = self.vocab.w2i.get(char_, self.UNK)
            char_embedding = self.VOCAB_LOOKUP[char_id]
            input_emb.append(char_embedding)
        self.biencoder = self.bilstm_transduce(self.fbuffRNN, self.bbuffRNN, input_emb)
    
#        losses = []
#        output = []
#        pred_history = [self.BEGIN] # <
        self.s = self.decoder.initial_state()
        self.s = self.s.add_input(self.VOCAB_LOOKUP[self.BEGIN])

    def predict_next(self, scores=False, hidden =False):
        (R, bias, W_c, W__a, U__a, v__a) = self.cg_params

        # soft attention vector
        att_scores = [v__a * dy.tanh(W__a * self.s.output() + U__a * h_input) for h_input in self.biencoder]
        alphas = dy.softmax(dy.concatenate(att_scores))
        c = dy.esum([h_input * dy.pick(alphas, j) for j, h_input in enumerate(self.biencoder)])
            
        # softmax over vocabulary
        h_output = dy.tanh(W_c * dy.concatenate([self.s.output(), c]))
        if not hidden:
            if not scores:
                return dy.softmax(R * h_output + bias)
            else:
                return R * h_output + bias
        else:
            return h_output

    def predict_next_(self, state, scores=False, hidden=False):
        (R, bias, W_c, W__a, U__a, v__a) = self.cg_params
        
        # soft attention vector
        att_scores = [v__a * dy.tanh(W__a * state.output() + U__a * h_input) for h_input in self.biencoder]
        alphas = dy.softmax(dy.concatenate(att_scores))
        c = dy.esum([h_input * dy.pick(alphas, j) for j, h_input in enumerate(self.biencoder)])
        
        # softmax over vocabulary
        h_output = dy.tanh(W_c * dy.concatenate([state.output(), c]))
        if not hidden:
            if not scores:
    #            print 'probs:'
                return dy.softmax(R * h_output + bias)
            else:
    #            print 'scores:'
                return R * h_output + bias
        else:
            return h_output

    def consume_next(self, pred_id):
        self.s = self.s.add_input(self.VOCAB_LOOKUP[pred_id])
    
    def consume_next_(self, state, pred_id):
        new_state = state.add_input(self.VOCAB_LOOKUP[pred_id])
        return new_state

    def train(self, input, _true_output):

        true_output = [self.vocab.w2i[a] for a in _true_output]
        true_output += [self.STOP]

        self.param_init(input)

        losses = []
#        pred_id = self.BEGIN
#        while not (output == MAX_PRED_SEQ_LEN or pred_id==self.STOP):
        for pred_id in true_output:
            probs = self.predict_next()
            losses.append(-dy.log(dy.pick(probs, pred_id)))
            self.consume_next(pred_id)
        return dy.average(losses)

    def predict_greedy(self, input):
        self.param_init(input)
        output = []
        while not len(output) == MAX_PRED_SEQ_LEN:
            probs = self.predict_next()
            pred_id = np.argmax(probs.npvalue())
            if pred_id == self.STOP:
                break
            else:
                pred_char = self.vocab.i2w.get(pred_id,UNK_CHAR)
                output.append(pred_char)
                self.consume_next(pred_id)
        output = u''.join(output)
        return output
    
    # This method is correct but not optimized
    def predict_old(self, input, beam = 1):
        self.param_init(input)
        output = []
        hypos = [(self.s, 0., [])]
#        (R, bias, W_c, W__a, U__a, v__a) = self.cg_params
        complete_hypotheses = []
        pred_length = 0

        while pred_length <= MAX_PRED_SEQ_LEN and len(hypos) > 0:
            expansion = []
            for s, log_p, word in hypos:
                # soft attention vector
#                probs = self.predict_next_(s)
#                log_probs_expr = dy.log(probs)
                log_probs_expr = dy.log_softmax(self.predict_next_(s, scores=True))
                log_probs = log_probs_expr.npvalue()
                top = np.argsort(log_probs)[-beam:]
#                print 'expansions ' + u', '.join([self.vocab.i2w.get(pred_id,UNK_CHAR) for pred_id in top])
                expansion.extend(( (s, log_p + log_probs[pred_id], copy.copy(word), pred_id) for pred_id in top ))

            hypos = []
            expansion.sort(key=lambda e: e[1])
            for e in expansion[-beam:]:
                s, log_p, word, pred_id = e
                if pred_id == self.STOP:
                    complete_hypotheses.append((log_p,word))
#                    complete_hypotheses.append((log_p,u''.join(word)))
#                    print u'complete hypo: {}'.format(u''.join(word))
                else:
#                    word.append(self.vocab.i2w.get(pred_id,UNK_CHAR))
                    word.append(pred_id)
                    hypos.append((self.consume_next_(s,pred_id), log_p, word))
#            print 'hypos :'
#            print u', '.join([u''.join(word) for s, log_p, word in hypos])
            pred_length += 1
                
        if not complete_hypotheses:
        # nothing found
            complete_hypotheses = [(log_p,u''.join([self.vocab.i2w.get(pred_id,UNK_CHAR) for pred_id in word])) for s, log_p, word in hypos]

        complete_hypotheses.sort(key=lambda h: h[0], reverse=True)
        final_hypos = []
        for log_p, word in complete_hypotheses[:beam]:
            final_hypos.append((log_p, u''.join([self.vocab.i2w.get(pred_id,UNK_CHAR) for pred_id in word])))
#        print complete_hypotheses
#        print complete_hypotheses[:beam]
        return final_hypos

    @staticmethod
    def _smallest(matrix, k, only_first_row=False):
        """Find k smallest elements of a matrix.
            Parameters
            ----------
            matrix : :class:`np.ndarray`
            The matrix.
            k : int
            The number of smallest elements required.
            Returns
            -------
            Tuple of ((row numbers, column numbers), values).
            """
        #flatten = matrix.flatten()
        if only_first_row:
            flatten = matrix[:1, :].flatten()
        else:
            flatten = matrix.flatten()
        args = np.argpartition(flatten, k)[:k]
        args = args[np.argsort(flatten[args])]
        return np.unravel_index(args, matrix.shape), flatten[args]

    def predict(self, input, beam_size, ignore_first_eol=False, as_arrays=False):
        """Performs beam search.
            If the beam search was not compiled, it also compiles it.
            Parameters
            ----------
            max_length : int
            Maximum sequence length, the search stops when it is reached.
            ignore_first_eol : bool, optional
            When ``True``, the end if sequence symbol generated at the
            first iteration are ignored. This useful when the sequence
            generator was trained on data with identical symbols for
            sequence start and sequence end.
            as_arrays : bool, optional
            If ``True``, the internal representation of search results
            is returned, that is a (matrix of outputs, mask,
            costs of all generated outputs) tuple.
            Returns
            -------
            outputs : list of lists of ints
            A list of the `beam_size` best sequences found in the order
            of decreasing likelihood.
            costs : list of floats
            A list of the costs for the `outputs`, where cost is the
            negative log-likelihood.
            """
        
        
        self.param_init(input)
        states = [self.s] * beam_size
        # This array will store all generated outputs, including those from
        # previous step and those from already finished sequences.
        all_outputs = np.full(shape=(1,beam_size),fill_value=self.BEGIN,dtype = int)
        all_masks = np.ones_like(all_outputs, dtype=float) # whether predicted symbol is self.STOP
        all_costs = np.zeros_like(all_outputs, dtype=float) # the cumulative cost of predictions
        
        for i in range(MAX_PRED_SEQ_LEN):
            if all_masks[-1].sum() == 0:
                break
        
            # We carefully hack values of the `logprobs` array to ensure
            # that all finished sequences are continued with `eos_symbol`.
            logprobs = np.array([-dy.log_softmax(self.predict_next_(s, scores=True)).npvalue() for s in states])
#            print logprobs
#            print all_masks[-1, :, None]
            next_costs = (all_costs[-1, :, None] + logprobs * all_masks[-1, :, None]) #take last row of cumul prev costs and turn into beam_size X 1 matrix, take logprobs distributions for unfinished hypos only and add it (elem-wise) with the array of prev costs; result: beam_size x vocab_len matrix of next costs
            (finished,) = np.where(all_masks[-1] == 0) # finished hypos have all their cost on the self.STOP symbol
            next_costs[finished, :self.STOP] = np.inf
            next_costs[finished, self.STOP + 1:] = np.inf
            
            # indexes - the hypos from prev step to keep, outputs - the next step prediction, chosen cost - cost of predicted symbol
            (indexes, outputs), chosen_costs = self._smallest(next_costs, beam_size, only_first_row=i == 0)
#            print outputs
            # Rearrange everything
            new_states = (states[ind] for ind in indexes)
            all_outputs = all_outputs[:, indexes]
            all_masks = all_masks[:, indexes]
            all_costs = all_costs[:, indexes]
            
            # Record chosen output and compute new states
            states = [self.consume_next_(s,pred_id) for s,pred_id in zip(new_states, outputs)]
            all_outputs = np.vstack([all_outputs, outputs[None, :]])
            all_costs = np.vstack([all_costs, chosen_costs[None, :]])
            mask = outputs != self.STOP
            if ignore_first_eol: #and i == 0:
                mask[:] = 1
            all_masks = np.vstack([all_masks, mask[None, :]])

        all_outputs = all_outputs[1:] # skipping first row of self.BEGIN
        all_masks = all_masks[1:-1] #? all_masks[:-1] # skipping first row of self.BEGIN and the last row of self.STOP
        all_costs = all_costs[1:] - all_costs[:-1] #turn cumulative cost ito cost of each step #?actually the last row would suffice for us?
        result = all_outputs, all_masks, all_costs
        if as_arrays:
            return result
        return self.result_to_lists(self.vocab,result)
    
    @staticmethod
    def result_to_lists(vocab, result):
        outputs, masks, costs = [array.T for array in result]
        outputs = [list(output[:int(mask.sum())]) for output, mask in zip(outputs, masks)]
        words = [u''.join([vocab.i2w.get(pred_id,UNK_CHAR) for pred_id in output]) for output in outputs]
        costs = list(costs.T.sum(axis=0))
        results = zip(costs, words)
        results.sort(key=lambda h: h[0])
        return results

def evaluate_ensemble(nmt_models, data, beam):
    # data is a list of tuples (an instance of SoftDataSet with iter method applied)
    correct = 0.
    final_results = []
    for i,(input,output) in enumerate(data):
        dy.renew_cg()
        predictions = predict_ensemble(nmt_models, input, beam)
        prediction = predictions[0][1]
        #            print i, input, predictions
        if prediction == output:
            correct += 1
#        else:
#            print u'{}, input: {}, pred: {}, true: {}'.format(i, input, prediction, output)
#            print predictions
        final_results.append((input,prediction))  # pred expected as list
    accuracy = correct / len(data)
    return accuracy, final_results


def predict_ensemble(nmt_models, input, beam_size, ignore_first_eol=False, as_arrays=False):
    """Performs beam search for ensemble of models.
    If the beam search was not compiled, it also compiles it.
    Parameters
    ----------
    max_length : int
    Maximum sequence length, the search stops when it is reached.
    ignore_first_eol : bool, optional
    When ``True``, the end if sequence symbol generated at the
    first iteration are ignored. This useful when the sequence
    generator was trained on data with identical symbols for
    sequence start and sequence end.
    as_arrays : bool, optional
    If ``True``, the internal representation of search results
    is returned, that is a (matrix of outputs, mask,
    costs of all generated outputs) tuple.
    Returns
    -------
    outputs : list of lists of ints
    A list of the `beam_size` best sequences found in the order
    of decreasing likelihood.
    costs : list of floats
    A list of the costs for the `outputs`, where cost is the
    negative log-likelihood.
    """
    nmt_vocab = nmt_models[0].vocab # same vocab file for all nmt_models!!
    BEGIN   = nmt_vocab.w2i[BEGIN_CHAR]
    STOP   = nmt_vocab.w2i[STOP_CHAR]
    
    for m in nmt_models:
        m.param_init(input)
    states = [[m.s] * beam_size for m in nmt_models] # ensemble x beam matrix of states
    # This array will store all generated outputs, including those from
    # previous step and those from already finished sequences.
    all_outputs = np.full(shape=(1,beam_size),fill_value=BEGIN,dtype = int)
    all_masks = np.ones_like(all_outputs, dtype=float) # whether predicted symbol is self.STOP
    all_costs = np.zeros_like(all_outputs, dtype=float) # the cumulative cost of predictions

    for i in range(MAX_PRED_SEQ_LEN):
        if all_masks[-1].sum() == 0:
            break

        # We carefully hack values of the `logprobs` array to ensure
        # that all finished sequences are continued with `eos_symbol`.
        logprobs_lst = []
        for j,m in enumerate(nmt_models):
            logprobs_m = np.array([-dy.log_softmax(m.predict_next_(s, scores=True)).npvalue() for s in states[j]]) # beam_size x vocab_len matrix
        #            print logprobs
        #            print all_masks[-1, :, None]
            next_costs = (all_costs[-1, :, None] + logprobs_m * all_masks[-1, :, None]) #take last row of cumul prev costs and turn into beam_size X 1 matrix, take logprobs distributions for unfinished hypos only and add it (elem-wise) with the array of prev costs; result: beam_size x vocab_len matrix of next costs
            (finished,) = np.where(all_masks[-1] == 0) # finished hypos have all their cost on the self.STOP symbol
            next_costs[finished, :STOP] = np.inf
            next_costs[finished, STOP + 1:] = np.inf
#            print next_costs
            (indexes, outputs), chosen_costs = SoftAttention._smallest(next_costs, beam_size, only_first_row=i == 0)
#            print j
#            print ','.join(nmt_vocab.i2w.get(pred_id,UNK_CHAR) for pred_id in outputs)
#            print chosen_costs
#            print indexes
#            print logprobs_m[indexes]
            logprobs_lst.append(logprobs_m)

#        logprobs_lst = np.array([[-dy.log_softmax(m.predict_next_(s, scores=True)).npvalue() for s in m_states] for m,m_states in zip(nmt_models,states)])
#        print logprobs_lst
#        print np.array(logprobs_lst).shape
        logprobs = np.sum(logprobs_lst, axis=0)
#        print logprobs.shape
#        print all_costs[-1, :, None]
#        print logprobs
#        print all_masks[-1, :, None]
        next_costs = (all_costs[-1, :, None] + logprobs * all_masks[-1, :, None]) #take last row of cumul prev costs and turn into beam_size X 1 matrix, take logprobs distributions for unfinished hypos only and add it (elem-wise) with the array of prev costs; result: beam_size x vocab_len matrix of next costs
        (finished,) = np.where(all_masks[-1] == 0) # finished hypos have all their cost on the self.STOP symbol
        next_costs[finished, :STOP] = np.inf
        next_costs[finished, STOP + 1:] = np.inf

        # indexes - the hypos from prev step to keep, outputs - the next step prediction, chosen cost - cost of predicted symbol
        (indexes, outputs), chosen_costs = SoftAttention._smallest(next_costs, beam_size, only_first_row=i == 0)
#        print 'ensemble:'
#        print ','.join(nmt_vocab.i2w.get(pred_id,UNK_CHAR) for pred_id in outputs)
#        print chosen_costs
#        print indexes
        # Rearrange everything
        new_states=[]
        for j,m in enumerate(nmt_models):
            new_states.append([states[j][ind] for ind in indexes])

#        new_states = ((states_m[ind] for ind in indexes) for states_m in states)
        all_outputs = all_outputs[:, indexes]
        all_masks = all_masks[:, indexes]
        all_costs = all_costs[:, indexes]

        # Record chosen output and compute new states
        states = [[m.consume_next_(s,pred_id) for s,pred_id in zip(m_new_states, outputs)] for m,m_new_states in zip(nmt_models, new_states)]
        all_outputs = np.vstack([all_outputs, outputs[None, :]])
        all_costs = np.vstack([all_costs, chosen_costs[None, :]])
        mask = outputs != STOP
#        if ignore_first_eol: # and i == 0:
#            mask[:] = 1
        all_masks = np.vstack([all_masks, mask[None, :]])

    all_outputs = all_outputs[1:] # skipping first row of self.BEGIN
    all_masks = all_masks[1:-1] #? all_masks[:-1] # skipping first row of self.BEGIN and the last row of self.STOP
    all_costs = all_costs[1:] - all_costs[:-1] #turn cumulative cost ito cost of each step #?actually the last row would suffice for us?
    result = all_outputs, all_masks, all_costs
    if as_arrays:
        return result
    return result_to_lists(nmt_vocab, result)

def result_to_lists(nmt_vocab, result):
    outputs, masks, costs = [array.T for array in result]
    outputs = [list(output[:int(mask.sum())]) for output, mask in zip(outputs, masks)]
    words = [u''.join([nmt_vocab.i2w.get(pred_id,UNK_CHAR) for pred_id in output]) for output in outputs]
    costs = list(costs.T.sum(axis=0))
    results = zip(costs, words)
    results.sort(key=lambda h: h[0])
    return results

if __name__ == "__main__":
    arguments = docopt(__doc__)
    print arguments
    
    np.random.seed(123)
    random.seed(123)
    
    model_folder = check_path(arguments['MODEL_FOLDER'], 'MODEL_FOLDER', is_data_path=False)
    
    if arguments['train']:
        
        print '=========TRAINING:========='
        
        assert (arguments['--train_path']!=None) & (arguments['--dev_path']!=None)
        
        # load data
        print 'Loading data...'
        data_set = SoftDataSet
        train_path = check_path(arguments['--train_path'], 'train_path')
        input_format = [int(col) for col in arguments['--input_format'].split(',')]
        train_data = data_set.from_file(train_path,input_format, arguments['--lowercase'])
        print 'Train data has {} examples'.format(train_data.length)
        dev_path = check_path(arguments['--dev_path'], 'dev_path')
        dev_data = data_set.from_file(dev_path,input_format, arguments['--lowercase'])
        print 'Dev data has {} examples'.format(dev_data.length)
    
        print 'Checking if any special symbols in data...'
        for data, name in [(train_data, 'train'), (dev_data, 'dev')]:
            data = set(data.inputs + data.outputs)
            for c in [BEGIN_CHAR, STOP_CHAR, UNK_CHAR]:
                assert c not in data
            print '{} data does not contain special symbols'.format(name)
        print
        
        if os.path.exists(arguments['--vocab_path']):
            vocab_path = arguments['--vocab_path'] # absolute path  to existing vocab file
        else:
            tmp = os.path.join(RESULTS_FOLDER, arguments['--vocab_path'])
            if os.path.exists(tmp): # relative path to existing vocab file
                vocab_path = tmp
            else:
                vocab_path = os.path.join(model_folder,arguments['--vocab_path']) # no vocab - use default name
                print 'Building vocabulary..'
                data = set(train_data.inputs + train_data.outputs)
                build_vocabulary(data, vocab_path)

#        if not os.path.exists(arguments['--vocab_path']):
#            vocab_path = os.path.join(model_folder,arguments['--vocab_path'])
#            print 'Building vocabulary..'
#            data = set(train_data.inputs + train_data.outputs)
#            build_vocabulary(data, vocab_path)

        # Paths for checks and results
        log_file_name   = model_folder + '/log.txt'
        best_model_path  = model_folder + '/bestmodel.txt'
        output_file_path = model_folder + '/best.dev'

        # Model hypoparameters
        model_hyperparams = {'INPUT_DIM': int(arguments['--input']),
                            'HIDDEN_DIM': int(arguments['--hidden']),
                            #'FEAT_INPUT_DIM': int(arguments['--feat-input']),
                            'LAYERS': int(arguments['--layers']),
                            'VOCAB_PATH': vocab_path}
    
        print 'Building model...'
        pc = dy.ParameterCollection()
        ti = SoftAttention(pc, model_hyperparams)

        # Training hypoparameters
        train_hyperparams = {'MAX_PRED_SEQ_LEN': MAX_PRED_SEQ_LEN,
                            'OPTIMIZATION': arguments['--optimization'],
                            'EPOCHS': int(arguments['--epochs']),
                            'PATIENCE': int(arguments['--patience']),
                            'DROPOUT': float(arguments['--dropout']),
                            'BEAM_WIDTH': 1,
                            'TRAIN_PATH': train_path,
                            'DEV_PATH': dev_path}

        print 'Train Hypoparameters:'
        for k, v in train_hyperparams.items():
            print '{:20} = {}'.format(k, v)
        print
        
        trainer = OPTIMIZERS[train_hyperparams['OPTIMIZATION']]
        trainer = trainer(pc)

        best_dev_accuracy = -1.
        sanity_set_size = 100 # for speed - check prediction accuracy on train set
        patience = 0

        # progress bar init
        widgets = [progressbar.Bar('>'), ' ', progressbar.ETA()]
        train_progress_bar = progressbar.ProgressBar(widgets=widgets, maxval=train_hyperparams['EPOCHS']).start()
        
        for epoch in xrange(train_hyperparams['EPOCHS']):
            print 'Start training...'
            then = time.time()

            # compute loss for each sample and update
            train_loss = 0.  # total train loss
            avg_train_loss = 0.  # avg training loss

            for i, (input, output) in enumerate(train_data.iter(shuffle=True)):
                # here we do training
                dy.renew_cg()
#                loss, _ = ti.transduce(input, output)
                loss = ti.train(input, output)
#                if loss is not None:
                train_loss += loss.scalar_value()
                loss.backward()
                trainer.update()

            avg_train_loss = train_loss / train_data.length

            print '\t...finished in {:.3f} sec'.format(time.time() - then)

            # get train accuracy
            print 'evaluating on train...'
            dy.renew_cg() # new graph for all the examples
            then = time.time()
            train_accuracy, _ = ti.evaluate(train_data.iter(indices=sanity_set_size), int(arguments['--beam']))
            print '\t...finished in {:.3f} sec'.format(time.time() - then)
            
            # get dev accuracy
            print 'evaluating on dev...'
            then = time.time()
            dy.renew_cg() # new graph for all the examples
            dev_accuracy, _ = ti.evaluate(dev_data.iter(), int(arguments['--beam']))
            print '\t...finished in {:.3f} sec'.format(time.time() - then)

            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
                # save best model
                ti.save_model(best_model_path)
                print 'saved new best model to {}'.format(best_model_path)
                patience = 0
            else:
                patience += 1

            # found "perfect" model
            if dev_accuracy == 1:
                train_progress_bar.finish()
                break

            print ('epoch: {0} train loss: {1:.4f} dev accuracy: {2:.4f} '
                   'train accuracy: {3:.4f} best dev accuracy: {4:.4f} patience = {5}').format(epoch, avg_train_loss, dev_accuracy, train_accuracy, best_dev_accuracy, patience)

            log_to_file(log_file_name, epoch, avg_train_loss, train_accuracy, dev_accuracy)

            if patience == train_hyperparams['PATIENCE']:
                print 'out of patience after {} epochs'.format(epoch)
                train_progress_bar.finish()
                break
            # finished epoch
            train_progress_bar.update(epoch)
                
        print 'finished training.'
        
        ti = SoftAttention(pc, model_hyperparams, best_model_path)
        dev_accuracy, dev_results = ti.evaluate(dev_data.iter(), int(arguments['--beam']))
        print 'Best dev accuracy: {}'.format(dev_accuracy)
        write_param_file(output_file_path, dict(model_hyperparams.items()+train_hyperparams.items()))
        write_pred_file(output_file_path, dev_results)
        write_eval_file(output_file_path, best_dev_accuracy, dev_path)

    elif arguments['test']:
        print '=========EVALUATION ONLY:========='
        # requires test path, model path of pretrained path and results path where to write the results to
        assert arguments['--test_path']!=None

        print 'Loading data...'
        test_path = check_path(arguments['--test_path'], '--test_path')
        data_set = SoftDataSet
        input_format = [int(col) for col in arguments['--input_format'].split(',')]
        test_data = data_set.from_file(test_path,input_format, arguments['--lowercase'])
        print 'Test data has {} examples'.format(test_data.length)

        print 'Checking if any special symbols in data...'
        data = set(test_data.inputs + test_data.outputs)
        for c in [BEGIN_CHAR, STOP_CHAR, UNK_CHAR]:
            assert c not in data
        print 'Test data does not contain special symbols'

        best_model_path  = model_folder + '/bestmodel.txt'
        output_file_path = os.path.join(model_folder,arguments['--pred_path'])
        hypoparams_file = model_folder + '/best.dev'
        
        hypoparams_file_reader = codecs.open(hypoparams_file, 'r', 'utf-8')
        hyperparams_dict = dict([line.strip().split(' = ') for line in hypoparams_file_reader.readlines()])
        model_hyperparams = {'INPUT_DIM': int(hyperparams_dict['INPUT_DIM']),
                            'HIDDEN_DIM': int(hyperparams_dict['HIDDEN_DIM']),
                            'LAYERS': int(hyperparams_dict['LAYERS']),
                            'VOCAB_PATH': hyperparams_dict['VOCAB_PATH']}
        # a fix for vocab path when transferring files b/n vm
        model_hyperparams['VOCAB_PATH'] = check_path(model_folder + '/vocab.txt', 'vocab_path', is_data_path=False)
        pc = dy.ParameterCollection()
        ti = SoftAttention(pc, model_hyperparams, best_model_path)

        print 'Evaluating on test..'
        t = time.clock()
        accuracy, test_results = ti.evaluate(test_data.iter(), int(arguments['--beam']))
        print 'Time: {}'.format(time.clock()-t)
        print 'accuracy: {}'.format(accuracy)
        write_pred_file(output_file_path, test_results)
        write_eval_file(output_file_path, accuracy, test_path)

    elif arguments['ensemble_test']:
        print '=========EVALUATION ONLY:========='
        # requires test path, model path of pretrained path and results path where to write the results to
        assert arguments['--test_path']!=None
        
        print 'Loading data...'
        test_path = check_path(arguments['--test_path'], '--test_path')
        data_set = SoftDataSet
        input_format = [int(col) for col in arguments['--input_format'].split(',')]
        test_data = data_set.from_file(test_path,input_format, arguments['--lowercase'])
        print 'Test data has {} examples'.format(test_data.length)
        
        print 'Checking if any special symbols in data...'
        data = set(test_data.inputs + test_data.outputs)
        for c in [BEGIN_CHAR, STOP_CHAR, UNK_CHAR]:
            assert c not in data
        print 'Test data does not contain special symbols'

        pc = dy.ParameterCollection()
        ed_models= []
        ed_model_params = []
        for i,path in enumerate(arguments['ED_MODEL_FOLDER'].split(',')):
            print '...Loading nmt model {}'.format(i)
            ed_model_folder =  check_path(path, 'ED_MODEL_FOLDER_{}'.format(i), is_data_path=False)
            best_model_path  = ed_model_folder + '/bestmodel.txt'
            hypoparams_file_reader = codecs.open(ed_model_folder + '/best.dev', 'r', 'utf-8')
            hyperparams_dict = dict([line.strip().split(' = ') for line in hypoparams_file_reader.readlines()])
            model_hyperparams = {'INPUT_DIM': int(hyperparams_dict['INPUT_DIM']),
                'HIDDEN_DIM': int(hyperparams_dict['HIDDEN_DIM']),
                    'LAYERS': int(hyperparams_dict['LAYERS']),
                        'VOCAB_PATH': hyperparams_dict['VOCAB_PATH']}
            # a fix for vocab path when transferring files b/n vm
            model_hyperparams['VOCAB_PATH'] = check_path(path + '/vocab.txt', 'vocab_path', is_data_path=False)
            ed_model_params.append(pc.add_subcollection('ed{}'.format(i)))
            ed_model =  SoftAttention(ed_model_params[i], model_hyperparams,best_model_path)
            
            ed_models.append(ed_model)
            best_model_path  = model_folder + '/ed_bestmodel_{}.txt'.format(i)

        ensemble_number = len(ed_models)
        output_file_path = os.path.join(model_folder,arguments['--pred_path'])

        print 'Evaluating on test..'
        t = time.clock()
        accuracy, test_results = evaluate_ensemble(ed_models, test_data.iter(), int(arguments['--beam']))
        print 'Time: {}'.format(time.clock()-t)
        print 'accuracy: {}'.format(accuracy)
        write_pred_file(output_file_path, test_results)
        write_eval_file(output_file_path, accuracy, test_path)

