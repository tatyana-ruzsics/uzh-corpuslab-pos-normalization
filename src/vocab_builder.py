#!/usr/bin/env python
# -*- coding: utf-8 -*
"""Builds vocabulary

Usage:
  vocab_builder.py build [--input_col=COL] [--segments] [--vocab_trunk=VOCAB_TRUNK] [--lowercase=LOW]
  DATA_PATH VOCAB_PATH
  vocab_builder.py apply [--segments] [--lowercase=LOW]
  DATA_PATH_IN VOCAB_PATH DATA_PATH_OUT
  
Arguments:
  DATA_PATH     path to date
  VOCAB_PATH    path to save vocab
  DATA_PATH_IN  path to data to be converted with vocab mapping
  DATA_PATH_OUT path to save converted data

Options:
  --vocab_trunk=VOCAB_TRUNK     precentage of vocabulary to be replaced with unk [default: 0]
  --segments                    build vocabulary over segments instead of chars
  --input_col=COL               input column [default: 0]
  --lowercase=LOW               use lowercased data [default: True]
 """

from docopt import docopt
from common import BEGIN_CHAR,STOP_CHAR,UNK_CHAR,BOUNDARY_CHAR, check_path
import codecs
import collections

# represents a bidirectional mapping from strings to ints
class Vocab(object):
    def __init__(self, w2i):
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.iteritems()}
    
    def save(self, vocab_path):
        with codecs.open(vocab_path, 'w', 'utf-8') as fh:
            for w,i in sorted(self.w2i.iteritems(),key=lambda v:v[0]):
                fh.write(u'{}\t{}\n'.format(w,i))
        return

    
    @classmethod
    def from_list(cls, words, w2i=None):
        if w2i:
            idx=len(w2i)
        else:
            w2i = {}
            idx = 0
        for word in words:
            w2i[word] = idx
            idx += 1
        return Vocab(w2i)
    
    @classmethod
    def from_file(cls, vocab_fname):
        w2i = {}
        with codecs.open(vocab_fname, 'r', 'utf-8') as fh:
            for line in fh:
                word, idx = line.rstrip().split('\t')
                w2i[word] = int(idx)
                #print word, idx
        return Vocab(w2i)
    
    def size(self): return len(self.w2i.keys())

def build_vocabulary(train_data, vocab_path, vocab_trunk=0):
    # Build vocabulary over items - chars or segments - and save it to 'vocab_path'
    
    if vocab_trunk==0:
        items = list(set([c for w in train_data for c in w])) #+ [STOP_CHAR] + [UNK_CHAR] + [BEGIN_CHAR]
#       print set([c for w in train_data for c in w])
    else:
        tokens = [c for w in train_data for c in w]
        counter = collections.Counter(tokens)
        print u'Word types in train set: {}'.format(len(counter))
        n = len(counter) - int(len(counter)*vocab_trunk)
        print u'Trunkating: {}'.format(n)
        items = [w for w,c in counter.most_common(n)]


    # to make sure that special symbols have the same index across models
    w2i = {}
    w2i[BEGIN_CHAR] = 0
    w2i[STOP_CHAR] = 1
    w2i[UNK_CHAR] = 2
    print 'Vocabulary size: {}'.format(len(items))
    print 'Example of vocabulary items:' + u', '.join(items[:10])
    print
    vocab = Vocab.from_list(items,w2i)
    vocab.save(vocab_path)
    return

def read(filename, input_col=0, over_segs=False, lowercase=False):
    """
        Read a file where each line is of the form "word1 word2 ..."
        Yields lists of the lines from file
    """
    with codecs.open(filename, encoding='utf8') as fh:
        for line in fh:
            if not len(line.strip())==0:
                try:
                    splt = line.strip().split('\t')
                    target = splt[input_col].lower() if lowercase else splt[input_col]
                    #language model is trained on the target side of the corpus
                    if over_segs:
                        # Segments
                        yield target.split(BOUNDARY_CHAR)
                    else:
                        # Chars
                        yield [c for c in target]
                except:
                    print u"bad line: {}".format(line)

def apply(file_in, file_out, vocab_path, over_segs=False, lowercase=False):
    vocab = Vocab.from_file(vocab_path)
    with codecs.open(file_in, 'r', 'utf8') as f_in:
        with codecs.open(file_out, 'w', 'utf-8') as f_out:
            for line in f_in:
                if not len(line.strip())==0:
#                    try:
                        target = line.strip().lower() if lowercase else line.strip()
                        if over_segs:
                            # Segments
                            mapped_items = [str(vocab.w2i.get(w, vocab.w2i[UNK_CHAR])) for w in target.split(BOUNDARY_CHAR)]
                            f_out.write(u'{}\n'.format(BOUNDARY_CHAR.join(mapped_items)))
                        else:
                            # Chars
                            mapped_items = [str(vocab.w2i.get(c, vocab.w2i[UNK_CHAR])) for c in target]
                            f_out.write(u'{}\n'.format(' '.join(mapped_items)))
#                    except:
#                        print u"bad line: {}".format(line)

if __name__ == "__main__":
    arguments = docopt(__doc__)
    print arguments
    
    if arguments['build']:
        assert arguments['DATA_PATH']!=None
        assert arguments['VOCAB_PATH']!=None
        
        print 'Loading data...'
        over_segs = arguments['--segments']
        data_path = check_path(arguments['DATA_PATH'], 'DATA_PATH')
        vocab_path = arguments['VOCAB_PATH']
        input_format = int(arguments['--input_col'])
        data = list(read(data_path, input_format, over_segs, arguments['--lowercase']))
        print 'Data has {} examples'.format(len(data))

        build_vocabulary(data,vocab_path, float(arguments['--vocab_trunk']))
    elif arguments['apply']:
        assert arguments['DATA_PATH_IN']!=None
        assert arguments['DATA_PATH_OUT']!=None
        assert arguments['VOCAB_PATH']!=None

        apply(arguments['DATA_PATH_IN'],arguments['DATA_PATH_OUT'],arguments['VOCAB_PATH'],arguments['--segments'],arguments['--lowercase'])


