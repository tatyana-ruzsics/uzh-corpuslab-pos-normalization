from srilm import readLM, initLM, getNgramProb, getIndexForWord, howManyNgrams

class SRILM_char_lm_loader(object):
    """TODO"""
    def __init__(self, model_path, order=2):
        self.lm_path = model_path
        self.order=order
        self.history=[]
        self.EOS_ID = 1
        self.history_len = order-1
        print u'Initialize LM with order {}'.format(order)
        self.lm = initLM(order)
        readLM(self.lm, self.lm_path)
        self.vocab_size = howManyNgrams(self.lm, 1)
    
    def param_init(self):
        """Initializes the history with the start-of-sentence symbol."""
        self.history = ['<s>'] if self.history_len > 0 else []
        
    def score(self, char):
        """retrieve the probability of a sequence from the language model
         based on the current history"""
        prefix = "%s " % ' '.join(self.history)
        order = len(self.history) + 1
#        print u'scoring char: {}, {}, {}'.format(prefix,char,order)
        #		print u'history, scoring with lm: {}, {}'.format(prefix,prefix + ("</s>" if char == self.EOS_ID else str(char)))

        ret = getNgramProb(self.lm, prefix + ("</s>" if char == self.EOS_ID else str(char)), order)
        return ret

    def consume(self, char):
        """add predicted item to vocabulary"""
        if len(self.history) >= self.history_len:
            self.history = self.history[1:]
        self.history.append(str(char))

    def get_state(self):
        """Returns the current n-gram history """
        return self.history

    def set_state(self, state):
        """Sets the current n-gram history """
        self.history = state

class SRILM_morpheme_lm_loader(SRILM_char_lm_loader):
    def __init__(self, model_path, order=2):
        super(SRILM_morpheme_lm_loader, self).__init__(model_path, order)
        self.vocab = None
        print 'Using LM over words'

    def score(self, morpheme, eow=0):
        """Score the target MORPHEME with the n-gram language model given the current history of MORPHEMES.
        Args:
        morpheme (string): morpheme to score
        eow (int): whether morpheme is followed by end of word symbold (1) or not (0)
        Returns:
        logprob (float). Language model score for the next morpheme
        """
        prefix = "%s " % ' '.join(self.history)
        order = len(self.history) + 1
        
        if eow==1:
            # Score for the end of word symbol:
            # logP(second-last-morf last-morf morf </s>) = logP(second-last-morf last-morf morf) + logP(last-morf morf </s>)
            prefix_eos = "%s " % ' '.join(self.history[1:]) if len(self.history) == self.history_len else "%s " % ' '.join(self.history)
            order_eos = order if len(self.history) == self.history_len else order+1
#            print u'scoring morfeme: {}, {}'.format(morpheme,order)
#            print u'scoring eow: {}, {}'.format(morpheme,order_eos)
            prob = (getNgramProb(self.lm, prefix + str(morpheme), order) + getNgramProb(self.lm, prefix_eos + str(morpheme) + " </s>", order_eos))
        else:
#            print u'scoring morfeme: {}, {}'.format(morpheme,order)
            prob = getNgramProb(self.lm, prefix + str(morpheme), order)
        return prob


if __name__=="__main__":
    import sys
    sri_lm_char = SRILM_char_lm_loader(sys.argv[1],order=7)
    sri_lm_morph = SRILM_morpheme_lm_loader(sys.argv[1],order=7)
    print sri_lm_char.predict_next(2035)
    print sri_lm_morph.predict_next(2035)
    sri_lm_morph.consume(2035)
    print sri_lm_morph.predict_next(1788)
