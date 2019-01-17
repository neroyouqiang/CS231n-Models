import numpy as np

from models import MultiLayerNet
from models.layers import RNN, WordEmbedding, LinearForRNN, Linear
from models.losses import SoftmaxForRNN


class RNNCaption(MultiLayerNet):
    """
    Structure:
        
    """
    def init(self):
        if self.seed: np.random.seed(self.seed)
        
        assert 'word_to_idx' in self.hyperparams, 'Please input the "word_to_idx" map'
        self.hyperparams.setdefault('num_hidden', 128)
        self.hyperparams.setdefault('num_vector', 128)
        
        word_to_idx = self.hyperparams['word_to_idx']
        num_hidden = self.hyperparams['num_hidden']
        num_vector = self.hyperparams['num_vector']
        num_word = len(word_to_idx)
        
        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx['<START>']
        self._end = word_to_idx['<END>']
            
        # init layers
        self.layers = [Linear(self.dim_input, num_hidden, init_scale=self.init_scale),
                       WordEmbedding(num_word, num_vector, init_scale=self.init_scale),
                       RNN(num_vector, num_hidden, init_scale=self.init_scale),
                       LinearForRNN(num_hidden, num_word, init_scale=self.init_scale)]
        
        self._linear = 0
        self._wembed = 1
        self._rnnlyr = 2
        self._linrnn = 3
        
        # init loss
        self.loss = SoftmaxForRNN()
        
        
    def predict(self, x, seed=None):
        """
        Always be 'test' mode
        """
        if seed: np.random.seed(seed)
        
        max_length = 30
        
        # init captions
        N = x.shape[0]
        caps = self._null * np.ones((N, max_length), dtype=np.int32)
        caps[:, 0] = self._start
        
        # calculate captions
        h, _ = self.layers[self._linear].forward(x, self.params[self._linear], mode='test')
        for i in range(max_length - 1):
            v, _ = self.layers[self._wembed].forward(caps[:, i], self.params[self._wembed], mode='test')
            h = self.layers[self._rnnlyr].forward_step(v, h, self.params[self._rnnlyr])
            s, _ = self.layers[self._linrnn].forward(h, self.params[self._linrnn], mode='test')
            
            caps[:, i + 1] = np.argmax(s, axis=1)
            
        # return captions
        return caps
    
    
    def backward(self, x, y, mode=None, seed=None):
        """
        For test mode, don't calculate gradient.
        """
        cap_in = y[:, :-1]
        cap_out = y[:, 1:]
        
        self.caches = [{} for _ in range(len(self.layers))]
        
        h0, self.caches[self._linear] = self.layers[self._linear].forward(x, self.params[self._linear], mode='train')
        vs, self.caches[self._wembed] = self.layers[self._wembed].forward(cap_in,  self.params[self._wembed], mode='train')
        
        hs, self.caches[self._rnnlyr] = self.layers[self._rnnlyr].forward(vs, h0, self.params[self._rnnlyr])
        
        ss, self.caches[self._linrnn] = self.layers[self._linrnn].forward(hs, self.params[self._linrnn], mode='train')

        # init dparams
        self.dparams = [{} for _ in range(len(self.layers))]
        
        # calculate grediant
        loss, dss = self.loss.backward(ss, cap_out)
        
        dhs, self.dparams[self._linrnn] = self.layers[self._linrnn].backward(dss, self.caches[self._linrnn])
            
        dvs, dh0, self.dparams[self._rnnlyr] = self.layers[self._rnnlyr].backward(dhs, self.caches[self._rnnlyr])
            
        dcap_in, self.dparams[self._wembed] = self.layers[self._wembed].backward(dvs, self.caches[self._wembed])
        dx, self.dparams[self._linear] = self.layers[self._linear].backward(dh0, self.caches[self._linear])
        
        # regularization
        for i in range(len(self.params)):
            if 'W' in self.params[i]:
                loss += self.reg * np.sum(np.square(self.params[i]['W'])) / 2.
                self.dparams[i]['W'] += self.reg * self.params[i]['W']
        
        # return 
        return loss