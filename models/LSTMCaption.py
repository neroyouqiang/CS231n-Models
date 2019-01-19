import numpy as np

from models import RNNCaption
from models.layers import LSTM, WordEmbedding, LinearForRNN, Linear
from models.losses import SoftmaxForRNN


class LSTMCaption(RNNCaption):
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
                       LSTM(num_vector, num_hidden, init_scale=self.init_scale),
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
        c = np.zeros_like(h)
        for i in range(max_length - 1):
            v, _ = self.layers[self._wembed].forward(caps[:, i], self.params[self._wembed], mode='test')
            h, c = self.layers[self._rnnlyr].forward_step(v, h, c, self.params[self._rnnlyr])
            s, _ = self.layers[self._linrnn].forward(h, self.params[self._linrnn], mode='test')
            
            caps[:, i + 1] = np.argmax(s, axis=1)
            
        # return captions
        return caps