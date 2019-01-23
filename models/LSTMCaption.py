from models import RNNCaption
from models.layers import LSTM, WordEmbedding, LinearForRNN, Linear
from models.losses import SoftmaxForRNN


class LSTMCaption(RNNCaption):
    """
    Structure:
        
    """
    def init(self):
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
        self.layers = [Linear(self.dim_input, num_hidden, 
                            init_scale=self.init_scale, device=self.device),
    
                       WordEmbedding(num_word, num_vector, 
                            init_scale=self.init_scale, device=self.device),
                                     
                       LSTM(num_vector, num_hidden, 
                            init_scale=self.init_scale, device=self.device),
                            
                       LinearForRNN(num_hidden, num_word, 
                            init_scale=self.init_scale, device=self.device)]
        
        self._linear = 0
        self._wembed = 1
        self._rnnlyr = 2
        self._linrnn = 3
        
        # init loss
        self.loss = SoftmaxForRNN(reg=self.reg, device=self.device)
        