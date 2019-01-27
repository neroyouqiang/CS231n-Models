#import minpy.numpy as np
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import time

from models import MultiLayerNet
from models.layers import RNN, WordEmbedding, LinearForRNN, Linear
from models.losses import SoftmaxForRNN


class RNNCaption(MultiLayerNet):
    """
    Structure:
        Input - Linear - LSTM - Linear
         WordEmbedding /
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
                            init_scale=self.init_scale),
    
                       WordEmbedding(num_word, num_vector, 
                            init_scale=self.init_scale),
                                     
                       RNN(num_vector, num_hidden, 
                            init_scale=self.init_scale),
                           
                       LinearForRNN(num_hidden, num_word, 
                            init_scale=self.init_scale)]
        
        self._linear = 0
        self._wembed = 1
        self._rnnlyr = 2
        self._linrnn = 3
        
        # init loss
        self.loss = SoftmaxForRNN(reg=self.reg)
        
        
#    def predict(self, x, seed=None):
#        """
#        Always be 'test' mode
#        """
#        if seed: np.random.seed(seed)
#        
#        max_length = 30
#        
#        # init captions
#        N = x.shape[0]
#        caps = self._null * np.ones((N, max_length), dtype=np.int32)
#        caps[:, 0] = self._start
#        
#        # calculate captions
#        h, _ = self.layers[self._linear].forward(x, self.params[self._linear], mode='test')
#        for i in range(max_length - 1):
#            v, _ = self.layers[self._wembed].forward(caps[:, i], self.params[self._wembed], mode='test')
#            h = self.layers[self._rnnlyr].forward_step(v, h, self.params[self._rnnlyr])
#            s, _ = self.layers[self._linrnn].forward(h, self.params[self._linrnn], mode='test')
#            
#            caps[:, i + 1] = np.argmax(s, axis=1)
#            
#        # return captions
#        return caps
    
    
    def predict(self, x, seed=None):
        """
        Always be 'test' mode
        """
        # the random seed
        if seed is not None: 
            np.random.seed(seed)
        else:
            np.random.seed()
        
        max_length = 30
        
        # init captions
        N = x.shape[0]
        caps = self._null * np.ones((N, max_length), dtype=np.int32)
        caps[:, 0] = self._start
        
        # calculate captions
        h, _ = self.layers[self._linear].forward(x, self.params[self._linear], mode='test')
        c = np.zeros(h.shape)
        for i in range(max_length - 1):
            v, _ = self.layers[self._wembed].forward(caps[:, i], self.params[self._wembed], mode='test')
            h, c = self.layers[self._rnnlyr].forward_step(v, h, c, self.params[self._rnnlyr])
            s, _ = self.layers[self._linrnn].forward(h, self.params[self._linrnn], mode='test')
            
            # caption will always be numpy
            if type(s) == nd.ndarray.NDArray:
                s = s.asnumpy()
            
            # combine captions
            caps[:, i + 1] = np.argmax(s, axis=1)
            
        # return captions
        return caps
    
    
    def backward(self, x, y, mode=None, seed=None, print_time=False):
        """
        Always be 'train' mode
        """
        # the random seed
        if seed is not None: 
            np.random.seed(seed)
        else:
            np.random.seed()
            
        cap_in = y[:, :-1]
        cap_out = y[:, 1:]
        
        # time recorder
        tfs = [time.time()]
        
        # forward
        self.caches = [{} for _ in range(len(self.layers))]
        
        h0, self.caches[self._linear] = self.layers[self._linear].forward(x, self.params[self._linear], mode='train')
        tfs.append(time.time())
        
        vs, self.caches[self._wembed] = self.layers[self._wembed].forward(cap_in,  self.params[self._wembed], mode='train')
        tfs.append(time.time())
        
        hs, self.caches[self._rnnlyr] = self.layers[self._rnnlyr].forward(vs, h0, self.params[self._rnnlyr])
        tfs.append(time.time())
        
        ss, self.caches[self._linrnn] = self.layers[self._linrnn].forward(hs, self.params[self._linrnn], mode='train')
        tfs.append(time.time())
        
        # backward
        self.dparams = [{} for _ in range(len(self.layers))]
        
        # time recorder
        tbs = [time.time()]
        
        loss, dss = self.loss.backward(ss, cap_out)
        tbs.append(time.time())
        
        dhs, self.dparams[self._linrnn] = self.layers[self._linrnn].backward(dss, self.caches[self._linrnn])
        tbs.append(time.time()) 
        
        dvs, dh0, self.dparams[self._rnnlyr] = self.layers[self._rnnlyr].backward(dhs, self.caches[self._rnnlyr])
        tbs.append(time.time()) 
        
        dcap_in, self.dparams[self._wembed] = self.layers[self._wembed].backward(dvs, self.caches[self._wembed])
        tbs.append(time.time())
        
        dx, self.dparams[self._linear] = self.layers[self._linear].backward(dh0, self.caches[self._linear])
        tbs.append(time.time())
        
        # time recorder
        trs = [time.time()]
        
        # regularization
        loss, self.dparams = self.loss.add_reg(loss, self.params, self.dparams)
        trs.append(time.time())
        
        # print running time
        if print_time: self._print_time(tfs, tbs, trs)
        
        # loss will always be numpy
        if type(loss) == nd.ndarray.NDArray:
            loss = loss.asnumpy()[0]
        
        # return 
        return loss
    
    
    def _print_time(self, tfs, tbs, trs):
        print('Forward time:', tfs[-1] - tfs[0])
        print('    Input linear forward time:', tfs[1] - tfs[0])
        print('    Word embedding forward time:', tfs[2] - tfs[1])
        print('    RNN forward time:', tfs[3] - tfs[2])
        print('    Output linear forward time:', tfs[4] - tfs[3])
        
        print('\nBackward time:', tbs[-1] - tbs[0])
        print('    Loss calculate time:', tbs[1] - tbs[0])
        print('    Output linear backward time:', tbs[2] - tbs[1])
        print('    RNN backward time:', tbs[3] - tbs[2])
        print('    Word embedding backward time:', tbs[4] - tbs[3])
        print('    Input linear backward time:', tbs[5] - tbs[4])
        
        print('\nReg time:', trs[-1] - trs[0])
        
        
        
        