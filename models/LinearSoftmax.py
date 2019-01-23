#import minpy.numpy as np
import numpy as np

from models.layers import Linear
from models.losses import Softmax


class LinearSoftmax:
    def __init__(self, num_input, num_output, hyperparams={}, seed=None):
        if seed: np.random.seed(seed)
        
        hyperparams.setdefault('reg', 1e-5)
        self.reg = hyperparams['reg']
        
        self.layers = [Linear(num_input, num_output, init_scale=0.0001)]
        self.loss = Softmax()
        
        self.params = [self.layers[0].get_init_param()]
        
    
    def predict(self, x, mode='test', seed=None):
        if seed: np.random.seed(seed)
        
        self.init_caches()
        scores, self.caches[0] = self.layers[0].forward(x, self.params[0], mode=mode)
        return scores
    
    
    def backward(self, x, y, mode='train', seed=None):
        # calculate loss
        scores = self.predict(x, mode=mode, seed=seed)
        
        # calculate grediant
        loss, dscores = self.loss.backward(scores, y)
        dx, self.dparams[0] = self.layers[0].backward(dscores, self.caches[0])
        
        # regularization
        loss += self.reg * np.sum(np.square(self.params[0]['W'])) / 2.
        self.dparams[0]['W'] += self.reg * self.params[0]['W']
        
        # return 
        return loss
        
    def init_caches(self):
        self.caches = [{}] 
        self.dparams = [{}] 