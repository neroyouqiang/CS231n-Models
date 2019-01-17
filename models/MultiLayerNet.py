import numpy as np
import pickle
import os

from models.layers import Linear, ReLU, DropOut, BatchNorm
from models.losses import Softmax, SVM

#from layers import Linear, ReLU
#from losses import Softmax, SVM


class MultiLayerNet:
    def __init__(self, dim_input, dim_output, hyperparams={}, seed=None, params=None):
        # hyperparameters
        self.hyperparams = hyperparams
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.seed = seed
            
        self.hyperparams.setdefault('reg', 0.)
        self.hyperparams.setdefault('init_scale', 1e-4)
            
        self.reg = self.hyperparams['reg']
        self.init_scale = self.hyperparams['init_scale']
            
        # init model
        self.init()
        
        # init parameters
        if params:
            self.params = params
        else:
            self.params = []
            for layer in self.layers:
                self.params.append(layer.get_init_param())
            
        
    def init(self):
        if self.seed: np.random.seed(self.seed) # seed used to fix the initialization
        
        self.hyperparams.setdefault('nums_hidden', [])
        self.hyperparams.setdefault('loss_type', 'softmax')
        self.hyperparams.setdefault('dropout', None)
        self.hyperparams.setdefault('batchnorm', False)
            
        # init layers
        self.layers = []
        
        ni = self.dim_input
        for no in self.hyperparams['nums_hidden']:
            self.layers.append(Linear(ni, no, init_scale=self.init_scale))
            self.layers.append(ReLU())
            if self.hyperparams['dropout']:
                self.layers.append(DropOut(p=self.hyperparams['dropout']))
            if self.hyperparams['batchnorm']:
                self.layers.append(BatchNorm(no))
            ni = no
            
        self.layers.append(Linear(ni, self.dim_output, init_scale=self.init_scale))
        
        # init loss
        if self.hyperparams['loss_type'] == 'softmax':
            self.loss = Softmax()
        elif self.hyperparams['loss_type'] == 'svm':
            self.loss = SVM()
            
    
    def predict(self, x, mode='test', seed=None):
        if seed: np.random.seed(seed) # seed used to fix the forward
        
        # init caches
        self.caches = [{} for _ in range(len(self.layers))]
        
        # calculate scores
        for i in range(len(self.layers)):
            x, self.caches[i] = self.layers[i].forward(x, self.params[i], mode=mode)
            
        return x
    
    
    def backward(self, x, y, mode='train', seed=None):
        """
        For test mode, don't calculate gradient.
        """
        # calculate loss
        scores = self.predict(x, mode=mode, seed=seed)
        
        # init dparams
        if mode == 'train': 
            self.dparams = [{} for _ in range(len(self.layers))]
        
        # calculate grediant
        loss, dy = self.loss.backward(scores, y)
        if mode == 'train': 
            for i in range(len(self.layers) - 1, -1, -1):
                dy, self.dparams[i] = self.layers[i].backward(dy, self.caches[i])
        
        # regularization
        for i in range(len(self.params)):
            if 'W' in self.params[i]:
                loss += self.reg * np.sum(np.square(self.params[i]['W'])) / 2.
                if mode == 'train': 
                    self.dparams[i]['W'] += self.reg * self.params[i]['W']
        
        # return 
        return loss
              
    
    def save(self, file_name):
        data = {'params': self.params,
                'info': {'seed': self.seed,
                         'dim_input': self.dim_input,
                         'dim_output': self.dim_output,
                         'hyperparams': self.hyperparams}}
        
        dir_name = os.path.dirname(file_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name) 
            
        pickle.dump(data, open(file_name, 'wb'))
    
    
    @classmethod 
    def load(cls, file_name):
        data = pickle.load(open(file_name, 'rb'))
        
        hyperparams = data['info']['hyperparams']
        dim_input = data['info']['dim_input']
        dim_output = data['info']['dim_output']
        seed = data['info']['seed']
        params = data['params']
        
        return cls(dim_input, dim_output, 
                   hyperparams=hyperparams, seed=seed, params=params)
        
    
if __name__ == '__main__':
    model = MultiLayerNet(2, 3)
    model.backward(np.array([[1, 2]]), np.array([[0, 1, 2]]))