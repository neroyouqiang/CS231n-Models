import numpy as np
import pickle
imoprt os

from models.layers import Linear, ReLU, DropOut, BatchNorm
from models.losses import Softmax, SVM

#from layers import Linear, ReLU
#from losses import Softmax, SVM


class MultiLayerNet:
    def __init__(self, dim_input=None, dim_output=None, hyperparams={}, seed=None):
        if dim_input:
            hyperparams.setdefault('reg', 0.)
            hyperparams.setdefault('nums_hidden', [])
            hyperparams.setdefault('init_scale', 1e-4)
            hyperparams.setdefault('loss_type', 'softmax')
            hyperparams.setdefault('dropout', None)
            hyperparams.setdefault('batchnorm', False)
            
            self.hyperparams = hyperparams
            self.dim_input = dim_input
            self.dim_output = dim_output
            self.seed = seed
            
            self.init()
            
        
    def init(self, params=None):
        if self.seed: np.random.seed(self.seed) # seed used to fix the initialization
        
        self.reg = self.hyperparams['reg']
        self.nums_hidden = self.hyperparams['nums_hidden']
        self.loss_type = self.hyperparams['loss_type']
        self.init_scale = self.hyperparams['init_scale']
        self.dropout = self.hyperparams['dropout']
        self.batchnorm = self.hyperparams['batchnorm']
            
        # init layers
        self.layers = []
        ni = self.dim_input
        for no in self.nums_hidden :
            self.layers.append(Linear(ni, no, init_scale=self.init_scale))
            self.layers.append(ReLU())
            if self.dropout:
                self.layers.append(DropOut(p=self.dropout))
            if self.batchnorm:
                self.layers.append(BatchNorm(no))
            ni = no
        self.layers.append(Linear(ni, self.dim_output, init_scale=self.init_scale))
        
        # init loss
        if self.loss_type == 'softmax':
            self.loss = Softmax()
        elif self.loss_type == 'svm':
            self.loss = SVM()
        
        # Init parameters
        if params is None:
            self.params = []
            for layer in self.layers:
                self.params.append(layer.get_init_param())
        else:
            self.params = params
            
    
    def predict(self, x, mode='test', seed=None):
        if seed: np.random.seed(seed) # seed used to fix the forward
        
        self._init_caches(mode=mode)
        
        for i in range(len(self.layers)):
            x, self.caches[i] = self.layers[i].forward(x, self.params[i], mode=mode)
            
        return x
    
    
    def backward(self, x, y, mode='train', seed=None):
        """
        For test mode, don't calculate gradient.
        """
        # calculate loss
        scores = self.predict(x, mode=mode, seed=seed)
        
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
    
    
    def _init_caches(self, mode):
        self.caches = [{} for _ in range(len(self.layers))]
        if mode == 'train': 
            self.dparams = [{} for _ in range(len(self.layers))]
            
            
    def load(self, file_name):
        data = pickle.load(open(file_name, 'rb'))
        
        self.hyperparams = data['info']['hyperparams']
        self.dim_input = data['info']['dim_input']
        self.dim_output = data['info']['dim_output']
        self.seed = data['info']['seed']
        
        self.init(data['params'])
    
        
    def save(self, file_name):
        data = {'params': self.params,
                'info': {'seed': self.seed,
                         'dim_input': self.dim_input,
                         'dim_output': self.dim_output,
                         'hyperparams': self.hyperparams}}
        
        if not os.path.exists(file_name):
            os.makedirs(path) 
            
        pickle.dump(data, open(file_name, 'wb'))
        
    
if __name__ == '__main__':
    model = MultiLayerNet(2, 3)
    model.backward(np.array([[1, 2]]), np.array([[0, 1, 2]]))