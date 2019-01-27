#import minpy.numpy as np
import numpy as np
import mxnet.ndarray as nd
import pickle
import os
import time

from models.layers import Linear, ReLU, DropOut, BatchNorm
from models.losses import Softmax, SVM

#from layers import Linear, ReLU
#from losses import Softmax, SVM


class MultiLayerNet:
    def __init__(self, dim_input, dim_output, hyperparams={}, 
                 seed=None, params=None, device='cpu'):
        # hyperparameters
        self.hyperparams = hyperparams
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.seed = seed
        self.device = device
            
        self.hyperparams.setdefault('reg', 0.)
        self.hyperparams.setdefault('init_scale', 1e-4)
            
        self.reg = self.hyperparams['reg']
        self.init_scale = self.hyperparams['init_scale']
        
        # random seed
        if self.seed is not None: 
            np.random.seed(self.seed)
#            mx.random.seed(self.seed)
        else:
            np.random.seed()
            
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
        self.hyperparams.setdefault('nums_hidden', [])
        self.hyperparams.setdefault('loss_type', 'softmax')
        self.hyperparams.setdefault('dropout', None)
        self.hyperparams.setdefault('batchnorm', False)
            
        # init layers
        self.layers = []
        
        ni = self.dim_input
        for no in self.hyperparams['nums_hidden']:
            self.layers.append(Linear(ni, no, init_scale=self.init_scale, device=self.device))
            self.layers.append(ReLU(device=self.device))
            if self.hyperparams['dropout']:
                self.layers.append(DropOut(p=self.hyperparams['dropout'], device=self.device))
            if self.hyperparams['batchnorm']:
                self.layers.append(BatchNorm(no, device=self.device))
            ni = no
            
        self.layers.append(Linear(ni, self.dim_output, init_scale=self.init_scale, device=self.device))
        
        # init loss
        if self.hyperparams['loss_type'] == 'softmax':
            self.loss = Softmax(device=self.device)
        elif self.hyperparams['loss_type'] == 'svm':
            self.loss = SVM(device=self.device)
            
    
    def predict(self, x, mode='test', seed=None, print_time=False):
        # the random seed
        if seed is not None: 
            np.random.seed(seed)
        else:
            np.random.seed()
        
        # init caches
        self.caches = [{} for _ in range(len(self.layers))]
        
        # time recorder
        tfs = [time.time()]
        
        # calculate scores
        for i in range(len(self.layers)):
            x, self.caches[i] = self.layers[i].forward(x, self.params[i], mode=mode)
            tfs.append(time.time())
        
        # print running time
        if print_time: self._print_time_forward(tfs)
            
        # return
        return x
    
    
    def backward(self, x, y, mode='train', seed=None, print_time=False):
        # the random seed
        if seed is not None: 
            np.random.seed(seed)
        else:
            np.random.seed()
            
        # calculate loss
        scores = self.predict(x, mode=mode, seed=np.random.randint(10000), 
                              print_time=print_time)
        
        # init dparams
        if mode == 'train': 
            self.dparams = [{} for _ in range(len(self.layers))]
            
        # time recorder
        tbs = [time.time()]
        
        # calculate grediant
        loss, dy = self.loss.backward(scores, y)
        tbs.append(time.time())
        
        if mode == 'train': 
            for i in range(len(self.layers) - 1, -1, -1):
                dy, self.dparams[i] = self.layers[i].backward(dy, self.caches[i])
                tbs.append(time.time())
        
        # time recorder
        trs = [time.time()]
        
        # regularization
        loss, self.dparams = self.loss.add_reg(loss, self.params, self.dparams)
        trs.append(time.time())
        
        # print running time
        if print_time: self._print_time_backward(tbs, trs)
        
        # loss will always be numpy
        if type(loss) == nd.ndarray.NDArray:
            loss = loss.asnumpy()[0]
        
        # return 
        return loss
    
    
    def _print_time_forward(self, tfs):
        print('\nForward time:', tfs[-1] - tfs[0])
        for i in range(len(self.layers)):
            print('    ', self.layers[i].__class__.__name__, ':', tfs[i + 1] - tfs[i])
            
    
    def _print_time_backward(self, tbs, trs):
        print('\nBackward time:', tbs[-1] - tbs[0])
        for i in range(len(self.layers) - 1, -1, -1):
            print('    ', self.layers[i].__class__.__name__, ':', tbs[i + 1] - tbs[i])
        
        print('\nReg time:', trs[-1] - trs[0])
        
    
    def save(self, file_name):
        data = {'params': self.params,
                'info': {'seed': self.seed,
                         'device': self.device,
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
        device = data['info']['device']
        params = data['params']
        
        return cls(dim_input, dim_output, 
                   hyperparams=hyperparams, params=params, 
                   seed=seed, device=device)
        
    
if __name__ == '__main__':
    model = MultiLayerNet(2, 3)
    model.backward(np.array([[1, 2]]), np.array([[0, 1, 2]]))