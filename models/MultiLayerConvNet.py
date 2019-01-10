import numpy as np

from models.MultiLayerNet import MultiLayerNet
from models.layers import Linear, ReLU, Conv, MaxPool, Spatial_BatchNorm, Spatial_GroupNorm
from models.losses import Softmax, SVM


class MultiLayerConvNet(MultiLayerNet):
    """
    Structure:
        norm - conv - relu - 2x2 max pool - affine - relu - affine - softmax
    """
    def __init__(self, dim_input=None, dim_output=None, hyperparams={}, seed=None):
        if dim_input:
            hyperparams.setdefault('reg', 0.)
            hyperparams.setdefault('init_scale', 1e-4)
            hyperparams.setdefault('loss_type', 'softmax')
            
            self.hyperparams = hyperparams
            self.dim_input = dim_input
            self.dim_output = dim_output
            self.seed = seed
            
            self.init()
            
            
    def init(self, params=None):
        if self.seed: np.random.seed(self.seed)
        
        self.reg = self.hyperparams['reg']
        self.loss_type = self.hyperparams['loss_type']
        self.init_scale = self.hyperparams['init_scale']
            
        # init layers
        Ci, Hi, Wi = self.dim_input
        self.layers = [Spatial_BatchNorm(Ci),
                       Conv(Ci, 16, 3, 3, S=1, P=1, init_scale=self.init_scale),
                       ReLU(),
                       MaxPool(2, 2, S=2),
                       
                       Spatial_BatchNorm(16),
                       Conv(16, 32, 3, 3, S=1, P=1, init_scale=self.init_scale),
                       ReLU(),
                       MaxPool(2, 2, S=2),
                       
                       Linear(32 * (Hi // 4) * (Wi // 4), 100, init_scale=self.init_scale),
                       ReLU(),
                       Linear(100, self.dim_output, init_scale=self.init_scale)]
        
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