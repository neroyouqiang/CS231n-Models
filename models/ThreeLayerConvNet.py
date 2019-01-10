import numpy as np

from models.MultiLayerNet import MultiLayerNet
from models.layers import Linear, ReLU, Conv, MaxPool, ConvNaive, MaxPoolNaive
from models.losses import Softmax, SVM


class ThreeLayerConvNet(MultiLayerNet):
    """
    Structure:
        conv - relu - 2x2 max pool - affine - relu - affine - softmax
    """
    def __init__(self, dim_input=None, dim_output=None, hyperparams={}, seed=None):
        if dim_input:
            hyperparams.setdefault('reg', 0.)
            hyperparams.setdefault('init_scale', 1e-4)
            hyperparams.setdefault('loss_type', 'softmax')
            hyperparams.setdefault('filter_size', 7)
            hyperparams.setdefault('num_filter', 32)
            hyperparams.setdefault('num_hidden', 100)
            
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
        self.filter_size = self.hyperparams['filter_size']
        self.num_filter = self.hyperparams['num_filter']
        self.num_hidden = self.hyperparams['num_hidden']
            
        # init layers
        Ci, Hi, Wi = self.dim_input
        self.layers = [Conv(Ci, self.num_filter, self.filter_size, self.filter_size, 
                            S=1, P=(self.filter_size - 1) // 2, 
                            init_scale=self.init_scale),
                       ReLU(),
                       MaxPool(2, 2, S=2),
                       Linear(self.num_filter * (Hi // 2) * (Wi // 2), self.num_hidden, 
                              init_scale=self.init_scale),
                       ReLU(),
                       Linear(self.num_hidden, self.dim_output, 
                              init_scale=self.init_scale)]
        
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