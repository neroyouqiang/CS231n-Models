import numpy as np

from models.MultiLayerNet import MultiLayerNet
from models.layers import Linear, ReLU, Conv, MaxPool, Spatial_BatchNorm, Spatial_GroupNorm
from models.losses import Softmax, SVM


class MultiLayerConvNet(MultiLayerNet):
    """
    Structure:
        [batchnorm-relu-conv] x N -> [affine] x M -> [softmax or SVM]
    """
    def init(self):
        if self.seed: np.random.seed(self.seed)
        
        self.hyperparams.setdefault('loss_type', 'softmax')
        self.hyperparams.setdefault('nums_conv', [32])
        self.hyperparams.setdefault('nums_hidden', [100])
            
        # init layers
        Ci, Hi, Wi = self.dim_input
        self.layers = []
        
        ci = Ci
        for co in self.hyperparams['nums_conv']:
            self.layers.append(Spatial_BatchNorm(ci))
            self.layers.append(Conv(ci, co, 3, 3, S=1, P=1, init_scale=self.init_scale))
            self.layers.append(ReLU())
            self.layers.append(MaxPool(2, 2, S=2))
            ci = co
        
        ni = ci * (Hi // 2 ** len(self.hyperparams['nums_conv'])) * (Wi // 2 ** len(self.hyperparams['nums_conv']))
        for no in self.hyperparams['nums_hidden']:
            self.layers.append(Linear(ni, no, init_scale=self.init_scale))
            self.layers.append(ReLU())
            ni = no
            
        self.layers.append(Linear(ni, self.dim_output, init_scale=self.init_scale))
        
#        self.layers = [Spatial_BatchNorm(Ci),
#                       Conv(Ci, 16, 3, 3, S=1, P=1, init_scale=self.init_scale),
#                       ReLU(),
#                       MaxPool(2, 2, S=2),
#                       
#                       Spatial_BatchNorm(16),
#                       Conv(16, 32, 3, 3, S=1, P=1, init_scale=self.init_scale),
#                       ReLU(),
#                       MaxPool(2, 2, S=2),
#                       
#                       Linear(32 * (Hi // 4) * (Wi // 4), 100, init_scale=self.init_scale),
#                       ReLU(),
#                       Linear(100, self.dim_output, init_scale=self.init_scale)]
        
        # init loss
        if self.hyperparams['loss_type'] == 'softmax':
            self.loss = Softmax()
        elif self.hyperparams['loss_type'] == 'svm':
            self.loss = SVM()