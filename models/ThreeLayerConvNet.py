import numpy as np

from models.MultiLayerNet import MultiLayerNet
from models.layers import Linear, ReLU, Conv, MaxPool, ConvNaive, MaxPoolNaive
from models.losses import Softmax, SVM


class ThreeLayerConvNet(MultiLayerNet):
    """
    Structure:
        conv - relu - 2x2 max pool - affine - relu - affine - softmax or SVM
    """
    def init(self):
        if self.seed: np.random.seed(self.seed)
        
        self.hyperparams.setdefault('loss_type', 'softmax')
        self.hyperparams.setdefault('filter_size', 7)
        self.hyperparams.setdefault('num_filter', 32)
        self.hyperparams.setdefault('num_hidden', 100)
        
        filter_size = self.hyperparams['filter_size']
        num_filter = self.hyperparams['num_filter']
        num_hidden = self.hyperparams['num_hidden']
            
        # init layers
        Ci, Hi, Wi = self.dim_input
        self.layers = [Conv(Ci, num_filter, filter_size, filter_size, 
                            S=1, P=(filter_size - 1) // 2, 
                            init_scale=self.init_scale),
                       ReLU(),
                       MaxPool(2, 2, S=2),
                       Linear(num_filter * (Hi // 2) * (Wi // 2), num_hidden, 
                              init_scale=self.init_scale),
                       ReLU(),
                       Linear(num_hidden, self.dim_output, 
                              init_scale=self.init_scale)]
        
        # init loss
        if self.hyperparams['loss_type'] == 'softmax':
            self.loss = Softmax()
        elif self.hyperparams['loss_type'] == 'svm':
            self.loss = SVM()