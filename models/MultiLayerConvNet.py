from models.MultiLayerNet import MultiLayerNet
from models.layers import Linear, ReLU, Conv, MaxPool, Spatial_BatchNorm, Spatial_GroupNorm
from models.losses import Softmax, SVM


class MultiLayerConvNet(MultiLayerNet):
    """
    Structure:
        [batchnorm-relu-conv] x N -> [affine] x M -> [softmax or SVM]
    """
    def init(self):
        self.hyperparams.setdefault('loss_type', 'softmax')
        self.hyperparams.setdefault('nums_conv', [32])
        self.hyperparams.setdefault('nums_hidden', [100])
            
        # init layers
        Ci, Hi, Wi = self.dim_input
        self.layers = []
        
        ci = Ci
        for co in self.hyperparams['nums_conv']:
            self.layers.append(Spatial_BatchNorm(ci, device=self.device))
            self.layers.append(Conv(ci, co, 3, 3, S=1, P=1, init_scale=self.init_scale, device=self.device))
            self.layers.append(ReLU(device=self.device))
            self.layers.append(MaxPool(2, 2, S=2, device=self.device))
            ci = co
        
        ni = ci * (Hi // 2 ** len(self.hyperparams['nums_conv'])) * (Wi // 2 ** len(self.hyperparams['nums_conv']))
        for no in self.hyperparams['nums_hidden']:
            self.layers.append(Linear(ni, no, init_scale=self.init_scale, device=self.device))
            self.layers.append(ReLU(device=self.device))
            ni = no
            
        self.layers.append(Linear(ni, self.dim_output, init_scale=self.init_scale, device=self.device))
        
        # init loss
        if self.hyperparams['loss_type'] == 'softmax':
            self.loss = Softmax(device=self.device)
        elif self.hyperparams['loss_type'] == 'svm':
            self.loss = SVM(device=self.device)