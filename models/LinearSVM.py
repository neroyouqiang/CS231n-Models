from models.layers import Linear
from models.losses import SVM
from models.LinearSoftmax import LinearSoftmax


class LinearSVM(LinearSoftmax):
    def __init__(self, num_input, num_output, hyperparams={}):
        
        hyperparams.setdefault('reg', 1e-5)
        self.reg = hyperparams['reg']
        
        self.layers = [Linear(num_input, num_output, init_scale=0.0001)]
        self.loss = SVM()
        
        self.params = [self.layers[0].get_init_param()]