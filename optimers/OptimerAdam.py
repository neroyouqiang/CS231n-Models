#import minpy.numpy as np
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd

from optimers.OptimerSGD import OptimerSGD

class OptimerAdam(OptimerSGD):
    def __init__(self, hyperparams={}, 
                 print_every=100, record_every=10, silence=False,
                 check_val_acc=True, check_train_acc=True):
        
        self._base_init(hyperparams, print_every, record_every, 
                        silence, check_val_acc, check_train_acc)
        
        hyperparams.setdefault('beta1', 0.9)
        hyperparams.setdefault('beta2', 0.999)
        hyperparams.setdefault('epsilon', 1e-8)
        
        self.beta1 = hyperparams['beta1']
        self.beta2 = hyperparams['beta2']
        self.epsilon = hyperparams['epsilon']
        
    
    def train(self, model, dataloader, print_time=False):
        self._init_scale(model)
        self._base_train(model, dataloader, print_time=print_time)
        
    
    def _init_scale(self, model):
        l = len(model.params)
        self.moment1 = [{} for _ in range(l)]
        self.moment2 = [{} for _ in range(l)]
        self.t = 1
        for layer in range(l):
            for key in model.params[layer]:
                if key not in ['cache', 'info']:
#                    self.moment1[layer][key] = np.zeros(model.params[layer][key].shape)
#                    self.moment2[layer][key] = np.zeros(model.params[layer][key].shape)
                    
                    if self._model_device(model) in ['gpu' , '']:
                        if model.device == 'gpu':
                            dvc = mx.gpu()
                        else:
                            dvc = mx.cpu()
                        self.moment1[layer][key] = nd.zeros(model.params[layer][key].shape, dvc)
                        self.moment2[layer][key] = nd.zeros(model.params[layer][key].shape, dvc)
                    else:
                        self.moment1[layer][key] = np.zeros(model.params[layer][key].shape)
                        self.moment2[layer][key] = np.zeros(model.params[layer][key].shape)
                
                
    def _step(self, param, dparam, layer, key):
        self.moment1[layer][key] = self.beta1 * self.moment1[layer][key] + (1 - self.beta1) * dparam
        self.moment2[layer][key] = self.beta2 * self.moment2[layer][key] + (1 - self.beta2) * (dparam * dparam)
        self.mom1_unbias = self.moment1[layer][key] / (1 - self.beta1 ** self.t)
        self.mom2_unbias = self.moment2[layer][key] / (1 - self.beta2 ** self.t)
        self.t += 1
        
        if type(param) == np.ndarray:
            return param - self.learn_rate * self.mom1_unbias / (np.sqrt(self.mom2_unbias) + self.epsilon)
        elif type(param) == nd.ndarray.NDArray:
            return param - self.learn_rate * self.mom1_unbias / (nd.sqrt(self.mom2_unbias) + self.epsilon)