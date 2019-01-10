import numpy as np

from optimers.OptimerSGD import OptimerSGD

class OptimerRMSProp(OptimerSGD):
    def __init__(self, hyperparams={}, 
                 print_every=100, silence=False,
                 check_val_acc=True, check_train_acc=True):
        
        self._base_init(hyperparams, print_every, silence, 
                        check_val_acc, check_train_acc)
        
        hyperparams.setdefault('scale_decay', 0.99)
        hyperparams.setdefault('epsilon', 1e-8)
        
        self.scale_decay = hyperparams['scale_decay']
        self.epsilon = hyperparams['epsilon']
        
    
    def train(self, model, dataloader):
        self._init_scale(model)
        self._base_train(model, dataloader)
        
    
    def _init_scale(self, model):
        l = len(model.params)
        self.scale = [{} for _ in range(l)]
        for layer in range(l):
            for key in model.params[layer].keys():
                if key not in ['cache', 'info']:
                    self.scale[layer][key] = np.zeros(model.params[layer][key].shape)
                
                
    def _step(self, param, dparam, layer, key):
        self.scale[layer][key] = self.scale_decay * self.scale[layer][key] + (1 - self.scale_decay) * (dparam * dparam)
        return param - self.learn_rate * self.scale[layer][key] / (np.sqrt(self.scale[layer][key]) + self.epsilon)