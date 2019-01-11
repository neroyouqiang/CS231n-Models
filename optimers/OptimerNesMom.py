import numpy as np

from optimers.OptimerSGD import OptimerSGD

class OptimerNesMom(OptimerSGD):
    def __init__(self, hyperparams={}, 
                 print_every=100, record_every=10, silence=False,
                 check_val_acc=True, check_train_acc=True):
        
        self._base_init(hyperparams, print_every, record_every, 
                        silence, check_val_acc, check_train_acc)
        
        hyperparams.setdefault('momentum', 0.9)
        self.momentum = hyperparams['momentum']
    
    
    def train(self, model, dataloader):
        self._init_velocity(model)
        self._base_train(model, dataloader)
    
    
    def _init_velocity(self, model):
        l = len(model.params)
        self.velocity = [{} for _ in range(l)]
        for layer in range(l):
            for key in model.params[layer].keys():
                if key not in ['cache', 'info']:
                    self.velocity[layer][key] = np.zeros(model.params[layer][key].shape)
        
        
    def _step(self, param, dparam, layer, key):
        vt = self.velocity[layer][key].copy()
        self.velocity[layer][key] = self.momentum * self.velocity[layer][key] - self.learn_rate * dparam
        return param + self.velocity[layer][key] + self.momentum * (self.velocity[layer][key] - vt)
        # param - self.momentum * vt + (1 + self.momentum) * self.velocity[layer][key]