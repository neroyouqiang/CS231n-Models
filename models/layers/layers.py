import numpy as np

class Linear:
    def __init__(self, num_input, num_output, init_scale=None):
        self.num_input = num_input
        self.num_output = num_output
        if init_scale:
            self.init_scale = init_scale
        else:
            self.init_scale = 1. / np.sqrt(self.num_input / 2.)
    
    def forward(self, x, param, mode='train'):
        x = x.reshape(x.shape[0], - 1)
        
        W, b = param['W'], param['b']
        y = np.dot(x, W) + b
        
        cache = (x, W)
        
        return y, cache
    
    def backward(self, dy, cache):
        dy = dy.reshape(dy.shape[0], -1)
        
        x, W = cache
        N, D = x.shape
        
        db = np.sum(dy, axis=0)
        dW = np.dot(x.T, dy)
        dx = np.dot(dy, W.T)
        
        dparam = {'W': dW, 'b': db}
        
        return dx, dparam
    
    def get_init_param(self):
        W = np.random.randn(self.num_input, self.num_output) * self.init_scale
        b = np.zeros(self.num_output)
        
        return {'W': W, 'b': b,
                'info': {'num_input': self.num_input, 
                         'num_output': self.num_output,
                         'init_scale': self.init_scale}}


class ReLU:
    def __init__(self):
        pass
        
    def forward(self, x, param, mode='train'):
        mask = x > 0
        y = np.zeros(x.shape)
        y[mask] = x[mask]
        
        cache = (mask)
        
        return y, cache
    
    def backward(self, dy, cache):
        mask = cache
        dy = dy.reshape(mask.shape)
        dx = np.zeros(mask.shape)
        dx[mask] = dy[mask]
        
        return dx, {}
    
    def get_init_param(self):
        return {'info': {}}


class DropOut:
    def __init__(self, p=0.9):
        self.p = p
        
    def forward(self, x, param, mode='train'):
        if mode == 'train':
            mask = np.random.random(x.shape) < self.p
            y = x * mask / self.p
        
            cache = (mask, self.p)
            
        elif mode == 'test':
            mask = np.ones(x.shape).astype(np.bool)
            y = x
        
            cache = (mask, 1)
        
        return y, cache
    
    def backward(self, dy, cache):
        mask, p = cache
        dx = dy * mask / p
        
        return dx, {}
    
    def get_init_param(self):
        return {'info': {'p': self.p}}


class BatchNorm:
    def __init__(self, num_input, momentum=0.9, eps=1e-5):
        self.C = num_input
        self.momentum = momentum
        self.eps = eps
        
    def forward(self, x, param, mode='train'):
        x_norm, cache = self._forward_mean_var(x, param, mode)
        mean, var, eps = cache
        
        y, cache = self._forward_gamma_beta(x_norm, param)
        gamma, beta = cache
        
        return y, (x, x_norm, mean, var, eps, gamma, beta)
    
    def backward(self, dy, cache):
        x, x_norm, mean, var, eps, gamma, beta = cache
        N, D = x.shape
        
        dx_norm, dgamma, dbeta = self._backward_gamma_beta(dy, (x_norm, gamma, beta))
        
        dx = self._backward_mean_var(dx_norm, (x, x_norm, mean, var, eps))
        
        return dx, {'gamma': dgamma, 'beta': dbeta}
        
    def get_init_param(self):
        gamma = np.ones(self.C)
        beta = np.zeros(self.C)
        
        return {'gamma': gamma, 'beta': beta,
                'cache': {'running_mean': 0, 
                          'running_var': 0}, 
                'info': {'momentum': self.momentum, 
                         'eps': self.eps, 
                         'C': self.C}}
    
    def _forward_mean_var(self, x, param, mode):
        if mode == 'train':
            sample_mean = np.mean(x, axis=0)
            sample_var = np.var(x, axis=0)
            
            x_norm = (x - sample_mean) / np.sqrt(sample_var + self.eps)
            
            cache = (sample_mean, sample_var, self.eps)
            
            param['cache']['running_mean'] = self.momentum * param['cache']['running_mean'] + (1 - self.momentum) * sample_mean
            param['cache']['running_var'] = self.momentum * param['cache']['running_var'] + (1 - self.momentum) * sample_var
            
        elif mode == 'test':
            running_mean = param['cache']['running_mean']
            running_var = param['cache']['running_var']
            
            x_norm = (x - running_mean) / np.sqrt(running_var + self.eps)
            cache = (running_mean, running_var, self.eps)
            
        return x_norm, cache
    
    def _backward_mean_var(self, dx_norm, cache):
        x, x_norm, mean, var, eps = cache
        
        N = x.shape[0]
        
        dmean = np.sum(dx_norm * -1.0 / np.sqrt(var + eps), axis=0)
        dvar = np.sum(dx_norm * -0.5 * x_norm / (var + eps), axis=0)
        dx = dx_norm / np.sqrt(var + eps) + dmean / N + dvar * (x - mean) * 2.0 / N
        
        return dx
    
    def _forward_gamma_beta(self, x_norm, param):
        gamma = param['gamma']
        beta = param['beta']
        
        y = x_norm * gamma + beta
        
        cache = (gamma, beta)
        
        return y, cache
    
    def _backward_gamma_beta(self, dy, cache):
        x_norm, gamma, beta = cache
        
        dgamma = np.sum(dy * x_norm, axis=0)
        dbeta = np.sum(dy, axis=0)
        dx_norm = dy * gamma
        
        return dx_norm, dgamma, dbeta