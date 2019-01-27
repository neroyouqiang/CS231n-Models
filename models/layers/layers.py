#import minpy.numpy as np
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd

class Layer:
    """
    Parent layer class. The layer can be used on both CPU and GPU.
    """
    def __init__(self, device='cpu'):
        self.device = device
            
    def forward(self, x, param, mode='train'):
        # Use numpy on CPU
        if self.device == 'cpu':
            # to numpy
            if type(x) == nd.ndarray.NDArray: 
                x = x.asnumpy()
                
            for key in param:
                if type(param[key]) == nd.ndarray.NDArray: 
                    param[key]= param[key].asnumpy()
                
            # function
            if mode == 'train':
                y, cache = self._forward_train(x, param)
                              
            elif mode == 'test':
                y, cache = self._forward_test(x, param)
            
        # Use mxnet.ndarray on GPU
        elif self.device == 'gpu' or self.device == '':
            if self.device == 'gpu':
                dvc = mx.gpu()
            else:
                dvc = mx.cpu()
                    
            # to ndarray
            if type(x) == np.ndarray: 
                x = nd.array(x, dvc)
            
            for key in param:
                if type(param[key]) == np.ndarray: 
                    param[key]= nd.array(param[key], dvc)
                
            # function
            if mode == 'train':
                y, cache = self._forward_train_mx(x, param, dvc)
                              
            elif mode == 'test':
                y, cache = self._forward_test_mx(x, param, dvc)
                
            # to numpy
            y.wait_to_read()
#            y = y.asnumpy()
        
        return y, cache
    
    def backward(self, dy, cache):
        # Use numpy on CPU
        if self.device == 'cpu':
            # to numpy
            if type(dy) != np.ndarray: dy = dy.asnumpy()
            
            # function
            dx, dparam = self._backward(dy, cache)
            
        # Use mxnet.ndarray on GPU
        elif self.device == 'gpu' or self.device == '':
            if self.device == 'gpu':
                dvc = mx.gpu()
            else:
                dvc = mx.cpu()
                    
            # to ndarray
            if type(dy) != nd.ndarray.NDArray: dy = nd.array(dy, dvc)
            
            # function
            dx, dparam = self._backward_mx(dy, cache, dvc)
            
            # to numpy
            dx.wait_to_read()
#            dx = dx.asnumpy()
            
        return dx, dparam
        
    def get_init_param(self):
        param = self._get_init_param()
        
        if self.device == 'cpu':
            for key in param:
                if type(param[key]) == nd.ndarray.NDArray:
                    param[key] = param[key].asnumpy()
            
        elif self.device == 'gpu' or self.device == '':
            if self.device == 'gpu':
                dvc = mx.gpu()
            else:
                dvc = mx.cpu()
                
            for key in param:
                if type(param[key]) == np.ndarray:
                    param[key] = nd.array(param[key], dvc)
        
        return param
    
    def _forward(self, x, param):
        y = x
        cache = ()
        return y, cache

    def _forward_mx(self, x, param, device):
        return self._forward(x, param)

    def _forward_train(self, x, param):
        return self._forward(x, param)
    
    def _forward_test(self, x, param):
        return self._forward(x, param)
    
    def _forward_train_mx(self, x, param, device):
        return self._forward_mx(x, param, device)
    
    def _forward_test_mx(self, x, param, device):
        return self._forward_mx(x, param, device)
    
    def _backward(self, dy, cache):
        dx = dy
        dparam = {}
        return dx, dparam
    
    def _backward_mx(self, dy, cache, device):
        return self._backward(dy, cache)
    
    def _get_init_param(self):
        param = {}
        return param


class Linear(Layer):
    def __init__(self, num_input, num_output, init_scale=None, device='cpu'):
        self.num_input = num_input
        self.num_output = num_output
        self.device = device
        if init_scale:
            self.init_scale = init_scale
        else:
            self.init_scale = 1. / np.sqrt(self.num_input / 2.)

    def _forward(self, x, param):
        x = x.reshape(x.shape[0], - 1)
        
        W, b = param['W'], param['b']
        y = np.dot(x, W) + b
        
        cache = (x, W)
        
        return y, cache

    def _forward_mx(self, x, param, device):
        x = x.reshape(x.shape[0], - 1)
        
        W, b = param['W'], param['b']
        y = nd.dot(x, W) + b
        
        cache = (x, W)
        
        return y, cache
    
    def _backward(self, dy, cache):
        dy = dy.reshape(dy.shape[0], -1)
        
        x, W = cache
        N, D = x.shape
        
        db = np.sum(dy, axis=0)
        dW = np.dot(x.T, dy)
        dx = np.dot(dy, W.T)
        
        dparam = {'W': dW, 'b': db}
        
        return dx, dparam
    
    def _backward_mx(self, dy, cache, device):
        dy = dy.reshape(dy.shape[0], -1)
        
        x, W = cache
        N, D = x.shape
        
        db = nd.sum(dy, axis=0)
        dW = nd.dot(x.T, dy)
        dx = nd.dot(dy, W.T)
        
        dparam = {'W': dW, 'b': db}
        
        return dx, dparam
    
    def _get_init_param(self):
        W = np.random.randn(self.num_input, self.num_output) * self.init_scale
        b = np.zeros(self.num_output)
        
        return {'W': W, 'b': b}
#                'info': {'num_input': self.num_input, 
#                         'num_output': self.num_output,
#                         'init_scale': self.init_scale}}


class ReLU(Layer):
    def _forward(self, x, param):
        mask = x > 0
        y = np.zeros(x.shape)
        y[mask] = x[mask]
        
        cache = (mask)
        
        return y, cache
    
    def _forward_mx(self, x, param, device):
        mask = x > 0
        y = x * mask
        
        cache = (mask)
        
        return y, cache
    
    def _backward(self, dy, cache):
        mask = cache
        dy = dy.reshape(mask.shape)
        dx = np.zeros(dy.shape)
        dx[mask] = dy[mask]
        
        return dx, {}
    
    def _backward_mx(self, dy, cache, device):
        mask = cache
        dy = dy.reshape(mask.shape)
        dx = dy * mask
        
        return dx, {}
    
#    def _get_init_param(self):
#        return {'info': {}}


class DropOut(Layer):
    def __init__(self, p=0.9, device='cpu'):
        self.p = p
        self.device = device
        
    def _forward_train(self, x, param):
        mask = np.random.random(x.shape) < self.p
        y = x * mask / self.p
        
        cache = (mask, self.p)
        
        return y, cache
        
    def _forward_test(self, x, param):
        mask = np.ones(x.shape).astype(np.bool)
        y = x
        
        cache = (mask, 1)
        
        return y, cache
        
    def _forward_train_mx(self, x, param, device):
        mask = nd.random.uniform(0, 1, shape=x.shape, ctx=device) < self.p
#        mask = mx.random.random(x.shape) < self.p
        y = x * mask / self.p
        
        cache = (mask, self.p)
        
        return y, cache
        
    def _forward_test_mx(self, x, param, device):
        mask = nd.ones(x.shape, device).astype(np.bool)
        y = x
        
        cache = (mask, 1)
        
        return y, cache
    
    def _backward(self, dy, cache):
        mask, p = cache
        dx = dy * mask / p
        
        return dx, {}
    
    def _backward_mx(self, dy, cache, device):
        mask, p = cache
        dx = dy * mask / p
        
        return dx, {}
    
#    def _get_init_param(self):
#        return {'info': {'p': self.p}}


class BatchNorm(Layer):
    def __init__(self, num_input, momentum=0.9, eps=1e-5, device='cpu'):
        self.C = num_input
        self.momentum = momentum
        self.eps = eps
        self.device = device
        
    def _forward_train(self, x, param):
        x_norm, cache = self.forward_mean_var(x, param, mode='train')
        mean, var, eps = cache
        
        y, cache = self.forward_gamma_beta(x_norm, param)
        gamma, beta = cache
        
        cache = (x, x_norm, mean, var, eps, gamma, beta)
        
        return y, cache
    
    def _forward_test(self, x, param):
        x_norm, cache = self.forward_mean_var(x, param, mode='test')
        mean, var, eps = cache
        
        y, cache = self.forward_gamma_beta(x_norm, param)
        gamma, beta = cache
        
        cache = (x, x_norm, mean, var, eps, gamma, beta)
        
        return y, cache
        
    def _forward_train_mx(self, x, param, device):
        x_norm, cache = self.forward_mean_var_mx(x, param, mode='train', device=device)
        mean, var, eps = cache
        
        y, cache = self.forward_gamma_beta_mx(x_norm, param, device)
        gamma, beta = cache
        
        cache = (x, x_norm, mean, var, eps, gamma, beta)
        
        return y, cache
    
    def _forward_test_mx(self, x, param, device):
        x_norm, cache = self.forward_mean_var_mx(x, param, mode='test', device=device)
        mean, var, eps = cache
        
        y, cache = self.forward_gamma_beta_mx(x_norm, param, device)
        gamma, beta = cache
        
        cache = (x, x_norm, mean, var, eps, gamma, beta)
        
        return y, cache
    
    def _backward(self, dy, cache):
        x, x_norm, mean, var, eps, gamma, beta = cache
        N, D = x.shape
        
        dx_norm, dgamma, dbeta = self.backward_gamma_beta(dy, (x_norm, gamma, beta))
        
        dx = self.backward_mean_var(dx_norm, (x, x_norm, mean, var, eps))
        
        return dx, {'gamma': dgamma, 'beta': dbeta}
    
    def _backward_mx(self, dy, cache, device):
        x, x_norm, mean, var, eps, gamma, beta = cache
        N, D = x.shape
        
        dx_norm, dgamma, dbeta = self.backward_gamma_beta_mx(dy, (x_norm, gamma, beta), device)
        
        dx = self.backward_mean_var_mx(dx_norm, (x, x_norm, mean, var, eps), device)
        
        return dx, {'gamma': dgamma, 'beta': dbeta}
        
    def _get_init_param(self):
        gamma = np.ones(self.C)
        beta = np.zeros(self.C)
        
        return {'gamma': gamma, 'beta': beta,
                'cache': {'running_mean': 0, 
                          'running_var': 0}}
#                'info': {'momentum': self.momentum, 
#                         'eps': self.eps, 
#                         'C': self.C}}
    
    def forward_mean_var(self, x, param, mode):
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
    
    def backward_mean_var(self, dx_norm, cache):
        x, x_norm, mean, var, eps = cache
        
        N = x.shape[0]
        
        dmean = np.sum(dx_norm * -1.0 / np.sqrt(var + eps), axis=0)
        dvar = np.sum(dx_norm * -0.5 * x_norm / (var + eps), axis=0)
        dx = dx_norm / np.sqrt(var + eps) + dmean / N + dvar * (x - mean) * 2.0 / N
        
        return dx
    
    def forward_mean_var_mx(self, x, param, mode, device):
        if mode == 'train':
            sample_mean = nd.mean(x, axis=0)
            x_no_mean = x - sample_mean
            sample_var = nd.mean(nd.square(x_no_mean), axis=0)
            
            x_norm = x_no_mean / nd.sqrt(sample_var + self.eps)
            
            cache = (sample_mean, sample_var, self.eps)
            
            param['cache']['running_mean'] = self.momentum * param['cache']['running_mean'] + (1 - self.momentum) * sample_mean
            param['cache']['running_var'] = self.momentum * param['cache']['running_var'] + (1 - self.momentum) * sample_var
            
        elif mode == 'test':
            running_mean = param['cache']['running_mean']
            running_var = param['cache']['running_var']
            
            x_norm = (x - running_mean) / nd.sqrt(running_var + self.eps)
            
            cache = (running_mean, running_var, self.eps)
            
        return x_norm, cache
    
    def backward_mean_var_mx(self, dx_norm, cache, device):
        x, x_norm, mean, var, eps = cache
        
        N = x.shape[0]
        
        dmean = nd.sum(dx_norm * -1.0 / nd.sqrt(var + eps), axis=0)
        dvar = nd.sum(dx_norm * -0.5 * x_norm / (var + eps), axis=0)
        dx = dx_norm / nd.sqrt(var + eps) + dmean / N + dvar * (x - mean) * 2.0 / N
        
        return dx
    
    def forward_gamma_beta(self, x_norm, param):
        gamma = param['gamma']
        beta = param['beta']
        
        y = x_norm * gamma + beta
        
        cache = (gamma, beta)
        
        return y, cache
    
    def forward_gamma_beta_mx(self, x_norm, param, device):
        gamma = param['gamma']
        beta = param['beta']
        
        y = x_norm * gamma + beta
        
        cache = (gamma, beta)
        
        return y, cache
    
    def backward_gamma_beta(self, dy, cache):
        x_norm, gamma, beta = cache
        
        dgamma = np.sum(dy * x_norm, axis=0)
        dbeta = np.sum(dy, axis=0)
        dx_norm = dy * gamma
        
        return dx_norm, dgamma, dbeta
    
    
    def backward_gamma_beta_mx(self, dy, cache, device):
        x_norm, gamma, beta = cache
        
        dgamma = nd.sum(dy * x_norm, axis=0)
        dbeta = nd.sum(dy, axis=0)
        dx_norm = dy * gamma
        
        return dx_norm, dgamma, dbeta