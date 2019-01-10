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
    
    
class Conv:
    def __init__(self, Ci, Co, Hf, Wf, S=1, P=0, init_scale=None):
        self.Ci = Ci
        self.Co = Co
        self.Hf = Hf
        self.Wf = Wf
        self.S = S
        self.P = P
        if init_scale:
            self.init_scale = init_scale
        else:
            self.init_scale = 1. / np.sqrt(self.Hf * self.Wf * self.Ci / 2.)
    
    def forward(self, x, param, mode='train'):
        N, Ci, Hi, Wi = x.shape
        Ho = (Hi - self.Hf + 2 * self.P) // self.S + 1
        Wo = (Wi - self.Wf + 2 * self.P) // self.S + 1
        
        W, b = param['W'], param['b']
        
        col_x = self.img2col(x, self.Hf, self.Wf, self.S, self.P)
        col_W = W.reshape([self.Co, self.Ci * self.Hf * self.Wf])
        col_W = col_W.T
        col_y = np.dot(col_x, col_W) + b
        
        y = col_y.reshape([N, Wo, Ho, self.Co])
        y = y.transpose([0, 3, 1, 2])
        
        return y, (col_x, col_W, [N, Ci, Hi, Wi])
    
    def backward(self, dy, cache):
        col_x, col_W, x_shape = cache
        
        N, Ci, Hi, Wi = x_shape
        Ho = (Hi - self.Hf + 2 * self.P) // self.S + 1
        Wo = (Wi - self.Wf + 2 * self.P) // self.S + 1
        
        dy = dy.reshape([N, self.Co, Ho, Wo])
        col_dy = dy.transpose([0, 2, 3, 1])
        col_dy = col_dy.reshape([N * Ho * Wo, self.Co])
        
        db = np.sum(col_dy, axis=0)
        col_dW = np.dot(col_x.T, col_dy)
        col_dW = col_dW.T
        col_dx = np.dot(col_dy, col_W.T)
        
        dW = col_dW.reshape([self.Co, self.Ci, self.Hf, self.Wf])
        dx = self.col2img(col_dx, [N, self.Ci, Hi, Wi], self.Hf, self.Wf, self.S, self.P)
        
        dparam = {'W': dW, 'b': db}
        
        return dx, dparam
    
    def img2col(self, img, Hf, Wf, S, P):
        N, C, Hi, Wi = img.shape
        Ho = (Hi - Hf + 2 * P) // S + 1
        Wo = (Wi - Wf + 2 * P) // S + 1
        
        img = np.pad(img, [(0, 0), (0, 0), (P, P), (P, P)], 'constant')
        
        col = np.zeros([N, C, Ho, Wo, Hf, Wf], dtype=img.dtype)
        for h in range(Hf):
            for w in range(Wf):
                hm = h + Ho * S
                wm = w + Wo * S
                col[:, :, :, :, h, w] = img[:, :, h: hm: S, w: wm: S]
        
        col = col.transpose([0, 2, 3, 1, 4, 5])
        col = col.reshape([N * Ho * Wo, C * Hf * Wf])
        
        return col
    
    def col2img(self, col, img_shape, Wf, Hf, S, P):
        N, C, Hi, Wi = img_shape
        Ho = (Hi - Hf + 2 * P) // S + 1
        Wo = (Wi - Wf + 2 * P) // S + 1
        
        col = col.reshape([N, Ho, Wo, C, Hf, Wf])
        col = col.transpose([0, 3, 1, 2, 4, 5])
        
        img = np.zeros([N, C, Hi + 2 * P, Wi + 2 * P])
        for h in range(Hf):
            for w in range(Wf):
                hm = h + Ho * S
                wm = w + Wo * S
                img[:, :, h: hm: S, w: wm: S] += col[:, :, :, :, h, w]
                
        return img[:, :, P: P + Hi, P: P + Wi]
        
    def get_init_param(self):
        W = np.random.randn(self.Co, self.Ci, self.Wf, self.Hf) * self.init_scale
        b = np.zeros(self.Co)
        
        param = {'W': W, 'b': b,
                 'info': {'Ci': self.Ci,
                          'Co': self.Co,
                          'Hf': self.Hf,
                          'Wf': self.Wf,
                          'S': self.S,
                          'P': self.P,
                          'init_scale': self.init_scale}}
        return param
    
    
class ConvNaive(Conv):
    def forward(self, x, param, mode='train'):
        N, Ci, Hi, Wi = x.shape
        Ho = (Hi - self.Hf + 2 * self.P) // self.S + 1
        Wo = (Wi - self.Wf + 2 * self.P) // self.S + 1
        
        W, b = param['W'], param['b']
        
        x = np.pad(x, [(0, 0), (0, 0), (self.P, self.P), (self.P, self.P)], 'constant')
        y = np.zeros([N, self.Co, Ho, Wo], dtype=x.dtype)
        
        for n in range(N):
            for h in range(Ho):
                for w in range(Wo):
                    for c in range(self.Co):
                        y[n, c, h, w] = np.sum(W[c] * x[n, :, h * self.S: h * self.S + self.Hf, w * self.S: w * self.S + self.Wf]) + b[c]
        
        return y, (x, W, [N, Ci, Hi, Wi])
    
    def backward(self, dy, cache):
        x, W, x_shape = cache
        
        N, Ci, Hi, Wi = x_shape
        Ho = (Hi - self.Hf + 2 * self.P) // self.S + 1
        Wo = (Wi - self.Wf + 2 * self.P) // self.S + 1
        
        dy = dy.reshape([N, self.Co, Ho, Wo])
        
        db = np.zeros(self.Co, dtype=dy.dtype)
        dW = np.zeros(W.shape, dtype=dy.dtype)
        dx = np.zeros(x.shape, dtype=dy.dtype)
        
        for n in range(N):
            for h in range(Ho):
                for w in range(Wo):
                    for c in range(self.Co):
                        db[c] += dy[n, c, h, w]
                        dW[c] += x[n, :, h * self.S: h * self.S + self.Hf, w * self.S: w * self.S + self.Wf] * dy[n, c, h, w]
                        dx[n, :, h * self.S: h * self.S + self.Hf, w * self.S: w * self.S + self.Wf] += W[c] * dy[n, c, h, w]
        
        dx = dx[:, :, self.P: self.P + Hi, self.P: self.P + Wi]
        
        dparam = {'W': dW, 'b': db}
        
        return dx, dparam
    

class MaxPool(Conv):
    def __init__(self, Hf, Wf, S=1, P=0):
        self.Hf = Hf
        self.Wf = Wf
        self.S = S
        self.P = P
    
    def forward(self, x, param, mode='train'):
        N, Ci, Hi, Wi = x.shape
        Ho = (Hi - self.Hf + 2 * self.P) // self.S + 1
        Wo = (Wi - self.Wf + 2 * self.P) // self.S + 1
        
        col_x = self.img2col(x, self.Hf, self.Wf, self.S, self.P)
        col_x = col_x.reshape([-1, self.Hf * self.Wf])
        
        col_i = np.argmax(col_x, axis=1)
        col_y = np.max(col_x, axis=1)
        
        y = col_y.reshape([N, Wo, Ho, Ci])
        y = y.transpose([0, 3, 1, 2])
        
        return y, (col_i, [N, Ci, Hi, Wi])
    
    def backward(self, dy, cache):
        col_i, x_shape = cache
        
        N, Ci, Hi, Wi = x_shape
        Ho = (Hi - self.Hf + 2 * self.P) // self.S + 1
        Wo = (Wi - self.Wf + 2 * self.P) // self.S + 1
        
        dy = dy.reshape([N, Ci, Ho, Wo])
        
        col_dy = dy.transpose([0, 2, 3, 1])
        col_dy = col_dy.reshape([N * Ho * Wo * Ci])
        
        col_dx = np.zeros([col_dy.shape[0], self.Hf * self.Wf])
        col_dx[np.arange(col_dy.shape[0]), col_i] = col_dy

        dx = self.col2img(col_dx, [N, Ci, Hi, Wi], self.Hf, self.Wf, self.S, self.P)
        
        return dx, {}
    
    def get_init_param(self):
        return {'info': {'Hf': self.Hf,
                         'Wf': self.Wf,
                         'S': self.S,
                         'P': self.P}}
      

class MaxPoolNaive(MaxPool):
    def forward(self, x, param, mode='train'):
        N, Ci, Hi, Wi = x.shape
        Ho = (Hi - self.Hf + 2 * self.P) // self.S + 1
        Wo = (Wi - self.Wf + 2 * self.P) // self.S + 1
        
        y = np.zeros([N, Ci, Ho, Wo], dtype=x.dtype)
        for n in range(N):
            for h in range(Ho):
                for w in range(Wo):
                    for c in range(Ci):
                        y[n, c, h, w] = np.max(x[n, c, h * self.S: h * self.S + self.Hf, w * self.S: w * self.S + self.Wf])
        
        return y, (x, [N, Ci, Hi, Wi])
    
    def backward(self, dy, cache):
        x, x_shape = cache
        
        N, Ci, Hi, Wi = x_shape
        Ho = (Hi - self.Hf + 2 * self.P) // self.S + 1
        Wo = (Wi - self.Wf + 2 * self.P) // self.S + 1
        
        dy = dy.reshape([N, Ci, Ho, Wo])
        
        dx = np.zeros(x.shape, dtype=dy.dtype)
        for n in range(N):
            for h in range(Ho):
                for w in range(Wo):
                    for c in range(Ci):
                        win = x[n, c, h * self.S: h * self.S + self.Hf, w * self.S: w * self.S + self.Wf]
                        dx[n, c, h * self.S: h * self.S + self.Hf, w * self.S: w * self.S + self.Wf] += (win == np.max(win)) * dy[n, c, h, w]
        
        dx = dx[:, :, self.P: self.P + Hi, self.P: self.P + Wi]
        
        return dx, {}


class Spatial_BatchNorm(BatchNorm):
    def __init__(self, C, momentum=0.9, eps=1e-5):
        self.C = C
        
        self.momentum = momentum
        self.eps = eps
        
        self.running_mean = 0
        self.running_var = 0
        
    def forward(self, x, param, mode='train'):
        N, C, H, W = x.shape
        x = x.transpose([0, 2, 3, 1]).reshape([N * H * W, C])
        
        x_norm, cache = self._forward_mean_var(x, param, mode)
        mean, var, eps = cache
        
        y, cache = self._forward_gamma_beta(x_norm, param)
        gamma, beta = cache
        
        y = y.reshape([N, H, W, C]).transpose([0, 3, 1, 2])
        
        return y, (x, x_norm, mean, var, eps, gamma, beta, [N, C, H, W])
    
    def backward(self, dy, cache):
        x, x_norm, mean, var, eps, gamma, beta, x_shape = cache

        N, C, H, W = x_shape
        dy = dy.reshape([N, C, H, W])
        dy = dy.transpose([0, 2, 3, 1]).reshape([N * H * W, C])
        
        dx_norm, dgamma, dbeta = self._backward_gamma_beta(dy, (x_norm, gamma, beta))
        
        dx = self._backward_mean_var(dx_norm, (x, x_norm, mean, var, eps))
        
        dx = dx.reshape([N, H, W, C]).transpose([0, 3, 1, 2])
        
        return dx, {'gamma': dgamma, 'beta': dbeta}
    
    
class Spatial_GroupNorm(BatchNorm):
    def __init__(self, C, G, momentum=0.9, eps=1e-5):
        self.C = C
        self.G = G
        
        self.momentum = momentum
        self.eps = eps
        
        self.running_mean = 0
        self.running_var = 0
        
    def forward(self, x, param, mode='train'):
        N, C, H, W = x.shape
        x = x.reshape([N * self.G, C // self.G * H * W]).T
        
        x_norm, cache = self._forward_mean_var(x, param, mode)
        mean, var, eps = cache
        
        x_norm = x_norm.T.reshape([N, C, H, W]).transpose([0, 2, 3, 1]).reshape([N * H * W, C])
        
        y, cache = self._forward_gamma_beta(x_norm, param)
        gamma, beta = cache
        
        y = y.reshape([N, H, W, C]).transpose([0, 3, 1, 2])
        
        return y, (x, x_norm, mean, var, eps, gamma, beta, [N, C, H, W])
    
    def backward(self, dy, cache):
        x, x_norm, mean, var, eps, gamma, beta, x_shape = cache

        N, C, H, W = x_shape
        dy = dy.reshape([N, C, H, W])
        dy = dy.transpose([0, 2, 3, 1]).reshape([N * H * W, C])
        
        dx_norm, dgamma, dbeta = self._backward_gamma_beta(dy, (x_norm, gamma, beta))
        
        x_norm = x_norm.reshape([N, H, W, C]).transpose([0, 3, 1, 2]).reshape([N * self.G, C // self.G * H * W]).T
        dx_norm = dx_norm.reshape([N, H, W, C]).transpose([0, 3, 1, 2]).reshape([N * self.G, C // self.G * H * W]).T
        
        dx = self._backward_mean_var(dx_norm, (x, x_norm, mean, var, eps))
        
        dx = dx.T.reshape([N, C, H, W])
        
        return dx, {'gamma': dgamma, 'beta': dbeta}
    
    
if __name__ == '__main__':
    layer = Conv(3, 2, 2, 2)
    img = np.array([[[[111,112,113], [121,122,123], [131,132,133]], 
                     [[211,212,213], [221,222,223], [231,232,233]], 
                     [[311,312,313], [321,322,323], [331,332,333]]]])
    
    col = layer.img2col(img, 2, 2, 1, 0)
    
    img2 = layer.col2img(col, [1, 3, 3, 3], 2, 2, 1, 0)