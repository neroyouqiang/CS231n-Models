import numpy as np
from models.layers import Linear

class RNN:
    def __init__(self, num_input, num_hidden, init_scale=None):
        self.num_input = num_input
        self.num_hidden = num_hidden
        if init_scale:
            self.init_scale = init_scale
        else:
            self.init_scale = 1. / np.sqrt(self.num_input)
            
    def forward(self, x, h0, param):
        Wx, Wh, b = param['Wx'], param['Wh'], param['b']
        N, T, D = x.shape
        N, H = h0.shape
        
        h = np.zeros([N, T + 1, H])
        h[:, 0, :] = h0
        
        for t in range(0, T, 1):
            h[:, t + 1, :] = self._step_forward(x[:, t, :], h[:, t, :], Wx, Wh, b)
        
        return h[:, 1: T + 1, :], (x, h, Wx, Wh, b)
    
    def forward_step(self, x, h0, param):
        Wx, Wh, b = param['Wx'], param['Wh'], param['b']
        
        h1 = self._step_forward(x, h0, Wx, Wh, b)
        
        return h1
        
    
    def backward(self, dh, cache):
        x, h, Wx, Wh, b = cache
        N, T, D = x.shape
        N, _, H = h.shape
        
        dx = np.zeros([N, T, D])
        dh0 = np.zeros([N, H])
        dWx = np.zeros([D, H])
        dWh = np.zeros([H, H])
        db = np.zeros(H)
        
        for t in range(T - 1, -1, -1):
            cache = (x[:, t, :], h[:, t, :], h[:, t + 1, :], Wx, Wh, b)
            dx[:, t, :], dh0, dWxt, dWht, dbt = self._step_backward(dh0 + dh[:, t, :], cache)
            dWx += dWxt
            dWh += dWht
            db += dbt
            
        return dx, dh0, {'Wx': dWx, 'Wh': dWh, 'b': db}
           
    def _step_forward(self, x, h0, Wx, Wh, b, active='tanh'):
        h1 = np.dot(x, Wx) + np.dot(h0, Wh) + b
        
        if active == 'tanh': 
            h1 = np.tanh(h1)
        else: 
            h1 = h1
        
        return h1
            
    def _step_backward(self, dh1, cache, active='tanh'):
        x, h0, h1, Wx, Wh, b = cache
        
        if active == 'tanh': 
            dh1 = dh1 * (1 - h1 ** 2)
        else: 
            dh1 = dh1
        
        dh0 = np.dot(dh1, Wh.T)
        dx = np.dot(dh1, Wx.T)
        
        dWx = np.dot(x.T, dh1)
        dWh = np.dot(h0.T, dh1)
        db = np.sum(dh1, axis=0)
        
        return dx, dh0, dWx, dWh, db
    
    
    def get_init_param(self):
        Wx = np.random.randn(self.num_input, self.num_hidden) * self.init_scale
        Wh = np.random.randn(self.num_hidden, self.num_hidden) * self.init_scale
        b = np.zeros(self.num_hidden)
        
        return {'Wx': Wx, 'Wh': Wh, 'b': b,
                'info': {'num_input': self.num_input, 
                         'num_output': self.num_hidden,
                         'init_scale': self.init_scale}}
        

class LSTM:
    def __init__(self, num_input, num_hidden, init_scale=None):
        self.num_input = num_input
        self.num_hidden = num_hidden
        if init_scale:
            self.init_scale = init_scale
        else:
            self.init_scale = 1. / np.sqrt(self.num_input / 2.)
            
    def forward(self, x, h0, param, mode='train'):
        Wx, Wh, b = param['Wx'], param['Wh'], param['b']
        N, T, D = x.shape
        N, H = h0.shape
        
        h = np.zeros([N, T + 1, H])
        c = np.zeros([N, T + 1, H])
        h[:, 0, :] = h0
        
        i = np.zeros([N, T, H])
        f = np.zeros([N, T, H])
        o = np.zeros([N, T, H])
        g = np.zeros([N, T, H])
        
        for t in range(0, T, 1):
            h[:, t + 1, :], c[:, t + 1, :], cache = self._step_forward(x[:, t, :], h[:, t, :], c[:, t, :], Wx, Wh, b)
            i[:, t, :], f[:, t, :], o[:, t, :], g[:, t, :] = cache
        
        return h[:, 1: T + 1, :], (x, h, c, i, f, o, g, Wx, Wh, b)
    
    def backward(self, dh, cache):
        x, h, c, i, f, o, g, Wx, Wh, b = cache
        N, T, D = x.shape
        N, _, H = h.shape
        
        dx = np.zeros([N, T, D])
        dh0 = np.zeros([N, H])
        dc0 = np.zeros([N, H])
        dWx = np.zeros([D, H * 4])
        dWh = np.zeros([H, H * 4])
        db = np.zeros(H * 4)
        
        for t in range(T - 1, -1, -1):
            cache = (x[:, t, :], h[:, t, :], h[:, t + 1, :], c[:, t, :], c[:, t + 1, :], 
                     i[:, t, :], f[:, t, :], o[:, t, :], g[:, t, :], Wx, Wh, b)
            dx[:, t, :], dh0, dc0, dWxt, dWht, dbt = self._step_backward(dh0 + dh[:, t, :], dc0, cache)
            dWx += dWxt
            dWh += dWht
            db += dbt
            
        return dx, dh0, {'Wx': dWx, 'Wh': dWh, 'b': db}
    
    def forward_step(self, x, h0, c0, param):
        Wx, Wh, b = param['Wx'], param['Wh'], param['b']
        
        h1, c1, cache = self._step_forward(x, h0, c0, Wx, Wh, b)
        
        return h1, c1
    
    
    def _step_forward(self, x, h0, c0, Wx, Wh, b):
        N, D = x.shape
        N, H = h0.shape
        
        y = np.dot(x, Wx) + np.dot(h0, Wh) + b
        i = y[:, 0: H]
        f = y[:, H: H * 2]
        o = y[:, H * 2: H * 3]
        g = y[:, H * 3: H * 4]
        
        i = self.sigmoid(i)
        f = self.sigmoid(f)
        o = self.sigmoid(o)
        g = np.tanh(g)
        
        c1 = f * c0 + i * g
        h1 = o * np.tanh(c1)
#        h1 = i + f
#        c1 = o + g + c0
        
        return h1, c1, (i, f, o, g)
            
    def _step_backward(self, dh1, dc1, cache, active='tanh'):
        x, h0, h1, c0, c1, i, f, o, g, Wx, Wh, b = cache
        
        c1_tanh = np.tanh(c1)
        
        do = dh1 * c1_tanh
        dc1 = dh1 * o * (1 - c1_tanh ** 2) + dc1
        
        dc0 = dc1 * f
        df = dc1 * c0
        di = dc1 * g
        dg = dc1 * i
        
        di = di * i * (1 - i)
        df = df * f * (1 - f)
        do = do * o * (1 - o)
        dg = dg * (1 - g ** 2)
        
        dy = np.concatenate((di, df, do, dg), axis=1)
        db = np.sum(dy, axis=0)
        dWx = np.dot(x.T, dy)
        dWh = np.dot(h0.T, dy)
        
        dx = np.dot(dy, Wx.T)
        dh0 = np.dot(dy, Wh.T)
        
        return dx, dh0, dc0, dWx, dWh, db
            
    def sigmoid(self, x):
        """
        A numerically stable version of the logistic sigmoid function.
        """
        pos_mask = (x >= 0)
        neg_mask = (x < 0)
        z = np.zeros_like(x)
        z[pos_mask] = np.exp(-x[pos_mask])
        z[neg_mask] = np.exp(x[neg_mask])
        top = np.ones_like(x)
        top[neg_mask] = z[neg_mask]
        return top / (1 + z)

    def get_init_param(self):
        Wx = np.random.randn(self.num_input, self.num_hidden * 4) * self.init_scale
        Wh = np.random.randn(self.num_hidden, self.num_hidden * 4) * self.init_scale
        b = np.zeros(self.num_hidden * 4)
        
        return {'Wx': Wx, 'Wh': Wh, 'b': b,
                'info': {'num_input': self.num_input, 
                         'num_output': self.num_hidden,
                         'init_scale': self.init_scale}}
        

class LinearForRNN(Linear):
    def __init__(self, num_input, num_output, init_scale=None):
        self.num_input = num_input
        self.num_output = num_output
        if init_scale:
            self.init_scale = init_scale
        else:
            self.init_scale = 1. / np.sqrt(self.num_input)
    
    def forward(self, x, param, mode='train'):
        W, b = param['W'], param['b']
        if mode == 'train':
            N, T, D = x.shape
            
            x = x.reshape(N * T, D)
            y = np.dot(x, W) + b
            y = y.reshape(N, T, -1)
                          
        elif mode == 'test':
            N, D = x.shape
            
            x = x.reshape(N, D)
            y = np.dot(x, W) + b
            y = y.reshape(N, -1)
        
        return y, (x, W)
    
    def backward(self, dy, cache):
        x, W = cache
        N, T, M = dy.shape
        
        dy = dy.reshape(N * T, M)
        
        db = np.sum(dy, axis=0)
        dW = np.dot(x.T, dy)
        dx = np.dot(dy, W.T)
        
        dx = dx.reshape(N, T, -1)
        
        return dx, {'W': dW, 'b': db}
        

class WordEmbedding:
    def __init__(self, num_word, num_vector, init_scale=None):
        self.num_word = num_word
        self.num_vector = num_vector
        if init_scale:
            self.init_scale = init_scale
        else:
            self.init_scale = 0.01
            
    
    def forward(self, x, param, mode='train'):
        W_embed = param['W_embed']
        
        y = W_embed[x]
        
        return y, (x, W_embed.shape)
    
    
    def backward(self, dy, cache):
        x, W_shape = cache
        
        dW_embed = np.zeros(W_shape)
        np.add.at(dW_embed, x, dy)
        
        return dy, {'W_embed': dW_embed}
    
            
    def get_init_param(self):
        W_embed = np.random.randn(self.num_word, self.num_vector) * self.init_scale
        
        return {'W_embed': W_embed,
                'info': {'num_word': self.num_word, 
                         'num_vector': self.num_vector,
                         'init_scale': self.init_scale}}
        
        
            
            